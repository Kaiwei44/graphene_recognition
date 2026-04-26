from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch import nn

from layer_recognition.features import (
    DEFAULT_FEATURE_NAMES,
    extract_features_from_coco,
    feature_matrix,
    labels_array,
    save_features_csv,
)
from layer_recognition.training_utils import (
    StandardScaler,
    class_balanced_weights,
    compute_metrics,
    group_train_val_split,
    layer_indices,
    set_seed,
    write_json,
    write_predictions_csv,
)


class CoralOrdinalMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.rank_weight = nn.Linear(dim, 1, bias=False)
        self.rank_bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.rank_weight(h) + self.rank_bias


def ordinal_targets(y_layers: np.ndarray, layer_min: int, num_classes: int) -> torch.Tensor:
    y_index = layer_indices(y_layers, layer_min)
    thresholds = np.arange(num_classes - 1, dtype=np.int64)
    return torch.from_numpy((y_index[:, None] > thresholds[None, :]).astype(np.float32))


@torch.no_grad()
def predict(model: CoralOrdinalMLP, x_scaled: np.ndarray, layer_min: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = model(torch.from_numpy(x_scaled).float())
    probs = torch.sigmoid(logits).cpu().numpy()
    pred_index = (probs >= 0.5).sum(axis=1)
    pred_layers = pred_index.astype(np.int64) + layer_min
    pred_scores = layer_min + probs.sum(axis=1)
    return pred_scores.astype(np.float32), pred_layers


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    feature_names = tuple(name.strip() for name in args.features.split(",") if name.strip())
    num_classes = args.layer_max - args.layer_min + 1

    train_features = extract_features_from_coco(
        args.train_image_dir,
        args.train_coco,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        roi_scale=args.roi_scale,
    )
    if args.test_image_dir and args.test_coco:
        test_features = extract_features_from_coco(
            args.test_image_dir,
            args.test_coco,
            layer_min=args.layer_min,
            layer_max=args.layer_max,
            roi_scale=args.roi_scale,
        )
    else:
        test_features = []

    save_features_csv(train_features, os.path.join(args.save_dir, "train_features.csv"))
    if test_features:
        save_features_csv(test_features, os.path.join(args.save_dir, "test_features.csv"))

    x_all = feature_matrix(train_features, feature_names)
    y_all = labels_array(train_features)
    train_idx, val_idx = group_train_val_split(train_features, args.val_fraction, args.seed)

    scaler = StandardScaler().fit(x_all[train_idx])
    x_train = scaler.transform(x_all[train_idx])
    x_val = scaler.transform(x_all[val_idx])
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    train_targets = ordinal_targets(y_train, args.layer_min, num_classes)
    val_targets = ordinal_targets(y_val, args.layer_min, num_classes)
    train_weights = torch.from_numpy(class_balanced_weights(y_train))

    pos = train_targets.sum(dim=0)
    neg = train_targets.shape[0] - pos
    pos_weight = torch.where((pos > 0) & (neg > 0), neg / (pos + 1e-6), torch.ones_like(pos))
    pos_weight = torch.clamp(pos_weight, 0.25, 4.0)

    model = CoralOrdinalMLP(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    x_train_t = torch.from_numpy(x_train).float()
    x_val_t = torch.from_numpy(x_val).float()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss_per_rank = bce(logits, train_targets)
        loss = (loss_per_rank.mean(dim=1) * train_weights).mean()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val_t)
                val_loss = bce(val_logits, val_targets).mean().item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += args.eval_interval
            if epoch % (args.eval_interval * 10) == 0:
                print(f"epoch={epoch} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_scores, train_pred = predict(model, scaler.transform(x_all), args.layer_min)
    train_metrics = compute_metrics(y_all, train_scores, train_pred, args.layer_min, args.layer_max)
    write_predictions_csv(train_features, train_scores, train_pred, os.path.join(args.save_dir, "train_predictions.csv"))

    payload = {
        "model_type": "coral_ordinal_mlp",
        "feature_names": list(feature_names),
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "scaler": scaler.to_dict(),
        "train_metrics": train_metrics,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }

    if test_features:
        x_test = scaler.transform(feature_matrix(test_features, feature_names))
        y_test = labels_array(test_features)
        test_scores, test_pred = predict(model, x_test, args.layer_min)
        test_metrics = compute_metrics(y_test, test_scores, test_pred, args.layer_min, args.layer_max)
        payload["test_metrics"] = test_metrics
        write_predictions_csv(test_features, test_scores, test_pred, os.path.join(args.save_dir, "test_predictions.csv"))
        print(f"Test accuracy={test_metrics['accuracy']:.3f}, within_1={test_metrics['within_1']:.3f}, MAE={test_metrics['mae_score']:.3f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "ordinal_mlp.pt"))
    write_json(payload, os.path.join(args.save_dir, "ordinal_mlp_meta.json"))
    print(f"Saved ordinal MLP outputs to {args.save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CORAL-style ordinal MLP for graphene layer recognition.")
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-coco", required=True)
    parser.add_argument("--test-image-dir")
    parser.add_argument("--test-coco")
    parser.add_argument("--save-dir", default="training_log/layer_ordinal_mlp_0_10")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURE_NAMES))
    parser.add_argument("--layer-min", type=int, default=0)
    parser.add_argument("--layer-max", type=int, default=10)
    parser.add_argument("--roi-scale", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

