from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import torch
from torch import nn

from layer_recognition.baselines import PiecewiseLinearBaseline, fit_piecewise_baseline
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
    rounded_clipped_layers,
    set_seed,
    write_json,
    write_predictions_csv,
)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, depth: int = 2, dropout: float = 0.1, residual_bound: float = 2.0):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.residual_bound = float(residual_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_bound * torch.tanh(self.net(x)).squeeze(-1)


def residual_inputs(x: np.ndarray, base_score: np.ndarray, gate_prob: np.ndarray) -> np.ndarray:
    return np.column_stack((x, base_score.astype(np.float32), gate_prob.astype(np.float32))).astype(np.float32)


@torch.no_grad()
def predict(
    baseline: PiecewiseLinearBaseline,
    model: ResidualMLP,
    scaler: StandardScaler,
    x_raw: np.ndarray,
    layer_min: int,
    layer_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    base = baseline.predict_score(x_raw)
    gate_prob = baseline.gate_probability(x_raw)
    r_input = scaler.transform(residual_inputs(x_raw, base, gate_prob))
    residual = model(torch.from_numpy(r_input).float()).cpu().numpy()
    score = np.clip(base + residual, layer_min, layer_max)
    layer = rounded_clipped_layers(score, layer_min, layer_max)
    return score.astype(np.float32), layer, base.astype(np.float32), residual.astype(np.float32)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    feature_names = tuple(name.strip() for name in args.features.split(",") if name.strip())

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

    baseline = fit_piecewise_baseline(x_all[train_idx], y_all[train_idx], args.split_layer)
    base_train_all = baseline.predict_score(x_all)
    gate_train_all = baseline.gate_probability(x_all)
    residual_target_all = y_all.astype(np.float32) - base_train_all

    residual_scaler = StandardScaler().fit(residual_inputs(x_all[train_idx], base_train_all[train_idx], gate_train_all[train_idx]))
    x_train = residual_scaler.transform(residual_inputs(x_all[train_idx], base_train_all[train_idx], gate_train_all[train_idx]))
    x_val = residual_scaler.transform(residual_inputs(x_all[val_idx], base_train_all[val_idx], gate_train_all[val_idx]))
    y_train_residual = residual_target_all[train_idx]
    y_val_residual = residual_target_all[val_idx]
    train_weights = torch.from_numpy(class_balanced_weights(y_all[train_idx]))

    model = ResidualMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        residual_bound=args.residual_bound,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.HuberLoss(reduction="none", delta=1.0)

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train_residual).float()
    x_val_t = torch.from_numpy(x_val).float()
    y_val_t = torch.from_numpy(y_val_residual).float()

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_residual = model(x_train_t)
        loss = (huber(pred_residual, y_train_t) * train_weights).mean()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_loss = huber(model(x_val_t), y_val_t).mean().item()
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

    train_scores, train_pred, train_base, train_residual = predict(
        baseline, model, residual_scaler, x_all, args.layer_min, args.layer_max
    )
    train_metrics = compute_metrics(y_all, train_scores, train_pred, args.layer_min, args.layer_max)
    write_predictions_csv(
        train_features,
        train_scores,
        train_pred,
        os.path.join(args.save_dir, "train_predictions.csv"),
        extra_columns={"base_score": train_base, "mlp_residual": train_residual},
    )

    payload = {
        "model_type": "piecewise_linear_plus_bounded_residual_mlp",
        "feature_names": list(feature_names),
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "split_layer": args.split_layer,
        "residual_scaler": residual_scaler.to_dict(),
        "train_metrics": train_metrics,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }

    if test_features:
        x_test = feature_matrix(test_features, feature_names)
        y_test = labels_array(test_features)
        test_scores, test_pred, test_base, test_residual = predict(
            baseline, model, residual_scaler, x_test, args.layer_min, args.layer_max
        )
        test_metrics = compute_metrics(y_test, test_scores, test_pred, args.layer_min, args.layer_max)
        payload["test_metrics"] = test_metrics
        write_predictions_csv(
            test_features,
            test_scores,
            test_pred,
            os.path.join(args.save_dir, "test_predictions.csv"),
            extra_columns={"base_score": test_base, "mlp_residual": test_residual},
        )
        print(f"Test accuracy={test_metrics['accuracy']:.3f}, within_1={test_metrics['within_1']:.3f}, MAE={test_metrics['mae_score']:.3f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "residual_mlp.pt"))
    joblib.dump(baseline, os.path.join(args.save_dir, "piecewise_linear_baseline.joblib"))
    write_json(payload, os.path.join(args.save_dir, "residual_mlp_meta.json"))
    print(f"Saved residual MLP outputs to {args.save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train piecewise linear baseline plus bounded residual MLP for layer recognition.")
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-coco", required=True)
    parser.add_argument("--test-image-dir")
    parser.add_argument("--test-coco")
    parser.add_argument("--save-dir", default="training_log/layer_residual_mlp_0_10")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURE_NAMES))
    parser.add_argument("--layer-min", type=int, default=0)
    parser.add_argument("--layer-max", type=int, default=10)
    parser.add_argument("--split-layer", type=int, default=5)
    parser.add_argument("--roi-scale", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-bound", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
