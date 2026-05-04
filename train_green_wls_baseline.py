from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from layer_recognition.green_ordinal import (
    ExtractConfig,
    FLAKE_REP_MODES,
    VALID_LAYERS,
    class_counts,
    coerce_rows,
    expand_path,
    extract_features,
    normalize_flake_rep_mode,
    read_csv,
    rows_to_dicts,
    split_counts,
    write_csv,
    write_json,
)


DEFAULT_BASELINE_FEATURES = "ndg_peak,wb1"


def parse_feature_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["layer"]) for row in rows], dtype=np.int64)


def feature_matrix(rows: list[dict], feature_cols: list[str]) -> np.ndarray:
    missing = [column for column in feature_cols if rows and column not in rows[0]]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return np.asarray([[float(row[column]) for column in feature_cols] for row in rows], dtype=np.float64)


def nearest_valid_layers(scores: np.ndarray) -> np.ndarray:
    distances = np.abs(scores[:, None] - VALID_LAYERS[None, :])
    return VALID_LAYERS[np.argmin(distances, axis=1)].astype(np.int64)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    abs_error = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "exact_acc": float(accuracy_score(y_true, y_pred)),
        "within1_acc": float(np.mean(abs_error <= 1)),
        "large_error_rate": float(np.mean(abs_error > 1)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "score_mae": float(mean_absolute_error(y_true.astype(float), scores)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=VALID_LAYERS, average="macro", zero_division=0)),
    }


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=VALID_LAYERS)
    output = []
    for layer, values in zip(VALID_LAYERS, matrix):
        row = {"true_layer": int(layer)}
        for pred_layer, count in zip(VALID_LAYERS, values):
            row[f"pred_{int(pred_layer)}"] = int(count)
        output.append(row)
    return output


def prediction_rows(rows: list[dict], scores: np.ndarray, pred: np.ndarray) -> list[dict]:
    output = []
    for row, score, predicted in zip(rows, scores, pred):
        item = dict(row)
        item["score"] = float(score)
        item["pred_layer"] = int(predicted)
        item["abs_error"] = float(abs(int(row["layer"]) - int(predicted)))
        output.append(item)
    return output


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def load_or_extract_rows(args: argparse.Namespace, out_dir: Path) -> list[dict]:
    feature_cache = expand_path(args.feature_cache) if args.feature_cache else out_dir / "features.csv"
    if feature_cache.exists() and not args.force_extract:
        print(f"[INFO] Loading cached features: {feature_cache}")
        rows = coerce_rows(read_csv(feature_cache))
    else:
        cfg = ExtractConfig(
            roi_scale=args.roi_scale,
            min_area_px=args.min_area_px,
            bg_min_pixels=args.bg_min_pixels,
            bg_sample_max=args.bg_sample_max,
            edge_dilate_px=args.edge_dilate_px,
            bg_residual_clip_sigma=args.bg_residual_clip_sigma,
            bg_residual_clip_iters=args.bg_residual_clip_iters,
            bg_residual_sigma_floor=args.bg_residual_sigma_floor,
            flake_rep_mode=args.flake_rep_mode,
            flake_peak_bins=args.flake_peak_bins,
            flake_inner_erode_px=args.flake_inner_erode_px,
            flake_min_inner_pixels=args.flake_min_inner_pixels,
            trim_low=args.trim_low,
            trim_high=args.trim_high,
        )
        rows = rows_to_dicts(
            extract_features(
                cfg=cfg,
                coco_dir=args.coco_dir,
                train_image_dir=args.train_image_dir,
                train_coco=args.train_coco,
                valid_image_dir=args.valid_image_dir,
                valid_coco=args.valid_coco,
                test_image_dir=args.test_image_dir,
                test_coco=args.test_coco,
            )
        )
        rows = coerce_rows(rows)
        write_csv(rows, feature_cache)
        print(f"[INFO] Saved features: {feature_cache}")

    valid_layer_set = set(int(layer) for layer in VALID_LAYERS.tolist())
    rows = [row for row in rows if int(row["layer"]) in valid_layer_set]
    if not rows:
        raise RuntimeError(f"No usable rows after filtering layers {sorted(valid_layer_set)}")
    return rows


def make_model(args: argparse.Namespace) -> Pipeline:
    if args.regressor == "wls":
        regressor = LinearRegression()
    elif args.regressor == "ridge":
        regressor = Ridge(alpha=args.alpha, random_state=args.seed)
    else:
        raise ValueError(args.regressor)

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if args.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", regressor))
    return Pipeline(steps)


def run(args: argparse.Namespace) -> None:
    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_or_extract_rows(args, out_dir)
    print("[INFO] Layer counts by split:")
    for split, counts in split_counts(rows).items():
        print(f"  {split}: {counts}")
    print(f"[INFO] Overall class counts: {class_counts(rows)}")

    train_rows = [row for row in rows if str(row["split"]).lower() in {"train", "valid", "val"}]
    test_rows = [row for row in rows if str(row["split"]).lower() == "test"]
    if not train_rows or not test_rows:
        raise RuntimeError("Need non-empty train/valid and test splits for baseline comparison")

    feature_cols = parse_feature_list(args.features)
    x_train = feature_matrix(train_rows, feature_cols)
    y_train = labels(train_rows)
    x_test = feature_matrix(test_rows, feature_cols)
    y_test = labels(test_rows)

    model = make_model(args)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train) if args.class_balanced else None
    model.fit(x_train, y_train.astype(float), model__sample_weight=sample_weight)

    train_scores = model.predict(x_train).astype(float)
    train_pred = nearest_valid_layers(train_scores)
    train_metrics = evaluate_predictions(y_train, train_pred, train_scores)

    test_scores = model.predict(x_test).astype(float)
    test_pred = nearest_valid_layers(test_scores)
    test_metrics = evaluate_predictions(y_test, test_pred, test_scores)

    print_metrics("WLS baseline train self-fit", train_metrics)
    print_metrics("WLS baseline Roboflow test", test_metrics)

    write_csv(prediction_rows(train_rows, train_scores, train_pred), out_dir / "train_predictions.csv")
    write_csv(prediction_rows(test_rows, test_scores, test_pred), out_dir / "test_predictions.csv")
    write_csv(confusion_rows(y_train, train_pred), out_dir / "train_confusion_matrix.csv")
    write_csv(confusion_rows(y_test, test_pred), out_dir / "test_confusion_matrix.csv")
    write_json(train_metrics, out_dir / "train_metrics.json")
    write_json(test_metrics, out_dir / "test_metrics.json")
    write_csv(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ],
        out_dir / "summary_metrics.csv",
    )

    regressor = model.named_steps["model"]
    coef = np.asarray(getattr(regressor, "coef_", []), dtype=float).ravel()
    coefficient_rows = [{"feature": feature, "coefficient": float(value)} for feature, value in zip(feature_cols, coef)]
    coefficient_rows.append({"feature": "intercept", "coefficient": float(getattr(regressor, "intercept_", 0.0))})
    write_csv(coefficient_rows, out_dir / "coefficients.csv")

    payload = {
        "model_name": "Green channel WLS baseline",
        "regressor": args.regressor,
        "alpha": float(args.alpha),
        "feature_cols": feature_cols,
        "classes": VALID_LAYERS.tolist(),
        "prediction_rule": "nearest valid layer from continuous regression score",
        "class_balanced": bool(args.class_balanced),
        "standardize": bool(args.standardize),
        "args": vars(args),
    }
    write_json(payload, out_dir / "model_config.json")
    joblib.dump({"model": model, "feature_cols": feature_cols, "classes": VALID_LAYERS}, out_dir / "wls_baseline_model.joblib")
    print(f"[INFO] Saved baseline outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple green-channel WLS baseline for layer recognition.")
    parser.add_argument("--coco-dir", type=str, default=None)
    parser.add_argument("--train-image-dir", type=str, default=None)
    parser.add_argument("--train-coco", type=str, default=None)
    parser.add_argument("--valid-image-dir", type=str, default=None)
    parser.add_argument("--valid-coco", type=str, default=None)
    parser.add_argument("--test-image-dir", type=str, default=None)
    parser.add_argument("--test-coco", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--feature-cache", type=str, default=None)
    parser.add_argument("--force-extract", action="store_true")

    parser.add_argument("--features", type=str, default=DEFAULT_BASELINE_FEATURES)
    parser.add_argument("--regressor", choices=["wls", "ridge"], default="wls")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-balanced", dest="class_balanced", action="store_false")
    parser.set_defaults(class_balanced=True)
    parser.add_argument("--standardize", action="store_true", help="Standardize features before regression.")

    parser.add_argument("--roi-scale", type=float, default=4.0)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--bg-min-pixels", type=int, default=100)
    parser.add_argument("--edge-dilate-px", type=int, default=3)
    parser.add_argument("--bg-sample-max", type=int, default=20000)
    parser.add_argument("--bg-residual-clip-sigma", type=float, default=2.5)
    parser.add_argument("--bg-residual-clip-iters", type=int, default=2)
    parser.add_argument("--bg-residual-sigma-floor", type=float, default=1.0)
    parser.add_argument(
        "--flake-rep-mode",
        default="largest_inner_median",
        help=f"Feature extraction representative. Valid: {', '.join(FLAKE_REP_MODES)}.",
    )
    parser.add_argument("--flake-peak-bins", type=int, default=50)
    parser.add_argument("--flake-inner-erode-px", type=int, default=2)
    parser.add_argument("--flake-min-inner-pixels", type=int, default=20)
    parser.add_argument("--trim-low", type=float, default=10.0)
    parser.add_argument("--trim-high", type=float, default=90.0)

    args = parser.parse_args()
    args.flake_rep_mode = normalize_flake_rep_mode(args.flake_rep_mode)
    return args


if __name__ == "__main__":
    run(parse_args())
