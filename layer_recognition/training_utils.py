from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from typing import Iterable

import numpy as np
import torch

from .features import LayerFeature


class StandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_train_val_split(
    features: list[LayerFeature],
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({feature.group_id for feature in features})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_count = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[:val_count])
    train_idx, val_idx = [], []
    for index, feature in enumerate(features):
        if feature.group_id in val_groups:
            val_idx.append(index)
        else:
            train_idx.append(index)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in y)
    present = len(counts)
    total = len(y)
    return np.asarray([total / (present * counts[int(value)]) for value in y], dtype=np.float32)


def layer_indices(y: np.ndarray, layer_min: int) -> np.ndarray:
    return y.astype(np.int64) - int(layer_min)


def rounded_clipped_layers(scores: np.ndarray, layer_min: int, layer_max: int) -> np.ndarray:
    return np.clip(np.rint(scores), layer_min, layer_max).astype(np.int64)


def compute_metrics(
    y_true: np.ndarray,
    pred_scores: np.ndarray,
    pred_layers: np.ndarray,
    layer_min: int,
    layer_max: int,
) -> dict:
    y_true = y_true.astype(np.int64)
    pred_layers = pred_layers.astype(np.int64)
    residual = y_true.astype(np.float32) - pred_scores.astype(np.float32)
    metrics = {
        "count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == pred_layers)) if len(y_true) else float("nan"),
        "within_1": float(np.mean(np.abs(y_true - pred_layers) <= 1)) if len(y_true) else float("nan"),
        "mae_score": float(np.mean(np.abs(residual))) if len(y_true) else float("nan"),
        "rmse_score": float(np.sqrt(np.mean(residual**2))) if len(y_true) else float("nan"),
        "per_layer": {},
    }
    for layer in range(layer_min, layer_max + 1):
        mask = y_true == layer
        if not np.any(mask):
            continue
        metrics["per_layer"][str(layer)] = {
            "count": int(np.count_nonzero(mask)),
            "accuracy": float(np.mean(pred_layers[mask] == y_true[mask])),
            "within_1": float(np.mean(np.abs(pred_layers[mask] - y_true[mask]) <= 1)),
        }
    return metrics


def write_json(payload: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_predictions_csv(
    features: list[LayerFeature],
    pred_scores: np.ndarray,
    pred_layers: np.ndarray,
    out_path: str,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    extra_columns = extra_columns or {}
    fieldnames = [
        "image_id",
        "ann_id",
        "file_name",
        "group_id",
        "layer_gt",
        "predicted_score",
        "predicted_layer",
        "normalized_delta_g",
        "wb1",
        "wb2",
        "delta_g",
        "area_px",
        *extra_columns.keys(),
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, feature in enumerate(features):
            row = {
                "image_id": feature.image_id,
                "ann_id": feature.ann_id,
                "file_name": feature.file_name,
                "group_id": feature.group_id,
                "layer_gt": feature.layer_gt,
                "predicted_score": float(pred_scores[index]),
                "predicted_layer": int(pred_layers[index]),
                "normalized_delta_g": feature.normalized_delta_g,
                "wb1": feature.wb1,
                "wb2": feature.wb2,
                "delta_g": feature.delta_g,
                "area_px": feature.area_px,
            }
            for name, values in extra_columns.items():
                row[name] = float(values[index])
            writer.writerow(row)


def dataclass_rows(features: Iterable[LayerFeature]) -> list[dict]:
    return [asdict(feature) for feature in features]

