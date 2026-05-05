from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from layer_recognition.green_ordinal import (
    ExtractConfig,
    extract_one_annotation_features,
    normalize_flake_rep_mode,
    parse_group_from_filename,
    parse_wb_from_filename,
)
from maskterial.structures.FlakeClass import Flake


DEFAULT_LAYER_MODEL_DIR = "training_log/green_ordinal_largest_inner_median_v10"


@dataclass(slots=True)
class LayerPrediction:
    flake_index: int
    status: str
    final_label: str
    category_key: str | None
    raw_pred_layer: int | None
    layer_score: float | None
    review: bool | None
    wb1: float
    wb2: float
    feature_values: dict[str, Any]
    reason: str = ""

    def csv_row(self) -> dict[str, Any]:
        row = {
            "flake_index": self.flake_index,
            "status": self.status,
            "pred_layer": self.final_label,
            "raw_pred_layer": self.raw_pred_layer if self.raw_pred_layer is not None else "",
            "layer_score": self.layer_score if self.layer_score is not None else "",
            "review": self.review if self.review is not None else "",
            "wb1": self.wb1,
            "wb2": self.wb2,
            "reason": self.reason,
        }
        row.update(self.feature_values)
        return row


def expand_path(path: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def _load_model_config(model_dir: Path) -> dict:
    path = model_dir / "model_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Layer model config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_config_from_model_config(model_config: dict) -> ExtractConfig:
    args_payload = model_config.get("args", {})
    kwargs = {}
    for field_name in ExtractConfig.__dataclass_fields__:
        if field_name in args_payload and args_payload[field_name] is not None:
            kwargs[field_name] = args_payload[field_name]
    cfg = ExtractConfig(**kwargs)
    cfg.flake_rep_mode = normalize_flake_rep_mode(cfg.flake_rep_mode)
    return cfg


def _coerce_train_medians(model, model_config: dict) -> dict[int, float]:
    medians = getattr(model, "train_medians_", None) or model_config.get("train_score_medians", {})
    output = {}
    for key, value in medians.items():
        output[int(key)] = float(value)
    required = {1, 2, 6, 7}
    missing = sorted(required - set(output))
    if missing:
        raise ValueError(f"Layer model train medians missing required layers: {missing}")
    return output


class LayerRecognitionRunner:
    def __init__(
        self,
        model_dir: str | os.PathLike = DEFAULT_LAYER_MODEL_DIR,
        boundary_margin: float | None = None,
    ) -> None:
        self.model_dir = expand_path(model_dir)
        model_path = self.model_dir / "green_ordinal_model.joblib"
        if not model_path.exists():
            model_path = self.model_dir / "green_ordinal_model_final.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No green ordinal joblib model found under {self.model_dir}")

        self.model = joblib.load(model_path)
        self.model_config = _load_model_config(self.model_dir)
        self.extract_config = _extract_config_from_model_config(self.model_config)
        args_payload = self.model_config.get("args", {})
        self.boundary_margin = (
            float(boundary_margin)
            if boundary_margin is not None
            else float(args_payload.get("boundary_margin", 0.2))
        )

        self.train_medians = _coerce_train_medians(self.model, self.model_config)
        self.background_score_threshold = self.train_medians[1] - 0.5 * (
            self.train_medians[2] - self.train_medians[1]
        )
        self.gt7_score_threshold = self.train_medians[7] + 0.5 * (
            self.train_medians[7] - self.train_medians[6]
        )
        self.interpolated_layer5_score = 0.5 * (self.train_medians[4] + self.train_medians[6])
        self.layer45_score_threshold = 0.5 * (self.train_medians[4] + self.interpolated_layer5_score)
        self.layer56_score_threshold = 0.5 * (self.interpolated_layer5_score + self.train_medians[6])

    def threshold_info(self) -> dict[str, Any]:
        return {
            "background_score_threshold": float(self.background_score_threshold),
            "gt7_score_threshold": float(self.gt7_score_threshold),
            "interpolated_layer5_score": float(self.interpolated_layer5_score),
            "layer45_score_threshold": float(self.layer45_score_threshold),
            "layer56_score_threshold": float(self.layer56_score_threshold),
            "train_score_medians": {str(key): float(value) for key, value in self.train_medians.items()},
            "boundary_margin": float(self.boundary_margin),
            "valid_supervised_layers": list(getattr(self.model, "config", lambda: {})().get("classes", []))
            if hasattr(self.model, "config")
            else self.model_config.get("classes", []),
            "interpolated_layers": [5],
        }

    def predict_image(
        self,
        image_bgr: np.ndarray,
        filename: str,
        image_path: str,
        image_id: int,
        flakes: list[Flake],
    ) -> list[LayerPrediction]:
        if not flakes:
            return []

        masks = [(np.asarray(flake.mask) > 0) for flake in flakes]
        union_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            union_mask |= mask

        wb1, wb2 = parse_wb_from_filename(filename)
        group = parse_group_from_filename(filename)
        predictions = []

        for index, mask in enumerate(masks, start=1):
            feature_values = extract_one_annotation_features(
                image_bgr=image_bgr,
                annotation_mask=mask,
                union_mask=union_mask,
                cfg=self.extract_config,
            )
            if feature_values is None:
                predictions.append(
                    LayerPrediction(
                        flake_index=index,
                        status="feature_failed",
                        final_label="",
                        category_key=None,
                        raw_pred_layer=None,
                        layer_score=None,
                        review=None,
                        wb1=float(wb1),
                        wb2=float(wb2),
                        feature_values={},
                        reason="green feature extraction returned no row",
                    )
                )
                continue

            row = {
                "split": "infer",
                "filename": filename,
                "image_path": image_path,
                "image_id": int(image_id),
                "ann_id": int(index),
                "group": group,
                "flake_rep_mode": self.extract_config.flake_rep_mode,
                "wb1": float(wb1),
                "wb2": float(wb2),
                "wb_product": float(wb1 * wb2) if np.isfinite(wb1) and np.isfinite(wb2) else float("nan"),
                "wb_sum": float(wb1 + wb2) if np.isfinite(wb1) and np.isfinite(wb2) else float("nan"),
            }
            row.update(feature_values)

            score = float(self.model.score_samples([row])[0])
            raw_pred_layer = int(self.model.predict([row])[0])
            review = bool(self.model.review_flags([row], boundary_margin=self.boundary_margin)[0])

            if score < self.background_score_threshold:
                status = "background"
                final_label = "background"
                category_key = None
                layer_source = "background_score_below_layer1"
            elif score > self.gt7_score_threshold:
                status = "layer"
                final_label = ">7"
                category_key = ">7"
                layer_source = "extrapolated_gt7"
            elif self.layer45_score_threshold <= score < self.layer56_score_threshold:
                status = "layer"
                final_label = "5"
                category_key = "5"
                layer_source = "interpolated_missing_5"
            else:
                status = "layer"
                final_label = str(raw_pred_layer)
                category_key = str(raw_pred_layer)
                layer_source = "ordinal_model"

            predictions.append(
                LayerPrediction(
                    flake_index=index,
                    status=status,
                    final_label=final_label,
                    category_key=category_key,
                    raw_pred_layer=raw_pred_layer,
                    layer_score=score,
                    review=review,
                    wb1=float(wb1),
                    wb2=float(wb2),
                    feature_values={
                        "ndg": float(row.get("ndg", float("nan"))),
                        "iqr_ndg": float(row.get("iqr_ndg", float("nan"))),
                        "delta_g": float(row.get("delta_g", float("nan"))),
                        "g_flake_rep": float(row.get("g_flake_rep", float("nan"))),
                        "g_bg_plane_median": float(row.get("g_bg_plane_median", float("nan"))),
                        "flake_rep_source": str(row.get("flake_rep_source", "")),
                        "roi_bg_pixels": int(row.get("roi_bg_pixels", 0)),
                        "layer_source": layer_source,
                    },
                )
            )

        return predictions
