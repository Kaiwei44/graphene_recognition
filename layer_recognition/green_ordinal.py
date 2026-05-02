from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


VALID_LAYERS = np.array([1, 2, 3, 4, 6, 7], dtype=np.int64)
ORDINAL_BOUNDARIES = VALID_LAYERS[1:].copy()

LOSS_MARGIN = 0.30
LABEL_SMOOTHING = 0.00
L2_SCALE = 1e-3
THRESHOLD_EPS = 1e-3
OPT_MAX_ITER = 2000

BOUNDARY_WEIGHT_MAP = {
    2: 1.0,
    3: 1.4,
    4: 1.5,
    6: 1.0,
    7: 1.4,
}

COMPACT_BASE_FEATURES = ["ndg", "iqr_ndg", "wb1", "wb2"]

COMPACT_PHI_FEATURES = [
    "z_ndg",
    "z_ndg2",
    "z_iqr",
    "z_wb1",
    "z_wb2",
    "z_ndg_wb1",
    "z_ndg_wb2",
]


@dataclass(slots=True)
class ExtractConfig:
    roi_scale: float = 3.0
    min_area_px: int = 20
    bg_min_pixels: int = 100
    bg_sample_max: int = 20000
    edge_dilate_px: int = 5
    trim_low: float = 10.0
    trim_high: float = 90.0


@dataclass(slots=True)
class GreenFeatureRow:
    split: str
    filename: str
    image_path: str
    image_id: int
    ann_id: int
    layer: int
    group: str
    wb1: float
    wb2: float
    wb_product: float
    wb_sum: float
    area_px: int
    log_area_px: float
    bbox_w: float
    bbox_h: float
    bbox_aspect: float
    roi_bg_pixels: int
    ndg: float
    delta_g: float
    abs_delta_g: float
    g_flake_median: float
    g_bg_plane_median: float
    g_delta_p10: float
    g_delta_p90: float
    g_delta_iqr: float
    iqr_ndg: float
    g_bg_std: float
    g_flake_std: float
    plane_a: float
    plane_b: float
    plane_c: float


class CompactOrdinalBCE:
    """
    Compact Green Ordinal Boundary BCE, Version C.

    Base features:
        ndg, iqr_ndg, wb1, wb2

    After standardization, fixed feature map:
        z_ndg
        z_ndg^2
        z_iqr
        z_wb1
        z_wb2
        z_ndg * z_wb1
        z_ndg * z_wb2

    Learned parameters:
        8 score parameters: intercept + 7 coefficients
        5 learned monotone ordinal thresholds
    """

    def __init__(self, alpha: float = 10.0, random_state: int = 42):
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.feature_medians_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.raw_thresholds_: np.ndarray | None = None
        self.thresholds_: np.ndarray | None = None
        self.train_medians_: dict[int, float] | None = None
        self.boundary_weights_: np.ndarray = np.asarray(
            [BOUNDARY_WEIGHT_MAP[int(boundary)] for boundary in ORDINAL_BOUNDARIES],
            dtype=np.float64,
        )
        self.loss_margin_: float = LOSS_MARGIN
        self.label_smoothing_: float = LABEL_SMOOTHING
        self.gamma_: float = self.alpha * L2_SCALE
        self.optimization_: dict | None = None

    @staticmethod
    def _base_matrix(rows: list[dict]) -> np.ndarray:
        return np.asarray(
            [[float(row.get(col, float("nan"))) for col in COMPACT_BASE_FEATURES] for row in rows],
            dtype=np.float64,
        )

    @staticmethod
    def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        output = np.empty_like(x, dtype=np.float64)
        positive = x >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        output[~positive] = exp_x / (1.0 + exp_x)
        return output

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, x)

    @staticmethod
    def _inverse_softplus(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.maximum(x, 1e-8)
        return np.where(x > 20.0, x, np.log(np.expm1(x)))

    @classmethod
    def _thresholds_from_raw(cls, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        theta0 = raw[0]
        if raw.size == 1:
            return np.asarray([theta0], dtype=np.float64)
        gaps = cls._softplus(raw[1:]) + THRESHOLD_EPS
        return np.concatenate([[theta0], theta0 + np.cumsum(gaps)]).astype(np.float64)

    @classmethod
    def _raw_from_thresholds(cls, thresholds: np.ndarray) -> np.ndarray:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        raw = np.empty_like(thresholds)
        raw[0] = thresholds[0]
        if thresholds.size > 1:
            gaps = np.diff(thresholds) - THRESHOLD_EPS
            raw[1:] = cls._inverse_softplus(np.maximum(gaps, 1e-6))
        return raw

    @staticmethod
    def _boundary_targets(y: np.ndarray) -> np.ndarray:
        targets = (y[:, None] >= ORDINAL_BOUNDARIES[None, :]).astype(np.float64)
        if LABEL_SMOOTHING > 0:
            targets = targets * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return targets

    @staticmethod
    def _fill_nan_with_medians(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).copy()
        for j in range(x.shape[1]):
            bad = ~np.isfinite(x[:, j])
            if np.any(bad):
                x[bad, j] = medians[j]
        return x

    @staticmethod
    def _phi_from_z(z: np.ndarray) -> np.ndarray:
        z_ndg = z[:, 0]
        z_iqr = z[:, 1]
        z_wb1 = z[:, 2]
        z_wb2 = z[:, 3]
        return np.column_stack(
            [
                z_ndg,
                z_ndg**2,
                z_iqr,
                z_wb1,
                z_wb2,
                z_ndg * z_wb1,
                z_ndg * z_wb2,
            ]
        )

    def _make_phi_train(self, rows: list[dict]) -> np.ndarray:
        x = self._base_matrix(rows)
        self.feature_medians_ = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        x = self._fill_nan_with_medians(x, self.feature_medians_)
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(x)
        return self._phi_from_z(z)

    def _make_phi_predict(self, rows: list[dict]) -> np.ndarray:
        if self.feature_medians_ is None or self.scaler_ is None:
            raise RuntimeError("Model has not been fitted")
        x = self._base_matrix(rows)
        x = self._fill_nan_with_medians(x, self.feature_medians_)
        z = self.scaler_.transform(x)
        return self._phi_from_z(z)

    @staticmethod
    def _fit_thresholds(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
        medians = {}
        med_list = []
        for layer in VALID_LAYERS:
            vals = scores[y == layer]
            med = float(np.median(vals)) if vals.size else float(layer)
            medians[int(layer)] = med
            med_list.append(med)

        med_arr = np.asarray(med_list, dtype=np.float64)
        if np.any(~np.isfinite(med_arr)) or np.any(np.diff(med_arr) <= 1e-8):
            thresholds = (VALID_LAYERS[:-1] + VALID_LAYERS[1:]) / 2.0
        else:
            thresholds = (med_arr[:-1] + med_arr[1:]) / 2.0
        return thresholds.astype(np.float64), medians

    @staticmethod
    def _predict_from_scores(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(thresholds, scores, side="right")
        indices = np.clip(indices, 0, len(VALID_LAYERS) - 1)
        return VALID_LAYERS[indices]

    def _objective_and_grad(
        self,
        params: np.ndarray,
        phi: np.ndarray,
        targets: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        n_features = phi.shape[1]
        w = params[:n_features]
        intercept = params[n_features]
        raw_thresholds = params[n_features + 1 :]
        thresholds = self._thresholds_from_raw(raw_thresholds)

        scores = phi @ w + intercept
        logits = scores[:, None] - thresholds[None, :]

        pos_arg = self.loss_margin_ - logits
        neg_arg = logits + self.loss_margin_
        boundary_loss = targets * self._softplus(pos_arg) + (1.0 - targets) * self._softplus(neg_arg)
        weighted_boundary_loss = boundary_loss * self.boundary_weights_[None, :]
        weighted_sample_loss = sample_weight * weighted_boundary_loss.sum(axis=1)
        loss = float(np.mean(weighted_sample_loss) + self.gamma_ * np.dot(w, w))

        dloss_dlogits = (
            targets * (-self._stable_sigmoid(pos_arg))
            + (1.0 - targets) * self._stable_sigmoid(neg_arg)
        )
        dloss_dlogits *= self.boundary_weights_[None, :]
        dloss_dlogits *= sample_weight[:, None] / max(len(sample_weight), 1)

        dloss_dscores = dloss_dlogits.sum(axis=1)
        grad_w = phi.T @ dloss_dscores + 2.0 * self.gamma_ * w
        grad_intercept = np.asarray([dloss_dscores.sum()], dtype=np.float64)

        grad_thresholds = -dloss_dlogits.sum(axis=0)
        grad_raw = np.empty_like(raw_thresholds)
        grad_raw[0] = grad_thresholds.sum()
        if raw_thresholds.size > 1:
            gap_grad = self._stable_sigmoid(raw_thresholds[1:])
            for index in range(1, raw_thresholds.size):
                grad_raw[index] = grad_thresholds[index:].sum() * gap_grad[index - 1]

        grad = np.concatenate([grad_w, grad_intercept, grad_raw])
        return loss, grad.astype(np.float64)

    def fit(self, rows: list[dict]) -> CompactOrdinalBCE:
        y = labels(rows)
        unsupported = sorted(set(int(value) for value in y) - set(VALID_LAYERS.tolist()))
        if unsupported:
            raise ValueError(f"Unsupported layers for CompactOrdinalBCE: {unsupported}; valid={VALID_LAYERS.tolist()}")

        phi = self._make_phi_train(rows)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)

        init_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        init_model.fit(phi, y, sample_weight=sample_weight)
        init_scores = init_model.predict(phi)
        init_thresholds, _ = self._fit_thresholds(init_scores, y)
        init_raw_thresholds = self._raw_from_thresholds(init_thresholds)
        init_params = np.concatenate(
            [
                np.asarray(init_model.coef_, dtype=np.float64).ravel(),
                np.asarray([float(init_model.intercept_)], dtype=np.float64),
                init_raw_thresholds,
            ]
        )

        targets = self._boundary_targets(y)
        result = minimize(
            fun=lambda params: self._objective_and_grad(params, phi, targets, sample_weight),
            x0=init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": OPT_MAX_ITER, "ftol": 1e-10, "gtol": 1e-7},
        )

        params = result.x.astype(np.float64)
        n_features = phi.shape[1]
        self.coef_ = params[:n_features]
        self.intercept_ = float(params[n_features])
        self.raw_thresholds_ = params[n_features + 1 :]
        self.thresholds_ = self._thresholds_from_raw(self.raw_thresholds_)
        train_scores = phi @ self.coef_ + self.intercept_
        _, self.train_medians_ = self._fit_thresholds(train_scores, y)
        self.optimization_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "fun": float(result.fun),
        }
        return self

    def score_samples(self, rows: list[dict]) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model has not been fitted")
        return self._make_phi_predict(rows) @ self.coef_ + self.intercept_

    def predict(self, rows: list[dict]) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Model thresholds are unavailable")
        return self._predict_from_scores(self.score_samples(rows), self.thresholds_)

    def review_flags(self, rows: list[dict], boundary_margin: float) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Model thresholds are unavailable")
        scores = self.score_samples(rows)
        boundary_distance = np.min(np.abs(scores[:, None] - self.thresholds_[None, :]), axis=1)
        near_boundary = boundary_distance < boundary_margin
        missing_five_zone = (scores >= 4.5) & (scores <= 5.5)
        return near_boundary | missing_five_zone

    def coefficient_rows(self) -> list[dict]:
        if self.coef_ is None or self.intercept_ is None or self.thresholds_ is None:
            raise RuntimeError("Model has not been fitted")
        rows = [{"feature": "intercept", "coef": float(self.intercept_), "boundary_weight": ""}]
        for name, coef in zip(COMPACT_PHI_FEATURES, self.coef_.ravel()):
            rows.append({"feature": name, "coef": float(coef), "boundary_weight": ""})
        for boundary, threshold, weight in zip(ORDINAL_BOUNDARIES, self.thresholds_, self.boundary_weights_):
            rows.append({"feature": f"theta_layer_ge_{int(boundary)}", "coef": float(threshold), "boundary_weight": float(weight)})
        return rows

    def config(self) -> dict:
        return {
            "model_name": "Compact Green Ordinal Boundary BCE Version C",
            "alpha": self.alpha,
            "gamma": self.gamma_,
            "loss_margin": self.loss_margin_,
            "label_smoothing": self.label_smoothing_,
            "l2_scale": L2_SCALE,
            "base_features": COMPACT_BASE_FEATURES,
            "phi_features": COMPACT_PHI_FEATURES,
            "classes": VALID_LAYERS.tolist(),
            "boundaries": ORDINAL_BOUNDARIES.tolist(),
            "boundary_weights": self.boundary_weights_.tolist(),
            "thresholds": self.thresholds_.tolist() if self.thresholds_ is not None else None,
            "raw_thresholds": self.raw_thresholds_.tolist() if self.raw_thresholds_ is not None else None,
            "train_score_medians": self.train_medians_,
            "feature_medians": self.feature_medians_.tolist() if self.feature_medians_ is not None else None,
            "optimization": self.optimization_,
            "no_area_features_used": True,
            "no_full_polynomial_features": True,
        }


# Backward-compatible name used by the existing train_green_ordinal.py entrypoint.
CompactOrdinalRidge = CompactOrdinalBCE


# -----------------------------
# Path, parsing, and COCO utilities
# -----------------------------


def expand_path(path: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def parse_wb_from_filename(filename: str) -> tuple[float, float]:
    stem = Path(filename).stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    blocks = re.findall(r"\d+", stem)
    if len(blocks) >= 4:
        return float(f"{blocks[0]}.{blocks[1]}"), float(f"{blocks[2]}.{blocks[3]}")

    digits = re.findall(r"\d", stem[:12])
    if len(digits) < 4:
        return float("nan"), float("nan")
    return int(digits[0]) + int(digits[1]) / 10.0, int(digits[2]) + int(digits[3]) / 10.0


def parse_group_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    parts = re.split(r"([_-])", stem)
    numeric_seen = 0
    output = []
    for part in parts:
        if part.isdigit() and numeric_seen < 4:
            numeric_seen += 1
            continue
        if numeric_seen < 4 and part in {"_", "-"}:
            continue
        output.append(part)
    grouped = "".join(output).strip("_- ")
    return grouped or stem


def parse_layer_from_category(name: str) -> int | None:
    text = str(name).strip().lower()
    if text.isdigit():
        value = int(text)
        return value if 1 <= value <= 8 else None
    numbers = re.findall(r"(?<!\d)([1-8])(?!\d)", text)
    if numbers:
        return int(numbers[0])
    return None


def load_coco(coco_path: str | os.PathLike) -> dict:
    with open(expand_path(coco_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_category_to_layer(coco_dict: dict) -> dict[int, int]:
    mapping = {}
    for category in coco_dict.get("categories", []):
        layer = parse_layer_from_category(category.get("name", ""))
        if layer is not None:
            mapping[int(category["id"])] = layer
    return mapping


def ann_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = annotation.get("bbox", [0, 0, 0, 0])
        x0 = int(max(0, math.floor(x)))
        y0 = int(max(0, math.floor(y)))
        x1 = int(min(width, math.ceil(x + w)))
        y1 = int(min(height, math.ceil(y + h)))
        mask[y0:y1, x0:x1] = 1
        return mask.astype(bool)

    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        return mask.astype(bool)

    if isinstance(segmentation, dict):
        if mask_utils is None:
            raise ImportError("RLE segmentation requires pycocotools. Run: pip install pycocotools")
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")


# -----------------------------
# Feature extraction
# -----------------------------


def trimmed_mean(values: np.ndarray, low: float = 10.0, high: float = 90.0) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [low, high])
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        return float(np.median(values))
    return float(np.mean(trimmed))


def robust_plane_fit(xs: np.ndarray, ys: np.ndarray, values: np.ndarray, sample_max: int) -> tuple[float, float, float]:
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    xs, ys, values = xs[finite], ys[finite], values[finite]
    if values.size < 3:
        constant = float(np.nanmedian(values)) if values.size else 0.0
        return 0.0, 0.0, constant

    lo, hi = np.percentile(values, [5, 95])
    keep = (values >= lo) & (values <= hi)
    xs, ys, values = xs[keep], ys[keep], values[keep]
    if values.size > sample_max:
        rng = np.random.default_rng(12345)
        indices = rng.choice(values.size, size=sample_max, replace=False)
        xs, ys, values = xs[indices], ys[indices], values[indices]

    design = np.column_stack([xs, ys, np.ones_like(xs)])
    coefs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(coefs[0]), float(coefs[1]), float(coefs[2])


def extract_one_annotation_features(image_bgr: np.ndarray, annotation_mask: np.ndarray, union_mask: np.ndarray, cfg: ExtractConfig) -> dict | None:
    height, width = annotation_mask.shape
    area_px = int(np.count_nonzero(annotation_mask))
    if area_px < cfg.min_area_px:
        return None

    ys, xs = np.where(annotation_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_w = max(1, x_max - x_min + 1)
    bbox_h = max(1, y_max - y_min + 1)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    roi_w = max(bbox_w * cfg.roi_scale, bbox_w + 40)
    roi_h = max(bbox_h * cfg.roi_scale, bbox_h + 40)
    x0 = int(max(0, math.floor(center_x - roi_w / 2.0)))
    x1 = int(min(width, math.ceil(center_x + roi_w / 2.0)))
    y0 = int(max(0, math.floor(center_y - roi_h / 2.0)))
    y1 = int(min(height, math.ceil(center_y + roi_h / 2.0)))

    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True

    kernel_size = max(1, int(cfg.edge_dilate_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_union = cv2.dilate(union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    background_mask = roi_mask & (~dilated_union)
    if int(np.count_nonzero(background_mask)) < cfg.bg_min_pixels:
        background_mask = roi_mask & (~union_mask)
    if int(np.count_nonzero(background_mask)) < cfg.bg_min_pixels:
        return None

    green = image_bgr[:, :, 1].astype(np.float32)
    bg_y, bg_x = np.where(background_mask)
    bg_values = green[background_mask]
    a, b, c = robust_plane_fit(bg_x, bg_y, bg_values, sample_max=cfg.bg_sample_max)

    plane_at_flake = a * xs.astype(np.float32) + b * ys.astype(np.float32) + c
    flake_values = green[annotation_mask].astype(np.float32)
    delta_values = flake_values - plane_at_flake.astype(np.float32)

    g_flake_median = float(np.median(flake_values))
    g_bg_plane_median = float(np.median(plane_at_flake))
    delta_g = trimmed_mean(delta_values, cfg.trim_low, cfg.trim_high)
    ndg = delta_g / max(abs(g_bg_plane_median), 1.0)
    p10, p90 = np.percentile(delta_values, [10, 90])
    g_delta_iqr = float(p90 - p10)
    iqr_ndg = g_delta_iqr / max(abs(g_bg_plane_median), 1.0)

    return {
        "area_px": area_px,
        "log_area_px": float(np.log1p(area_px)),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_aspect": float(bbox_w / max(bbox_h, 1)),
        "roi_bg_pixels": int(np.count_nonzero(background_mask)),
        "ndg": float(ndg),
        "delta_g": float(delta_g),
        "abs_delta_g": float(abs(delta_g)),
        "g_flake_median": g_flake_median,
        "g_bg_plane_median": g_bg_plane_median,
        "g_delta_p10": float(p10),
        "g_delta_p90": float(p90),
        "g_delta_iqr": g_delta_iqr,
        "iqr_ndg": float(iqr_ndg),
        "g_bg_std": float(np.std(bg_values)),
        "g_flake_std": float(np.std(flake_values)),
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_c": float(c),
    }


def read_split_features(split: str, image_dir: str | os.PathLike, coco_path: str | os.PathLike, cfg: ExtractConfig) -> list[GreenFeatureRow]:
    image_dir = expand_path(image_dir)
    coco_dict = load_coco(coco_path)
    category_to_layer = build_category_to_layer(coco_dict)
    images = {int(image["id"]): image for image in coco_dict.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco_dict.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rows: list[GreenFeatureRow] = []
    items = list(annotations_by_image.items())
    for image_index, (image_id, annotations) in enumerate(items, start=1):
        image_info = images.get(image_id)
        if image_info is None:
            continue
        file_name = image_info.get("file_name", "")
        image_path = image_dir / file_name
        if not image_path.exists():
            image_path = image_dir / Path(file_name).name
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue
        height, width = image_bgr.shape[:2]

        decoded: list[tuple[dict, int, np.ndarray]] = []
        union_mask = np.zeros((height, width), dtype=bool)
        valid_or_missing_five = set(VALID_LAYERS.tolist() + [5])
        for annotation in annotations:
            layer = category_to_layer.get(int(annotation.get("category_id", -1)))
            if layer is None or layer not in valid_or_missing_five:
                continue
            mask = ann_to_mask(annotation, height, width)
            if int(np.count_nonzero(mask)) < cfg.min_area_px:
                continue
            decoded.append((annotation, layer, mask))
            union_mask |= mask

        if not decoded:
            continue

        wb1, wb2 = parse_wb_from_filename(file_name)
        group = parse_group_from_filename(file_name)
        for annotation, layer, mask in decoded:
            if layer not in set(VALID_LAYERS.tolist()):
                continue
            feature_values = extract_one_annotation_features(image_bgr, mask, union_mask, cfg)
            if feature_values is None:
                continue
            rows.append(
                GreenFeatureRow(
                    split=split,
                    filename=Path(file_name).name,
                    image_path=str(image_path),
                    image_id=int(image_id),
                    ann_id=int(annotation.get("id", -1)),
                    layer=int(layer),
                    group=group,
                    wb1=float(wb1),
                    wb2=float(wb2),
                    wb_product=float(wb1 * wb2) if np.isfinite(wb1) and np.isfinite(wb2) else float("nan"),
                    wb_sum=float(wb1 + wb2) if np.isfinite(wb1) and np.isfinite(wb2) else float("nan"),
                    **feature_values,
                )
            )

        if image_index % 25 == 0:
            print(f"[{split}] processed {image_index}/{len(items)} images, features={len(rows)}")

    return rows


def discover_splits(coco_dir: str | os.PathLike) -> list[tuple[str, Path, Path]]:
    root = expand_path(coco_dir)
    split_specs = []
    for split in ("train", "valid", "val", "test"):
        split_dir = root / split
        annotation_path = split_dir / "_annotations.coco.json"
        if annotation_path.exists():
            normalized = "valid" if split == "val" else split
            split_specs.append((normalized, split_dir, annotation_path))
    if not split_specs:
        raise FileNotFoundError(f"No split _annotations.coco.json files found under {root}")
    return split_specs


def extract_features(
    cfg: ExtractConfig,
    coco_dir: str | None = None,
    train_image_dir: str | None = None,
    train_coco: str | None = None,
    test_image_dir: str | None = None,
    test_coco: str | None = None,
    valid_image_dir: str | None = None,
    valid_coco: str | None = None,
) -> list[GreenFeatureRow]:
    split_specs: list[tuple[str, str | os.PathLike, str | os.PathLike]] = []
    if coco_dir:
        split_specs.extend(discover_splits(coco_dir))
    else:
        if not train_image_dir or not train_coco:
            raise ValueError("Provide --coco-dir or both --train-image-dir and --train-coco")
        split_specs.append(("train", train_image_dir, train_coco))
        if valid_image_dir and valid_coco:
            split_specs.append(("valid", valid_image_dir, valid_coco))
        if test_image_dir and test_coco:
            split_specs.append(("test", test_image_dir, test_coco))

    all_rows: list[GreenFeatureRow] = []
    for split, image_dir, coco_path in split_specs:
        print(f"[INFO] Extracting {split}: {coco_path}")
        all_rows.extend(read_split_features(split, image_dir, coco_path, cfg))
    return all_rows


# -----------------------------
# CSV and row utilities
# -----------------------------


def rows_to_dicts(rows: Iterable[GreenFeatureRow]) -> list[dict]:
    return [asdict(row) for row in rows]


def write_csv(rows: list[dict], path: str | os.PathLike) -> None:
    path = expand_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | os.PathLike) -> list[dict]:
    with open(expand_path(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def coerce_rows(rows: list[dict]) -> list[dict]:
    numeric_columns = set(GreenFeatureRow.__dataclass_fields__) - {"split", "filename", "image_path", "group"}
    output = []
    for row in rows:
        converted = dict(row)
        for column in numeric_columns:
            if column in converted:
                try:
                    converted[column] = float(converted[column])
                except Exception:
                    converted[column] = float("nan")

        if "iqr_ndg" not in converted or not np.isfinite(float(converted.get("iqr_ndg", float("nan")))):
            g_delta_iqr = float(converted.get("g_delta_iqr", float("nan")))
            g_bg = abs(float(converted.get("g_bg_plane_median", float("nan"))))
            converted["iqr_ndg"] = g_delta_iqr / max(g_bg, 1.0) if np.isfinite(g_delta_iqr) and np.isfinite(g_bg) else float("nan")

        converted["image_id"] = int(float(converted.get("image_id", -1)))
        converted["ann_id"] = int(float(converted.get("ann_id", -1)))
        converted["layer"] = int(float(converted["layer"]))
        if converted["layer"] not in set(VALID_LAYERS.tolist()):
            continue
        converted["area_px"] = int(float(converted.get("area_px", 0)))
        converted["roi_bg_pixels"] = int(float(converted.get("roi_bg_pixels", 0)))
        output.append(converted)
    return output


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["layer"]) for row in rows], dtype=np.int64)


def groups(rows: list[dict]) -> np.ndarray:
    return np.asarray([str(row["group"]) for row in rows])


def class_counts(rows: list[dict]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(row["layer"]) for row in rows).items())}


def split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        output[split] = class_counts([row for row in rows if row["split"] == split])
    return output


def write_json(payload: dict, path: str | os.PathLike) -> None:
    path = expand_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# -----------------------------
# Metrics, training, and run modes
# -----------------------------


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, review: np.ndarray | None = None) -> dict:
    abs_error = np.abs(y_true - y_pred)
    metrics = {
        "n": int(len(y_true)),
        "exact_acc": float(accuracy_score(y_true, y_pred)),
        "within1_acc": float(np.mean(abs_error <= 1)),
        "large_error_rate": float(np.mean(abs_error > 1)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=VALID_LAYERS, average="macro", zero_division=0)),
    }
    if review is not None:
        keep = ~review
        metrics["review_rate"] = float(np.mean(review))
        metrics["kept_n"] = int(np.count_nonzero(keep))
        if np.any(keep):
            metrics["kept_exact_acc"] = float(accuracy_score(y_true[keep], y_pred[keep]))
            metrics["kept_within1_acc"] = float(np.mean(np.abs(y_true[keep] - y_pred[keep]) <= 1))
            metrics["kept_mae"] = float(mean_absolute_error(y_true[keep], y_pred[keep]))
        else:
            metrics["kept_exact_acc"] = float("nan")
            metrics["kept_within1_acc"] = float("nan")
            metrics["kept_mae"] = float("nan")
    return metrics


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=VALID_LAYERS)
    rows = []
    for layer, values in zip(VALID_LAYERS, matrix):
        row = {"true_layer": int(layer)}
        for pred_layer, count in zip(VALID_LAYERS, values):
            row[f"pred_{int(pred_layer)}"] = int(count)
        rows.append(row)
    return rows


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def safe_filename(text: str, max_len: int = 140) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    if not text:
        text = "item"
    return text[:max_len]


def export_large_error_images(
    prediction_rows: list[dict],
    out_dir: Path,
    subdir: str = "large_error_images",
    min_abs_error: float = 2.0,
) -> None:
    error_rows = [row for row in prediction_rows if float(row.get("abs_error", 0.0)) >= min_abs_error]
    if not error_rows:
        return

    image_out_dir = out_dir / subdir
    image_out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for index, row in enumerate(error_rows, start=1):
        src = Path(str(row.get("image_path", "")))
        ext = src.suffix.lower() if src.suffix else ".jpg"
        out_name = safe_filename(
            f"{index:03d}_gt{int(row['layer'])}_pred{int(row['pred_layer'])}"
            f"_err{float(row['abs_error']):.0f}_score{float(row['score']):.3f}"
            f"_ann{int(row.get('ann_id', -1))}_{Path(str(row.get('filename', 'image'))).stem}"
        ) + ext
        dst = image_out_dir / out_name

        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is not None:
            label = (
                f"gt={int(row['layer'])} pred={int(row['pred_layer'])} "
                f"err={float(row['abs_error']):.0f} score={float(row['score']):.3f} "
                f"ann={int(row.get('ann_id', -1))}"
            )
            cv2.rectangle(image, (0, 0), (min(image.shape[1], 900), 44), (255, 255, 255), thickness=-1)
            cv2.putText(image, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(dst), image)
        elif src.exists():
            shutil.copy2(src, dst)
        else:
            dst = Path("")

        out_row = {
            "exported_image": str(dst),
            "source_image": str(src),
            "filename": row.get("filename", ""),
            "image_id": row.get("image_id", ""),
            "ann_id": row.get("ann_id", ""),
            "layer": int(row["layer"]),
            "pred_layer": int(row["pred_layer"]),
            "score": float(row["score"]),
            "abs_error": float(row["abs_error"]),
            "review": row.get("review", ""),
        }
        index_rows.append(out_row)

    write_csv(index_rows, image_out_dir / "index.csv")
    print(f"[INFO] Exported {len(index_rows)} large-error images to {image_out_dir}")


def evaluate_model(model: CompactOrdinalRidge, test_rows: list[dict], boundary_margin: float) -> tuple[list[dict], dict]:
    y_true = labels(test_rows)
    scores = model.score_samples(test_rows)
    pred = model.predict(test_rows)
    review = model.review_flags(test_rows, boundary_margin=boundary_margin)
    metrics = evaluate_predictions(y_true, pred, review)

    prediction_rows = []
    for row, score, predicted, should_review in zip(test_rows, scores, pred, review):
        pred_row = dict(row)
        pred_row["score"] = float(score)
        pred_row["pred_layer"] = int(predicted)
        pred_row["review"] = bool(should_review)
        pred_row["abs_error"] = float(abs(int(row["layer"]) - int(predicted)))
        prediction_rows.append(pred_row)
    return prediction_rows, metrics


def parse_alpha_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def tune_alpha_on_rows(rows: list[dict], alphas: list[float], folds: int, seed: int, boundary_margin: float) -> tuple[float, list[dict]]:
    y = labels(rows)
    min_class_count = min(Counter(y.tolist()).values())
    n_splits = max(2, min(folds, min_class_count))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    tuning_rows = []
    for alpha in alphas:
        fold_metrics = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(rows)), y), start=1):
            train_rows = [rows[int(index)] for index in train_idx]
            valid_rows = [rows[int(index)] for index in valid_idx]
            model = CompactOrdinalRidge(alpha=alpha, random_state=seed).fit(train_rows)
            _, metrics = evaluate_model(model, valid_rows, boundary_margin=boundary_margin)
            fold_metrics.append(metrics)

        row = {"alpha": float(alpha), "folds": int(n_splits)}
        for key in ["exact_acc", "within1_acc", "large_error_rate", "mae", "macro_f1", "review_rate", "kept_exact_acc"]:
            values = [m[key] for m in fold_metrics if key in m and np.isfinite(m[key])]
            row[f"cv_{key}"] = float(np.mean(values)) if values else float("nan")
        tuning_rows.append(row)

    tuning_rows = sorted(
        tuning_rows,
        key=lambda r: (
            -r["cv_exact_acc"],
            -r["cv_macro_f1"],
            r["cv_large_error_rate"],
            r["cv_mae"],
            r["alpha"],
        ),
    )
    return float(tuning_rows[0]["alpha"]), tuning_rows


def fit_compact_model(
    train_rows: list[dict],
    tune: bool,
    alpha: float,
    alpha_grid: list[float],
    folds: int,
    seed: int,
    boundary_margin: float,
    tuning_csv_path: Path | None = None,
) -> tuple[CompactOrdinalRidge, float]:
    if tune:
        alpha, tuning_rows = tune_alpha_on_rows(train_rows, alpha_grid, folds, seed, boundary_margin)
        if tuning_csv_path is not None:
            write_csv(tuning_rows, tuning_csv_path)
        print(f"[INFO] Best alpha from train-only CV: {alpha}")

    model = CompactOrdinalRidge(alpha=alpha, random_state=seed).fit(train_rows)
    return model, alpha


def run_roboflow_test(
    rows: list[dict],
    out_dir: Path,
    tune: bool,
    alpha: float,
    alpha_grid: list[float],
    folds: int,
    seed: int,
    boundary_margin: float,
    args_payload: dict,
) -> None:
    train_rows = [row for row in rows if str(row["split"]).lower() in {"train", "valid", "val"}]
    test_rows = [row for row in rows if str(row["split"]).lower() == "test"]
    if not train_rows or not test_rows:
        raise RuntimeError("roboflow_test mode requires non-empty train/valid and test splits")

    model, alpha = fit_compact_model(
        train_rows=train_rows,
        tune=tune,
        alpha=alpha,
        alpha_grid=alpha_grid,
        folds=folds,
        seed=seed,
        boundary_margin=boundary_margin,
        tuning_csv_path=out_dir / "alpha_tuning_train_cv.csv" if tune else None,
    )

    train_prediction_rows, train_metrics = evaluate_model(model, train_rows, boundary_margin=boundary_margin)
    print_metrics("Compact ordinal boundary BCE train self-fit", train_metrics)
    write_csv(train_prediction_rows, out_dir / "train_predictions.csv")
    write_json(train_metrics, out_dir / "train_self_metrics.json")
    export_large_error_images(train_prediction_rows, out_dir, subdir="train_large_error_images")

    prediction_rows, metrics = evaluate_model(model, test_rows, boundary_margin=boundary_margin)
    print_metrics("Compact ordinal boundary BCE Roboflow test", metrics)
    print(f"[INFO] alpha: {alpha}")
    print(f"[INFO] classes: {VALID_LAYERS}")
    print(f"[INFO] boundaries: {ORDINAL_BOUNDARIES}")
    print(f"[INFO] thresholds: {model.thresholds_}")
    print(f"[INFO] train score medians: {model.train_medians_}")

    write_csv(prediction_rows, out_dir / "test_predictions.csv")
    export_large_error_images(prediction_rows, out_dir, subdir="test_large_error_images")
    y_true = labels(prediction_rows)
    y_pred = np.asarray([int(row["pred_layer"]) for row in prediction_rows], dtype=np.int64)
    write_csv(confusion_rows(y_true, y_pred), out_dir / "test_confusion_matrix.csv")
    write_csv(model.coefficient_rows(), out_dir / "coefficients.csv")
    write_json(metrics, out_dir / "test_metrics.json")
    write_json(model.config() | {"args": args_payload}, out_dir / "model_config.json")
    joblib.dump(model, out_dir / "green_ordinal_model.joblib")
    print(f"[INFO] Saved model: {out_dir / 'green_ordinal_model.joblib'}")


def run_cv(
    rows: list[dict],
    out_dir: Path,
    folds: int,
    seed: int,
    tune: bool,
    alpha: float,
    alpha_grid: list[float],
    boundary_margin: float,
) -> None:
    y = labels(rows)
    g = groups(rows)
    min_class_count = min(Counter(y.tolist()).values())
    n_splits = max(2, min(folds, min_class_count))

    if StratifiedGroupKFold is not None and len(set(g.tolist())) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(rows)), y, g)
        splitter_name = "StratifiedGroupKFold"
    elif len(set(g.tolist())) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(np.zeros(len(rows)), y, g)
        splitter_name = "GroupKFold"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(rows)), y)
        splitter_name = "StratifiedKFold"

    print(f"[INFO] CV splitter: {splitter_name}, folds={n_splits}")
    all_predictions = []
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_rows = [rows[int(index)] for index in train_idx]
        test_rows = [rows[int(index)] for index in test_idx]
        model, fold_alpha = fit_compact_model(
            train_rows=train_rows,
            tune=tune,
            alpha=alpha,
            alpha_grid=alpha_grid,
            folds=folds,
            seed=seed,
            boundary_margin=boundary_margin,
            tuning_csv_path=None,
        )
        prediction_rows, metrics = evaluate_model(model, test_rows, boundary_margin=boundary_margin)
        for row in prediction_rows:
            row["fold"] = fold
        metrics["fold"] = fold
        metrics["alpha"] = fold_alpha
        metrics["thresholds"] = ",".join(f"{value:.6g}" for value in model.thresholds_)
        fold_metrics.append(metrics)
        all_predictions.extend(prediction_rows)
        print_metrics(f"Fold {fold}", metrics)

    write_csv(all_predictions, out_dir / "cv_predictions.csv")
    export_large_error_images(all_predictions, out_dir, subdir="cv_large_error_images")
    write_csv(fold_metrics, out_dir / "cv_metrics_by_fold.csv")
    y_true = labels(all_predictions)
    y_pred = np.asarray([int(row["pred_layer"]) for row in all_predictions], dtype=np.int64)
    review = np.asarray([str(row["review"]).lower() == "true" or row["review"] is True for row in all_predictions])
    overall = evaluate_predictions(y_true, y_pred, review)
    print_metrics("CV overall", overall)
    write_json(overall, out_dir / "cv_overall_metrics.json")
    write_csv(confusion_rows(y_true, y_pred), out_dir / "cv_confusion_matrix.csv")


def run_final(
    rows: list[dict],
    out_dir: Path,
    tune: bool,
    alpha: float,
    alpha_grid: list[float],
    folds: int,
    seed: int,
    boundary_margin: float,
    args_payload: dict,
) -> None:
    model, alpha = fit_compact_model(
        train_rows=rows,
        tune=tune,
        alpha=alpha,
        alpha_grid=alpha_grid,
        folds=folds,
        seed=seed,
        boundary_margin=boundary_margin,
        tuning_csv_path=out_dir / "alpha_tuning_final_cv.csv" if tune else None,
    )
    prediction_rows, metrics = evaluate_model(model, rows, boundary_margin=boundary_margin)
    print_metrics("Final self-fit, not validation", metrics)
    write_csv(prediction_rows, out_dir / "trainset_self_predictions.csv")
    export_large_error_images(prediction_rows, out_dir, subdir="final_large_error_images")
    write_csv(model.coefficient_rows(), out_dir / "coefficients.csv")
    write_json(metrics, out_dir / "final_self_metrics.json")
    write_json(model.config() | {"args": args_payload}, out_dir / "model_config.json")
    joblib.dump(model, out_dir / "green_ordinal_model_final.joblib")
    print(f"[INFO] Saved final model: {out_dir / 'green_ordinal_model_final.joblib'}")
