from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
from pycocotools import mask as mask_utils
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


VALID_LAYERS = np.array([1, 2, 3, 4, 6, 7, 8], dtype=np.int64)

DEFAULT_FEATURE_COLS = [
    "ndg",
    "delta_g",
    "abs_delta_g",
    "g_flake_median",
    "g_bg_plane_median",
    "g_delta_p10",
    "g_delta_p90",
    "g_delta_iqr",
    "g_bg_std",
    "g_flake_std",
    "wb1",
    "wb2",
    "wb_product",
    "wb_sum",
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
    g_bg_std: float
    g_flake_std: float
    plane_a: float
    plane_b: float
    plane_c: float


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

    # For names like 1-7-3-9-6-10-6_png, drop the first four numeric WB blocks.
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
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")


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


def robust_plane_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    sample_max: int,
) -> tuple[float, float, float]:
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


def extract_one_annotation_features(
    image_bgr: np.ndarray,
    annotation_mask: np.ndarray,
    union_mask: np.ndarray,
    cfg: ExtractConfig,
) -> dict | None:
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
        "g_delta_iqr": float(p90 - p10),
        "g_bg_std": float(np.std(bg_values)),
        "g_flake_std": float(np.std(flake_values)),
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_c": float(c),
    }


def read_split_features(
    split: str,
    image_dir: str | os.PathLike,
    coco_path: str | os.PathLike,
    cfg: ExtractConfig,
) -> list[GreenFeatureRow]:
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
        for annotation in annotations:
            layer = category_to_layer.get(int(annotation.get("category_id", -1)))
            if layer is None or layer not in set(VALID_LAYERS.tolist() + [5]):
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
            row = GreenFeatureRow(
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
            rows.append(row)

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
                converted[column] = float(converted[column])
        converted["image_id"] = int(converted["image_id"])
        converted["ann_id"] = int(converted["ann_id"])
        converted["layer"] = int(float(converted["layer"]))
        converted["area_px"] = int(float(converted["area_px"]))
        converted["roi_bg_pixels"] = int(float(converted["roi_bg_pixels"]))
        output.append(converted)
    return output


def feature_matrix(rows: list[dict], feature_cols: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(row.get(column, float("nan"))) for column in feature_cols])
    return np.asarray(matrix, dtype=np.float64)


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([int(row["layer"]) for row in rows], dtype=np.int64)


def groups(rows: list[dict]) -> np.ndarray:
    return np.asarray([str(row["group"]) for row in rows])


def make_model(regressor: str, degree: int, ridge_alpha: float, huber_alpha: float, huber_epsilon: float) -> Pipeline:
    if regressor == "ridge":
        model = Ridge(alpha=ridge_alpha)
    elif regressor == "huber":
        model = HuberRegressor(epsilon=huber_epsilon, alpha=huber_alpha, max_iter=1000)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale1", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale2", StandardScaler()),
            ("model", model),
        ]
    )


def fit_thresholds(scores: np.ndarray, y: np.ndarray, classes: np.ndarray = VALID_LAYERS) -> np.ndarray:
    medians = []
    for layer in classes:
        values = scores[y == layer]
        medians.append(float(np.median(values)) if values.size else float(layer))
    medians = np.asarray(medians, dtype=np.float64)
    if np.any(~np.isfinite(medians)) or np.any(np.diff(medians) <= 1e-6):
        return (classes[:-1] + classes[1:]) / 2.0
    thresholds = (medians[:-1] + medians[1:]) / 2.0
    if np.any(np.diff(thresholds) <= 1e-6):
        return (classes[:-1] + classes[1:]) / 2.0
    return thresholds


def predict_from_scores(scores: np.ndarray, thresholds: np.ndarray, classes: np.ndarray = VALID_LAYERS) -> np.ndarray:
    indices = np.searchsorted(thresholds, scores, side="right")
    indices = np.clip(indices, 0, len(classes) - 1)
    return classes[indices]


def review_flags(scores: np.ndarray, thresholds: np.ndarray, boundary_margin: float) -> np.ndarray:
    if thresholds.size:
        boundary_distance = np.min(np.abs(scores[:, None] - thresholds[None, :]), axis=1)
        near_boundary = boundary_distance < boundary_margin
    else:
        near_boundary = np.zeros_like(scores, dtype=bool)
    missing_five_zone = (scores >= 4.5) & (scores <= 5.5)
    return near_boundary | missing_five_zone


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, review: np.ndarray | None = None) -> dict:
    metrics = {
        "n": int(len(y_true)),
        "exact_acc": float(accuracy_score(y_true, y_pred)),
        "within1_acc": float(np.mean(np.abs(y_true - y_pred) <= 1)),
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


def class_counts(rows: list[dict]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(row["layer"]) for row in rows).items())}


def split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        output[split] = class_counts([row for row in rows if row["split"] == split])
    return output


def select_feature_cols(rows: list[dict], requested: list[str]) -> list[str]:
    available = set(rows[0].keys()) if rows else set()
    selected = [column for column in requested if column in available]
    missing = [column for column in requested if column not in available]
    if missing:
        print(f"[WARN] Ignoring missing feature columns: {missing}")
    if not selected:
        raise ValueError("No valid feature columns selected")
    return selected


def train_and_predict(
    train_rows: list[dict],
    test_rows: list[dict],
    feature_cols: list[str],
    regressor: str,
    degree: int,
    ridge_alpha: float,
    huber_alpha: float,
    huber_epsilon: float,
    boundary_margin: float,
) -> tuple[Pipeline, np.ndarray, list[dict], dict]:
    x_train = feature_matrix(train_rows, feature_cols)
    y_train = labels(train_rows)
    x_test = feature_matrix(test_rows, feature_cols)
    y_test = labels(test_rows)

    pipeline = make_model(regressor, degree, ridge_alpha, huber_alpha, huber_epsilon)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)

    train_scores = pipeline.predict(x_train)
    thresholds = fit_thresholds(train_scores, y_train)
    scores = pipeline.predict(x_test)
    pred = predict_from_scores(scores, thresholds)
    review = review_flags(scores, thresholds, boundary_margin)

    prediction_rows = []
    for row, score, predicted, should_review in zip(test_rows, scores, pred, review):
        pred_row = dict(row)
        pred_row["score"] = float(score)
        pred_row["pred_layer"] = int(predicted)
        pred_row["review"] = bool(should_review)
        pred_row["abs_error"] = float(abs(int(row["layer"]) - int(predicted)))
        prediction_rows.append(pred_row)

    return pipeline, thresholds, prediction_rows, evaluate_predictions(y_test, pred, review)


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, labels=VALID_LAYERS)
    rows = []
    for layer, values in zip(VALID_LAYERS, matrix):
        row = {"true_layer": int(layer)}
        for pred_layer, count in zip(VALID_LAYERS, values):
            row[f"pred_{int(pred_layer)}"] = int(count)
        rows.append(row)
    return rows


def write_json(payload: dict, path: str | os.PathLike) -> None:
    path = expand_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def run_roboflow_test(
    rows: list[dict],
    feature_cols: list[str],
    out_dir: Path,
    regressor: str,
    degree: int,
    ridge_alpha: float,
    huber_alpha: float,
    huber_epsilon: float,
    boundary_margin: float,
    args_payload: dict,
) -> None:
    train_rows = [row for row in rows if str(row["split"]).lower() in {"train", "valid", "val"}]
    test_rows = [row for row in rows if str(row["split"]).lower() == "test"]
    if not train_rows or not test_rows:
        raise RuntimeError("roboflow_test mode requires non-empty train/valid and test splits")

    model, thresholds, prediction_rows, metrics = train_and_predict(
        train_rows,
        test_rows,
        feature_cols,
        regressor,
        degree,
        ridge_alpha,
        huber_alpha,
        huber_epsilon,
        boundary_margin,
    )
    print_metrics("Roboflow test", metrics)
    print(f"[INFO] thresholds: {thresholds}")

    write_csv(prediction_rows, out_dir / "test_predictions.csv")
    y_true = labels(prediction_rows)
    y_pred = np.asarray([int(row["pred_layer"]) for row in prediction_rows], dtype=np.int64)
    write_csv(confusion_rows(y_true, y_pred), out_dir / "test_confusion_matrix.csv")
    write_json(metrics, out_dir / "test_metrics.json")
    joblib.dump(
        {
            "model": model,
            "thresholds": thresholds,
            "classes": VALID_LAYERS,
            "feature_cols": feature_cols,
            "args": args_payload,
        },
        out_dir / "green_ordinal_model.joblib",
    )
    print(f"[INFO] Saved model: {out_dir / 'green_ordinal_model.joblib'}")


def run_cv(
    rows: list[dict],
    feature_cols: list[str],
    out_dir: Path,
    folds: int,
    seed: int,
    regressor: str,
    degree: int,
    ridge_alpha: float,
    huber_alpha: float,
    huber_epsilon: float,
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
        _, thresholds, prediction_rows, metrics = train_and_predict(
            train_rows,
            test_rows,
            feature_cols,
            regressor,
            degree,
            ridge_alpha,
            huber_alpha,
            huber_epsilon,
            boundary_margin,
        )
        for row in prediction_rows:
            row["fold"] = fold
        metrics["fold"] = fold
        metrics["thresholds"] = ",".join(f"{value:.6g}" for value in thresholds)
        fold_metrics.append(metrics)
        all_predictions.extend(prediction_rows)
        print_metrics(f"Fold {fold}", metrics)

    write_csv(all_predictions, out_dir / "cv_predictions.csv")
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
    feature_cols: list[str],
    out_dir: Path,
    regressor: str,
    degree: int,
    ridge_alpha: float,
    huber_alpha: float,
    huber_epsilon: float,
    boundary_margin: float,
    args_payload: dict,
) -> None:
    model, thresholds, prediction_rows, metrics = train_and_predict(
        rows,
        rows,
        feature_cols,
        regressor,
        degree,
        ridge_alpha,
        huber_alpha,
        huber_epsilon,
        boundary_margin,
    )
    print_metrics("Final self-fit, not validation", metrics)
    write_csv(prediction_rows, out_dir / "trainset_self_predictions.csv")
    write_json(metrics, out_dir / "final_self_metrics.json")
    joblib.dump(
        {
            "model": model,
            "thresholds": thresholds,
            "classes": VALID_LAYERS,
            "feature_cols": feature_cols,
            "args": args_payload,
        },
        out_dir / "green_ordinal_model_final.joblib",
    )

