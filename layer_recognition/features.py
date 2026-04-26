from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from pycocotools import mask as mask_utils


DEFAULT_FEATURE_NAMES = ("normalized_delta_g", "wb1")


@dataclass(slots=True)
class LayerFeature:
    image_id: int
    ann_id: int
    file_name: str
    group_id: str
    layer_gt: int | None
    wb1: float
    wb2: float
    g_flake_peak: float
    g_bg_median: float
    delta_g: float
    normalized_delta_g: float
    area_px: int
    log_area_px: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


def load_coco(coco_json_path: str) -> dict:
    coco_json_path = os.path.expanduser(coco_json_path)
    with open(coco_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def image_group_id(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem


def parse_wb_from_name(file_name: str) -> tuple[float, float] | None:
    stem = image_group_id(file_name)
    blocks = re.findall(r"\d+", stem)
    if len(blocks) < 4:
        return None
    return float(f"{blocks[0]}.{blocks[1]}"), float(f"{blocks[2]}.{blocks[3]}")


def parse_layer_from_category_name(name: str) -> int | None:
    text = str(name).strip().lower()
    if text.isdigit():
        return int(text)
    if text.endswith("l") and text[:-1].isdigit():
        return int(text[:-1])
    return None


def build_category_to_layer(coco_dict: dict) -> dict[int, int]:
    output = {}
    for category in coco_dict.get("categories", []):
        layer = parse_layer_from_category_name(category.get("name", ""))
        if layer is not None:
            output[int(category["id"])] = layer
    return output


def ann_to_mask(annotation: dict, image_height: int, image_width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        raise ValueError(f"Annotation {annotation.get('id')} has no segmentation")

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, image_height, image_width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, image_height, image_width)
        else:
            rle = segmentation
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    decoded = mask_utils.decode(rle)
    return decoded.astype(bool)


def expanded_roi(
    image_shape: tuple[int, int],
    bbox: Iterable[float],
    scale: float = 3.0,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x, y, w, h = [float(value) for value in bbox]
    cx = x + w / 2.0
    cy = y + h / 2.0
    roi_w = max(w * scale, 8.0)
    roi_h = max(h * scale, 8.0)
    x1 = max(0, int(cx - roi_w / 2.0))
    y1 = max(0, int(cy - roi_h / 2.0))
    x2 = min(width, int(cx + roi_w / 2.0))
    y2 = min(height, int(cy + roi_h / 2.0))
    return x1, y1, x2, y2


def crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return array[y1:y2, x1:x2]


def estimate_peak(values: np.ndarray, num_bins: int = 50) -> float:
    if values.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(values.astype(np.float32), bins=num_bins)
    peak_index = int(np.argmax(hist))
    return float((bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0)


def _fit_background_plane(g_smooth: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_coords, x_coords = np.nonzero(bg_mask)
    if len(x_coords) < 16:
        constant = float(np.median(g_smooth)) if g_smooth.size else 0.0
        return np.full_like(g_smooth, constant, dtype=np.float32), np.array([0.0, 0.0, constant])

    design = np.column_stack((x_coords, y_coords, np.ones_like(x_coords)))
    values = g_smooth[bg_mask]
    coefs, *_ = np.linalg.lstsq(design, values, rcond=None)
    height, width = g_smooth.shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    bg_fit = coefs[0] * x_grid + coefs[1] * y_grid + coefs[2]
    return bg_fit.astype(np.float32), coefs.astype(np.float32)


def process_green_background(
    image_rgb: np.ndarray,
    ignore_mask: np.ndarray | None,
    threshold_high: float = 180.0,
    tol_low: float = 3.0,
    tol_high: float = 3.0,
) -> dict[str, np.ndarray]:
    green = image_rgb[:, :, 1].astype(np.float32)
    g_bg_smooth = cv2.bilateralFilter(green, d=-1, sigmaColor=10, sigmaSpace=3)
    g_flake_smooth = cv2.bilateralFilter(green, d=-1, sigmaColor=3, sigmaSpace=8)

    candidate_mask = g_bg_smooth <= threshold_high
    if ignore_mask is not None:
        candidate_mask = candidate_mask & (~ignore_mask.astype(bool))
    if np.count_nonzero(candidate_mask) < 16:
        candidate_mask = np.ones_like(g_bg_smooth, dtype=bool)
        if ignore_mask is not None:
            candidate_mask = candidate_mask & (~ignore_mask.astype(bool))

    bg_peak = estimate_peak(g_bg_smooth[candidate_mask], num_bins=256)
    bg_mask = candidate_mask & (g_bg_smooth >= bg_peak - tol_low) & (g_bg_smooth <= bg_peak + tol_high)
    if np.count_nonzero(bg_mask) < 16:
        bg_mask = candidate_mask

    bg_fit, plane = _fit_background_plane(g_bg_smooth, bg_mask)
    g_hybrid = g_bg_smooth.copy()
    if ignore_mask is not None:
        g_hybrid[ignore_mask.astype(bool)] = g_flake_smooth[ignore_mask.astype(bool)]

    return {
        "g_corr": g_hybrid - bg_fit,
        "g_bg_fit": bg_fit,
        "bg_mask": bg_mask,
        "plane": plane,
    }


def compute_layer_feature(
    image_rgb: np.ndarray,
    image_info: dict,
    annotation: dict,
    flake_mask: np.ndarray,
    ignore_mask: np.ndarray,
    layer_gt: int | None,
    roi_scale: float,
) -> LayerFeature | None:
    file_name = image_info["file_name"]
    wb_pair = parse_wb_from_name(file_name)
    if wb_pair is None:
        return None
    wb1, wb2 = wb_pair

    bbox = annotation["bbox"]
    roi = expanded_roi(image_rgb.shape[:2], bbox, scale=roi_scale)
    image_crop = crop(image_rgb, roi)
    flake_crop = crop(flake_mask, roi)
    ignore_crop = crop(ignore_mask, roi)
    area_px = int(np.count_nonzero(flake_crop))
    if area_px == 0:
        return None

    bg_result = process_green_background(image_crop, ignore_crop)
    g_corr = bg_result["g_corr"]
    g_bg_fit = bg_result["g_bg_fit"]
    bg_mask = bg_result["bg_mask"]

    g_flake_peak = estimate_peak(g_corr[flake_crop], num_bins=50)
    g_bg_median = float(np.median(g_corr[bg_mask])) if np.count_nonzero(bg_mask) else 0.0
    bg_at_flake = float(np.median(g_bg_fit[flake_crop])) if area_px else 0.0
    delta_g = float(g_flake_peak - g_bg_median)
    normalized_delta_g = 0.0 if abs(bg_at_flake) < 1e-6 else float(delta_g / bg_at_flake)

    x, y, w, h = [float(value) for value in bbox]
    return LayerFeature(
        image_id=int(image_info["id"]),
        ann_id=int(annotation["id"]),
        file_name=file_name,
        group_id=image_group_id(file_name),
        layer_gt=layer_gt,
        wb1=float(wb1),
        wb2=float(wb2),
        g_flake_peak=float(g_flake_peak),
        g_bg_median=g_bg_median,
        delta_g=delta_g,
        normalized_delta_g=normalized_delta_g,
        area_px=area_px,
        log_area_px=float(np.log1p(area_px)),
        bbox_x=x,
        bbox_y=y,
        bbox_w=w,
        bbox_h=h,
    )


def extract_features_from_coco(
    image_root: str,
    coco_json_path: str,
    layer_min: int = 0,
    layer_max: int = 10,
    roi_scale: float = 3.0,
    verbose: bool = True,
) -> list[LayerFeature]:
    image_root = os.path.expanduser(image_root)
    coco_json_path = os.path.expanduser(coco_json_path)
    coco_dict = load_coco(coco_json_path)
    category_to_layer = build_category_to_layer(coco_dict)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco_dict.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output: list[LayerFeature] = []
    images = coco_dict.get("images", [])
    for image_index, image_info in enumerate(images, start=1):
        annotations = annotations_by_image.get(int(image_info["id"]), [])
        if not annotations:
            continue

        image_path = os.path.join(image_root, image_info["file_name"])
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            if verbose:
                print(f"Skipping unreadable image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        masks_by_ann_id = {}
        ignore_mask = np.zeros((height, width), dtype=bool)
        for annotation in annotations:
            mask = ann_to_mask(annotation, height, width)
            masks_by_ann_id[int(annotation["id"])] = mask
            ignore_mask |= mask

        for annotation in annotations:
            layer = category_to_layer.get(int(annotation.get("category_id", -1)))
            if layer is None or layer < layer_min or layer > layer_max:
                continue
            feature = compute_layer_feature(
                image_rgb=image_rgb,
                image_info=image_info,
                annotation=annotation,
                flake_mask=masks_by_ann_id[int(annotation["id"])],
                ignore_mask=ignore_mask,
                layer_gt=layer,
                roi_scale=roi_scale,
            )
            if feature is not None:
                output.append(feature)

        if verbose and image_index % 25 == 0:
            print(f"Processed {image_index}/{len(images)} images, features={len(output)}")

    return output


def feature_matrix(features: list[LayerFeature], feature_names: Iterable[str]) -> np.ndarray:
    rows = []
    for feature in features:
        rows.append([float(getattr(feature, name)) for name in feature_names])
    return np.asarray(rows, dtype=np.float32)


def labels_array(features: list[LayerFeature]) -> np.ndarray:
    return np.asarray([int(feature.layer_gt) for feature in features], dtype=np.int64)


def save_features_csv(features: list[LayerFeature], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(LayerFeature.__dataclass_fields__.keys())
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            writer.writerow({name: getattr(feature, name) for name in fieldnames})

