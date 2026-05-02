from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from layer_recognition.green_ordinal import (
    ExtractConfig,
    FLAKE_REP_MODES,
    ann_to_mask,
    build_category_to_layer,
    compute_flake_representatives,
    dilate_boolean_mask,
    estimate_background_peak,
    expand_path,
    fit_background_plane_clipped,
    load_coco,
    ndg_from_rep,
    normalize_flake_rep_mode,
    parse_wb_from_filename,
    select_flake_representative,
    trimmed_mean,
)


def parse_layers(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def read_prediction_filter(path: str | None, min_abs_error: float) -> set[int] | None:
    if not path:
        return None
    keep: set[int] = set()
    with open(expand_path(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if float(row.get("abs_error", 0.0)) >= min_abs_error:
                    keep.add(int(float(row["ann_id"])))
            except Exception:
                continue
    return keep


def normalize_gray(values: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    values = values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((values - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def draw_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 26), (255, 255, 255), thickness=-1)
    cv2.putText(output, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def resize_panel(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_visualization(
    image_bgr: np.ndarray,
    annotation_mask: np.ndarray,
    union_mask: np.ndarray,
    cfg: ExtractConfig,
    fit_source: str,
) -> tuple[np.ndarray, dict] | None:
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

    roi_slice = np.s_[y0:y1, x0:x1]
    flake_roi = annotation_mask[roi_slice]
    union_roi = union_mask[roi_slice]
    if int(np.count_nonzero(flake_roi)) < cfg.min_area_px:
        return None
    blocked_roi = dilate_boolean_mask(union_roi.astype(bool), cfg.edge_dilate_px)

    green = image_bgr[:, :, 1].astype(np.float32)
    green_crop = green[roi_slice]
    g_smooth_bg = cv2.bilateralFilter(
        green_crop,
        d=-1,
        sigmaColor=cfg.bg_bilateral_sigma_color,
        sigmaSpace=cfg.bg_bilateral_sigma_space,
    )
    g_smooth_flake = cv2.bilateralFilter(
        green_crop,
        d=-1,
        sigmaColor=cfg.flake_bilateral_sigma_color,
        sigmaSpace=cfg.flake_bilateral_sigma_space,
    )

    if fit_source == "raw":
        fit_map = green_crop
        flake_map = green_crop
        candidate_source = green_crop
    else:
        fit_map = g_smooth_bg
        flake_map = g_smooth_flake if fit_source == "training" else g_smooth_bg
        candidate_source = g_smooth_bg

    candidate_mask = candidate_source <= cfg.bg_threshold_high
    candidate_mask = candidate_mask & (~blocked_roi)
    if int(np.count_nonzero(candidate_mask)) < cfg.bg_min_pixels:
        candidate_mask = ~blocked_roi
    if int(np.count_nonzero(candidate_mask)) < cfg.bg_min_pixels:
        candidate_mask = ~union_roi.astype(bool)
    if int(np.count_nonzero(candidate_mask)) < cfg.bg_min_pixels:
        return None

    bg_peak = estimate_background_peak(candidate_source[candidate_mask], cfg.bg_peak_bins)
    background_mask = (
        candidate_mask
        & (candidate_source >= bg_peak - cfg.bg_tol_low)
        & (candidate_source <= bg_peak + cfg.bg_tol_high)
    )
    if int(np.count_nonzero(background_mask)) < cfg.bg_min_pixels:
        background_mask = candidate_mask
    if int(np.count_nonzero(background_mask)) < cfg.bg_min_pixels:
        return None

    bg_y, bg_x = np.where(background_mask)
    bg_values = fit_map[background_mask]
    initial_bg_pixels = int(bg_values.size)
    a, b, c, bg_keep, clip_iterations, residual_sigma = fit_background_plane_clipped(
        bg_x,
        bg_y,
        bg_values,
        sample_max=cfg.bg_sample_max,
        min_pixels=cfg.bg_min_pixels,
        sigma_clip=cfg.bg_residual_clip_sigma,
        sigma_floor=cfg.bg_residual_sigma_floor,
        max_iters=cfg.bg_residual_clip_iters,
    )
    clipped_background_mask = np.zeros_like(background_mask, dtype=bool)
    clipped_background_mask[bg_y[bg_keep], bg_x[bg_keep]] = True
    if int(np.count_nonzero(clipped_background_mask)) >= cfg.bg_min_pixels:
        background_mask = clipped_background_mask
        bg_y, bg_x = np.where(background_mask)
        bg_values = fit_map[background_mask]

    crop_height, crop_width = green_crop.shape
    grid_x, grid_y = np.meshgrid(np.arange(crop_width, dtype=np.float32), np.arange(crop_height, dtype=np.float32))
    plane = a * grid_x + b * grid_y + c
    hybrid = fit_map.copy()
    hybrid[union_roi.astype(bool)] = flake_map[union_roi.astype(bool)]
    delta = hybrid - plane

    plane_at_flake = plane[flake_roi]
    delta_values = delta[flake_roi].astype(np.float32)
    flake_reps = compute_flake_representatives(delta, flake_roi, cfg)
    g_flake_rep, flake_rep_source = select_flake_representative(flake_reps, cfg.flake_rep_mode)
    g_bg_corr_median = float(np.median(delta[background_mask])) if int(np.count_nonzero(background_mask)) else 0.0
    g_bg_plane_median = float(np.median(plane_at_flake))
    delta_g = float(g_flake_rep - g_bg_corr_median)
    ndg = delta_g / max(abs(g_bg_plane_median), 1.0)
    p10, p90 = np.percentile(delta_values, [10, 90])
    g_delta_iqr = float(p90 - p10)
    iqr_ndg = g_delta_iqr / max(abs(g_bg_plane_median), 1.0)
    ndg_peak = ndg_from_rep(float(flake_reps["peak"]), g_bg_corr_median, g_bg_plane_median)
    ndg_mask_median = ndg_from_rep(float(flake_reps["median"]), g_bg_corr_median, g_bg_plane_median)
    ndg_trimmed_median = ndg_from_rep(float(flake_reps["trimmed_median"]), g_bg_corr_median, g_bg_plane_median)
    ndg_central_median = ndg_from_rep(float(flake_reps["central_median"]), g_bg_corr_median, g_bg_plane_median)
    ndg_largest_inner_median = ndg_from_rep(float(flake_reps["largest_inner_median"]), g_bg_corr_median, g_bg_plane_median)

    raw_green = cv2.cvtColor(normalize_gray(green_crop), cv2.COLOR_GRAY2BGR)
    draw_contours(raw_green, flake_roi, (0, 0, 255), thickness=1)
    bilateral_green = cv2.cvtColor(normalize_gray(g_smooth_bg), cv2.COLOR_GRAY2BGR)
    draw_contours(bilateral_green, flake_roi, (0, 0, 255), thickness=1)
    plane_vis = cv2.cvtColor(normalize_gray(plane), cv2.COLOR_GRAY2BGR)
    draw_contours(plane_vis, flake_roi, (0, 0, 255), thickness=1)

    flake_delta_only = np.zeros_like(delta, dtype=np.float32)
    flake_delta_only[flake_roi] = delta[flake_roi]
    delta_vis = cv2.cvtColor(normalize_gray(flake_delta_only, p_low=0.0, p_high=100.0), cv2.COLOR_GRAY2BGR)
    draw_contours(delta_vis, flake_roi, (0, 0, 255), thickness=1)

    panel_size = 360
    panels = [
        label_panel(resize_panel(raw_green, panel_size), f"raw green channel | mask={area_px}px bbox={bbox_w}x{bbox_h}"),
        label_panel(resize_panel(bilateral_green, panel_size), "green after bilateral filter"),
        label_panel(resize_panel(plane_vis, panel_size), f"regressed background plane ({fit_source})"),
        label_panel(resize_panel(delta_vis, panel_size), f"delta | {flake_rep_source} ndg={ndg:.4f} iqr={iqr_ndg:.4f}"),
    ]
    stats = {
        "area_px": area_px,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "roi_bg_pixels": int(np.count_nonzero(background_mask)),
        "roi_bg_initial_pixels": int(initial_bg_pixels),
        "roi_bg_clip_iterations": int(clip_iterations),
        "roi_bg_residual_sigma": float(residual_sigma),
        "flake_rep_mode": cfg.flake_rep_mode,
        "flake_rep_source": str(flake_rep_source),
        "ndg": float(ndg),
        "ndg_peak": float(ndg_peak),
        "ndg_mask_median": float(ndg_mask_median),
        "ndg_trimmed_median": float(ndg_trimmed_median),
        "ndg_central_median": float(ndg_central_median),
        "ndg_largest_inner_median": float(ndg_largest_inner_median),
        "delta_g": float(delta_g),
        "g_flake_rep": float(g_flake_rep),
        "g_flake_peak": float(flake_reps["peak"]),
        "g_flake_mask_median": float(flake_reps["median"]),
        "g_flake_trimmed_median": float(flake_reps["trimmed_median"]),
        "g_flake_central_median": float(flake_reps["central_median"]),
        "g_flake_largest_inner_median": float(flake_reps["largest_inner_median"]),
        "g_flake_median": float(g_flake_rep),
        "flake_inner_area_px": int(flake_reps["inner_area_px"]),
        "flake_largest_inner_area_px": int(flake_reps["largest_inner_area_px"]),
        "g_bg_plane_median": g_bg_plane_median,
        "g_delta_p10": float(p10),
        "g_delta_p90": float(p90),
        "g_delta_iqr": g_delta_iqr,
        "iqr_ndg": float(iqr_ndg),
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_c": float(c),
    }
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    canvas = np.vstack([top, bottom])
    return canvas, stats


def visualize(args: argparse.Namespace) -> None:
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
    image_dir = expand_path(args.image_dir)
    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(args.coco)
    category_to_layer = build_category_to_layer(coco)
    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    layer_filter = parse_layers(args.layers)
    ann_filter = read_prediction_filter(args.predictions_csv, args.min_abs_error)
    index_rows = []
    exported = 0

    for image_id, annotations in annotations_by_image.items():
        if exported >= args.max_annotations:
            break
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
        decoded = []
        union_mask = np.zeros((height, width), dtype=bool)
        for annotation in annotations:
            layer = category_to_layer.get(int(annotation.get("category_id", -1)))
            if layer is None:
                continue
            if layer_filter is not None and layer not in layer_filter:
                continue
            if ann_filter is not None and int(annotation.get("id", -1)) not in ann_filter:
                continue
            mask = ann_to_mask(annotation, height, width)
            if int(np.count_nonzero(mask)) < cfg.min_area_px:
                continue
            decoded.append((annotation, layer, mask))

        if not decoded:
            continue
        for annotation in annotations:
            try:
                union_mask |= ann_to_mask(annotation, height, width)
            except Exception:
                continue

        wb1, wb2 = parse_wb_from_filename(file_name)
        for annotation, layer, mask in decoded:
            if exported >= args.max_annotations:
                break
            result = make_visualization(image_bgr, mask, union_mask, cfg, args.fit_source)
            if result is None:
                continue
            canvas, stats = result
            ann_id = int(annotation.get("id", -1))
            out_name = (
                f"{exported + 1:04d}_layer{layer}_ann{ann_id}_"
                f"wb{wb1:.1f}_{wb2:.1f}_{Path(file_name).stem}.jpg"
            )
            out_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in out_name)
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), canvas)
            row = {
                "visualization": str(out_path),
                "source_image": str(image_path),
                "filename": Path(file_name).name,
                "image_id": int(image_id),
                "ann_id": ann_id,
                "layer": int(layer),
                "wb1": float(wb1),
                "wb2": float(wb2),
                **stats,
            }
            index_rows.append(row)
            exported += 1

    if index_rows:
        with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
    print(f"[INFO] Exported {exported} green preprocessing visualizations to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize green-channel preprocessing used by train_green_ordinal.py.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default=None, help="Optional comma-separated layer filter, e.g. 3,4.")
    parser.add_argument("--predictions-csv", default=None, help="Optional predictions CSV; exports only annotations with abs_error above threshold.")
    parser.add_argument("--min-abs-error", type=float, default=2.0)
    parser.add_argument("--max-annotations", type=int, default=80)
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
        default="peak",
        help=f"Flake representative for ndg. Valid: {', '.join(FLAKE_REP_MODES)}. Alias: mode=peak.",
    )
    parser.add_argument("--flake-peak-bins", type=int, default=50)
    parser.add_argument("--flake-inner-erode-px", type=int, default=2)
    parser.add_argument("--flake-min-inner-pixels", type=int, default=20)
    parser.add_argument("--trim-low", type=float, default=10.0)
    parser.add_argument("--trim-high", type=float, default=90.0)
    parser.add_argument(
        "--fit-source",
        choices=["training", "raw", "bilateral"],
        default="training",
        help="training matches green_ordinal.py: background bilateral for plane, flake bilateral inside masks.",
    )
    args = parser.parse_args()
    args.flake_rep_mode = normalize_flake_rep_mode(args.flake_rep_mode)
    return args


if __name__ == "__main__":
    visualize(parse_args())
