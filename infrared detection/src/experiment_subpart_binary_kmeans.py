#!/usr/bin/env python3
"""Binary KMeans ABC/stacking shade detection inside final INFRA subparts.

This standalone script is the final post-segmentation shade/stacking detector.
It does not modify the subpart segmentation pipeline. It starts from saved final
subpart masks, runs a fixed binary KMeans shade split inside each subpart, and
filters the split with simple interpretable rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


CLASS_COLORS = {
    1: np.array([180, 95, 35], dtype=np.uint8),   # dark class: blue in BGR
    2: np.array([35, 110, 230], dtype=np.uint8),  # bright class: orange in BGR
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("D:/Desktop/infrared/infrared detection.v2-pair_v2.coco-segmentation/train"),
        help="COCO image folder containing the raw/infra image files.",
    )
    ap.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("training_log/infrared_detection/v13_aggressive_raw_canny_gate"),
        help="Pipeline output folder with pair_registration_decisions.csv and per_pair/*_masks.npz.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("training_log/infrared_detection/subpart_binary_kmeans_abc_final"),
        help="Output folder for CSVs and visualizations.",
    )
    ap.add_argument("--erode-px", type=int, default=2)
    ap.add_argument("--trim-low", type=float, default=10.0)
    ap.add_argument("--trim-high", type=float, default=95.0)
    ap.add_argument("--gaussian-sigma", type=float, default=0.5)
    ap.add_argument("--max-gray-delta", type=float, default=55.0)
    ap.add_argument(
        "--min-gray-delta",
        type=float,
        default=0.0,
        help="Optional lower bound. Default 0 keeps only the user-requested max-gray-delta gate.",
    )
    ap.add_argument("--min-minority-frac", type=float, default=0.25)
    ap.add_argument("--min-effective-px", type=int, default=20)
    return ap.parse_args()


def infer_pair_name(raw_file: str) -> str:
    return raw_file.split("_raw_")[0]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_decisions(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def erode_mask(mask: np.ndarray, px: int, min_effective_px: int) -> np.ndarray:
    mask = mask.astype(bool)
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded if int(eroded.sum()) >= min_effective_px else mask


def bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def run_binary_kmeans(
    gray_raw: np.ndarray,
    gray_smooth: np.ndarray,
    inner_mask: np.ndarray,
    trim_low: float,
    trim_high: float,
    min_effective_px: int,
    seed: int,
) -> Optional[dict]:
    valid = inner_mask.copy().astype(bool)
    raw_vals_all = gray_raw[valid].astype(np.float32)
    if raw_vals_all.size < min_effective_px:
        return None

    q_low = float(np.percentile(raw_vals_all, trim_low))
    q_high = float(np.percentile(raw_vals_all, trim_high))
    valid &= (gray_raw >= q_low) & (gray_raw <= q_high)
    if int(valid.sum()) < min_effective_px:
        return None

    smooth_vals = gray_smooth[valid].astype(np.float32)
    if len(np.unique(smooth_vals)) < 2:
        return None

    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.04)
    compactness, labels, centers = cv2.kmeans(
        smooth_vals.reshape(-1, 1).astype(np.float32),
        2,
        None,
        criteria,
        12,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.reshape(-1)
    order = np.argsort(centers)
    remap = np.zeros(2, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx + 1

    mapped_labels = remap[labels.reshape(-1)]
    class_map = np.zeros(gray_raw.shape, dtype=np.uint8)
    ys, xs = np.where(valid)
    class_map[ys, xs] = mapped_labels

    effective_px = int(valid.sum())
    class_rows = []
    for class_id in (1, 2):
        class_mask = class_map == class_id
        raw_vals = gray_raw[class_mask].astype(np.float32)
        smooth_class_vals = gray_smooth[class_mask].astype(np.float32)
        if raw_vals.size == 0:
            return None
        n_cc, cc, cc_stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), 8)
        component_areas = [int(cc_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)]
        largest_component_px = max(component_areas) if component_areas else int(raw_vals.size)
        class_rows.append(
            {
                "class_id": class_id,
                "area_px": int(raw_vals.size),
                "area_frac_effective": float(raw_vals.size / max(1, effective_px)),
                "raw_mean": float(np.mean(raw_vals)),
                "raw_median": float(np.median(raw_vals)),
                "raw_p5": float(np.percentile(raw_vals, 5)),
                "raw_p95": float(np.percentile(raw_vals, 95)),
                "smooth_mean": float(np.mean(smooth_class_vals)),
                "smooth_median": float(np.median(smooth_class_vals)),
                "component_count": int(len(component_areas)),
                "largest_component_px": int(largest_component_px),
                "largest_component_frac_of_class": float(largest_component_px / max(1, int(raw_vals.size))),
                "largest_component_frac_effective": float(largest_component_px / max(1, effective_px)),
            }
        )

    gray_delta = abs(class_rows[1]["raw_median"] - class_rows[0]["raw_median"])
    smooth_delta = abs(class_rows[1]["smooth_median"] - class_rows[0]["smooth_median"])
    minority = min(class_rows, key=lambda row: row["area_frac_effective"])

    return {
        "class_map": class_map,
        "effective_mask": valid,
        "effective_px": effective_px,
        "trim_low_gray": q_low,
        "trim_high_gray": q_high,
        "compactness": float(compactness),
        "compactness_per_px": float(compactness / max(1, effective_px)),
        "gray_median_delta": float(gray_delta),
        "smooth_median_delta": float(smooth_delta),
        "minority_class": int(minority["class_id"]),
        "minority_frac": float(minority["area_frac_effective"]),
        "class_rows": class_rows,
    }


def acceptance_reason(
    result: Optional[dict],
    min_gray_delta: float,
    max_gray_delta: float,
    min_minority_frac: float,
) -> Tuple[bool, str]:
    if result is None:
        return False, "not_enough_pixels"
    delta = result["gray_median_delta"]
    minority_frac = result["minority_frac"]
    if delta < min_gray_delta:
        return False, "gray_delta_below_min"
    if delta > max_gray_delta:
        return False, "gray_delta_above_max"
    if minority_frac <= min_minority_frac:
        return False, "minority_frac_not_above_min"
    return True, "pass"


def draw_contours(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], thickness: int = 1) -> None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness, lineType=cv2.LINE_AA)


def colorize_result(gray: np.ndarray, subpart_mask: np.ndarray, result: Optional[dict]) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    outside = ~subpart_mask.astype(bool)
    out[outside] = (out[outside] * 0.45 + 245 * 0.55).astype(np.uint8)
    if result is None:
        return out
    class_map = result["class_map"]
    for class_id, color in CLASS_COLORS.items():
        m = class_map == class_id
        if np.any(m):
            out[m] = (0.22 * out[m] + 0.78 * color).astype(np.uint8)
    return out


def tile_image(img: np.ndarray, title: str, w: int = 360, h: int = 230) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas = np.full((h + 28, w, 3), 255, dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(w / max(1, iw), h / max(1, ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x0 = (w - nw) // 2
    y0 = 28 + (h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    cv2.putText(canvas, title[:56], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def text_panel(text: str, w: int = 500, h: int = 258) -> np.ndarray:
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    y = 17
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=62) or [""]:
            if y > h - 8:
                cv2.putText(canvas, "...", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 1, cv2.LINE_AA)
                return canvas
            cv2.putText(canvas, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 1, cv2.LINE_AA)
            y += 15
    return canvas


def hstack_same_height(items: Iterable[np.ndarray]) -> np.ndarray:
    images = list(items)
    height = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] < height:
            pad = np.full((height - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
    return np.hstack(padded)


def vstack_same_width(items: Iterable[np.ndarray]) -> np.ndarray:
    images = list(items)
    width = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        if im.shape[1] < width:
            pad = np.full((im.shape[0], width - im.shape[1], 3), 255, dtype=np.uint8)
            im = np.hstack([im, pad])
        padded.append(im)
    return np.vstack(padded)


def result_text(label: int, result: Optional[dict], accepted: bool, reason: str) -> str:
    if result is None:
        return f"L{label}: reject {reason}"
    rows = result["class_rows"]
    c1 = rows[0]
    c2 = rows[1]
    return (
        f"L{label}: {'ACCEPT' if accepted else 'reject'} {reason}\n"
        f"delta={result['gray_median_delta']:.1f} smooth_delta={result['smooth_median_delta']:.1f} "
        f"minority=C{result['minority_class']} {100.0 * result['minority_frac']:.1f}%\n"
        f"C1 dark: {100.0 * c1['area_frac_effective']:.1f}% med={c1['raw_median']:.1f} "
        f"cc={c1['component_count']} largest={100.0 * c1['largest_component_frac_effective']:.1f}%\n"
        f"C2 bright: {100.0 * c2['area_frac_effective']:.1f}% med={c2['raw_median']:.1f} "
        f"cc={c2['component_count']} largest={100.0 * c2['largest_component_frac_effective']:.1f}%"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_dir = args.out_dir / "per_pair"
    per_pair_dir.mkdir(parents=True, exist_ok=True)

    decisions = read_decisions(args.segmentation_dir / "pair_registration_decisions.csv")
    class_stats_rows: List[dict] = []
    summary_rows: List[dict] = []
    overview_rows: List[np.ndarray] = []

    for row in decisions:
        raw_file = row["raw_file"]
        infra_file = row["infra_file"]
        raw_id = int(row["raw_id"])
        infra_id = int(row["infra_id"])
        pair = infer_pair_name(raw_file)

        gray = read_gray(args.dataset_dir / infra_file)
        gray_smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=args.gaussian_sigma, sigmaY=args.gaussian_sigma)
        npz_path = args.segmentation_dir / "per_pair" / f"raw{raw_id}_infra{infra_id}_masks.npz"
        z = np.load(npz_path)
        final_label = z["final_label"]
        if final_label.shape != gray.shape:
            final_label = cv2.resize(final_label.astype(np.int32), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        labels = [int(x) for x in np.unique(final_label) if int(x) > 0]
        full_class_map = np.zeros_like(gray, dtype=np.uint8)
        accepted_minority_map = np.zeros_like(gray, dtype=np.int32)
        effective_mask_map = np.zeros_like(gray, dtype=np.uint8)
        detail_rows: List[np.ndarray] = []
        full_class_viz = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.55 + 245 * 0.45).astype(np.uint8)
        accepted_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        text_lines = []

        header = np.full((58, 1780, 3), 255, dtype=np.uint8)
        cv2.putText(
            header,
            f"{pair} binary shade KMeans | trim p5/p95 + Gaussian sigma={args.gaussian_sigma} + k=2",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"accept if {args.min_gray_delta:g} <= raw median gray delta <= {args.max_gray_delta:g} and minority_frac > {args.min_minority_frac:g}",
            (8, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        detail_rows.append(header)

        for label in labels:
            sub_mask = final_label == label
            inner_mask = erode_mask(sub_mask, args.erode_px, args.min_effective_px)
            result = run_binary_kmeans(
                gray,
                gray_smooth,
                inner_mask,
                args.trim_low,
                args.trim_high,
                args.min_effective_px,
                seed=1301 + raw_id * 97 + infra_id * 13 + label,
            )
            accepted, reason = acceptance_reason(
                result,
                args.min_gray_delta,
                args.max_gray_delta,
                args.min_minority_frac,
            )

            if result is not None:
                class_map = result["class_map"]
                effective_mask_map[result["effective_mask"]] = 1
                full_class_map[class_map > 0] = class_map[class_map > 0]
                for class_id, color in CLASS_COLORS.items():
                    m = class_map == class_id
                    if np.any(m):
                        full_class_viz[m] = (0.20 * full_class_viz[m] + 0.80 * color).astype(np.uint8)

                minority_mask = class_map == result["minority_class"]
                if accepted:
                    accepted_minority_map[minority_mask] = label
                    accepted_viz[minority_mask] = (0.20 * accepted_viz[minority_mask] + 0.80 * np.array([70, 220, 70], dtype=np.uint8)).astype(np.uint8)

                summary_rows.append(
                    {
                        "pair": pair,
                        "subpart_label": label,
                        "accepted": int(accepted),
                        "reason": reason,
                        "raw_gray_median_delta": result["gray_median_delta"],
                        "smooth_gray_median_delta": result["smooth_median_delta"],
                        "minority_class": result["minority_class"],
                        "minority_frac": result["minority_frac"],
                        "effective_px": result["effective_px"],
                        "trim_low_gray": result["trim_low_gray"],
                        "trim_high_gray": result["trim_high_gray"],
                        "compactness_per_px": result["compactness_per_px"],
                        "raw_file": raw_file,
                        "infra_file": infra_file,
                        "raw_id": raw_id,
                        "infra_id": infra_id,
                        "decision": row.get("decision", ""),
                        "raw_canny_final_reason": row.get("raw_canny_final_reason", ""),
                    }
                )
                for class_row in result["class_rows"]:
                    class_stats_rows.append(
                        {
                            "pair": pair,
                            "subpart_label": label,
                            "accepted": int(accepted),
                            "reason": reason,
                            "class_id_dark_to_bright": class_row["class_id"],
                            "area_px": class_row["area_px"],
                            "effective_px": result["effective_px"],
                            "area_frac_effective": class_row["area_frac_effective"],
                            "raw_mean": class_row["raw_mean"],
                            "raw_median": class_row["raw_median"],
                            "raw_p5": class_row["raw_p5"],
                            "raw_p95": class_row["raw_p95"],
                            "smooth_mean": class_row["smooth_mean"],
                            "smooth_median": class_row["smooth_median"],
                            "component_count": class_row["component_count"],
                            "largest_component_px": class_row["largest_component_px"],
                            "largest_component_frac_of_class": class_row["largest_component_frac_of_class"],
                            "largest_component_frac_effective": class_row["largest_component_frac_effective"],
                            "raw_file": raw_file,
                            "infra_file": infra_file,
                            "decision": row.get("decision", ""),
                        }
                    )
            else:
                summary_rows.append(
                    {
                        "pair": pair,
                        "subpart_label": label,
                        "accepted": 0,
                        "reason": reason,
                        "raw_gray_median_delta": np.nan,
                        "smooth_gray_median_delta": np.nan,
                        "minority_class": 0,
                        "minority_frac": np.nan,
                        "effective_px": 0,
                        "trim_low_gray": np.nan,
                        "trim_high_gray": np.nan,
                        "compactness_per_px": np.nan,
                        "raw_file": raw_file,
                        "infra_file": infra_file,
                        "raw_id": raw_id,
                        "infra_id": infra_id,
                        "decision": row.get("decision", ""),
                        "raw_canny_final_reason": row.get("raw_canny_final_reason", ""),
                    }
                )

            y0, y1, x0, x1 = bbox_from_mask(sub_mask)
            crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
            draw_contours(crop, sub_mask[y0:y1, x0:x1], (0, 255, 255), 1)
            class_crop = colorize_result(gray, sub_mask, result)[y0:y1, x0:x1].copy()
            draw_contours(class_crop, sub_mask[y0:y1, x0:x1], (0, 255, 255), 1)
            if result is not None:
                draw_contours(class_crop, result["effective_mask"][y0:y1, x0:x1], (255, 255, 255), 1)
                if accepted:
                    draw_contours(class_crop, (result["class_map"] == result["minority_class"])[y0:y1, x0:x1], (0, 255, 0), 2)

            minority_crop = crop.copy()
            if result is not None:
                minority_mask = (result["class_map"] == result["minority_class"])[y0:y1, x0:x1]
                if accepted:
                    minority_crop[minority_mask] = (
                        0.20 * minority_crop[minority_mask] + 0.80 * np.array([70, 220, 70], dtype=np.uint8)
                    ).astype(np.uint8)
                    draw_contours(minority_crop, minority_mask, (0, 255, 0), 2)
                else:
                    minority_crop[minority_mask] = (
                        0.35 * minority_crop[minority_mask] + 0.65 * np.array([160, 160, 160], dtype=np.uint8)
                    ).astype(np.uint8)
                    draw_contours(minority_crop, minority_mask, (120, 120, 120), 1)
            draw_contours(minority_crop, sub_mask[y0:y1, x0:x1], (0, 255, 255), 1)

            text = result_text(label, result, accepted, reason)
            text_lines.append(text.replace("\n", " | "))
            detail_rows.append(
                hstack_same_height(
                    [
                        tile_image(crop, f"L{label} infra crop", 300, 210),
                        tile_image(class_crop, "KMeans classes", 300, 210),
                        tile_image(minority_crop, "minority shade candidate", 300, 210),
                        text_panel(text, 560, 238),
                    ]
                )
            )

        for label in labels:
            draw_contours(full_class_viz, final_label == label, (0, 255, 255), 1)
            draw_contours(accepted_viz, final_label == label, (0, 255, 255), 1)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for label in labels:
            draw_contours(base, final_label == label, (0, 255, 255), 1)

        np.savez_compressed(
            per_pair_dir / f"{pair}_binary_kmeans_maps.npz",
            final_label=final_label,
            binary_class_map=full_class_map,
            accepted_minority_map=accepted_minority_map,
            effective_mask=effective_mask_map,
        )
        cv2.imwrite(str(per_pair_dir / f"{pair}_binary_kmeans_filtered.jpg"), vstack_same_width(detail_rows))
        overview_rows.append(
            hstack_same_height(
                [
                    tile_image(base, f"{pair} final | {row.get('decision', '')}", 360, 230),
                    tile_image(full_class_viz, "binary KMeans classes", 360, 230),
                    tile_image(accepted_viz, "accepted minority shade", 360, 230),
                    text_panel("\n".join(text_lines[:8]), 650, 258),
                ]
            )
        )

    write_csv(args.out_dir / "binary_kmeans_shade_summary.csv", summary_rows)
    write_csv(args.out_dir / "binary_kmeans_shade_class_stats.csv", class_stats_rows)

    header = np.full((64, 1760, 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        "Binary KMeans shade filter | trim p5/p95 + Gaussian sigma=0.5 + k=2",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"Default accept rule: raw median gray delta <= {args.max_gray_delta:g}, minority class fraction > {args.min_minority_frac:g}. Green marks accepted minority class.",
        (8, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "binary_kmeans_shade_overview.jpg"), vstack_same_width([header] + overview_rows))

    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "trim_p5_p95 + gaussian_s0p5 + binary_kmeans",
                "dataset_dir": str(args.dataset_dir),
                "segmentation_dir": str(args.segmentation_dir),
                "erode_px": args.erode_px,
                "trim_low": args.trim_low,
                "trim_high": args.trim_high,
                "gaussian_sigma": args.gaussian_sigma,
                "k": 2,
                "min_gray_delta": args.min_gray_delta,
                "max_gray_delta": args.max_gray_delta,
                "min_minority_frac": args.min_minority_frac,
                "min_effective_px": args.min_effective_px,
                "accepted_minority_component_gate": "not used",
            },
            f,
            indent=2,
        )

    accepted_count = sum(1 for row in summary_rows if int(row["accepted"]) == 1)
    print(args.out_dir)
    print(f"subparts={len(summary_rows)} accepted={accepted_count}")


if __name__ == "__main__":
    main()
