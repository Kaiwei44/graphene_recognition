#!/usr/bin/env python3
"""Calibrate raw-channel evidence thresholds from COCO subpart annotations.

This script uses manual RAW subpart labels as positive boundaries and RAW
within-subpart quantile edges as negative pseudo-boundaries. The output is a
small set of robust contrast/gradient statistics for deciding when an infra
candidate boundary has enough RAW support.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt


def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-zip", type=Path)
    src.add_argument("--dataset-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raw-prefix", default="raw_png")
    ap.add_argument("--big-category", default="gra")
    ap.add_argument("--subpart-category", default="subparts")
    ap.add_argument("--window-radius", type=int, default=5)
    ap.add_argument("--pseudo-classes", type=int, default=4)
    return ap.parse_args()


def prepare_dataset(args):
    if args.dataset_dir:
        return args.dataset_dir, None
    tmp = tempfile.TemporaryDirectory(prefix="raw_boundary_calib_")
    with zipfile.ZipFile(args.input_zip) as z:
        z.extractall(tmp.name)
    return Path(tmp.name), tmp


def find_ann(root: Path) -> Path:
    hits = sorted(root.rglob("_annotations.coco.json"))
    if not hits:
        raise FileNotFoundError("No _annotations.coco.json found")
    return hits[0]


def image_path(root: Path, ann_path: Path, fn: str) -> Path:
    p = ann_path.parent / fn
    if p.exists():
        return p
    hits = list(root.rglob(fn))
    if not hits:
        raise FileNotFoundError(fn)
    return hits[0]


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def poly_mask(h: int, w: int, ann: Dict) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for seg in ann.get("segmentation", []):
        pts = np.array(seg, np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 255)
    return m


def masks_for(im: Dict, anns_by: Dict[int, List[Dict]], cat_id: int) -> List[np.ndarray]:
    out = []
    for ann in anns_by.get(int(im["id"]), []):
        if int(ann.get("category_id", -1)) == int(cat_id):
            out.append(poly_mask(im["height"], im["width"], ann))
    return out


def union_masks(ms: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    for m in ms:
        out[m > 0] = 255
    return out


def relabel(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label, np.int32)
    labs = [int(x) for x in np.unique(label) if x > 0]
    for i, lab in enumerate(labs, 1):
        out[label == lab] = i
    return out


def fill_zero_inside(label: np.ndarray, inside: np.ndarray) -> np.ndarray:
    out = label.copy().astype(np.int32)
    need = (out == 0) & inside
    if need.any() and (out > 0).any():
        _, inds = distance_transform_edt(out == 0, return_indices=True)
        near = out[tuple(inds)]
        out[need] = near[need]
    return out


def label_from_masks(masks: Sequence[np.ndarray], inside: np.ndarray) -> np.ndarray:
    lab = np.zeros_like(inside, np.int32)
    for i, m in enumerate(masks, 1):
        lab[(m > 0) & inside] = i
    if lab[inside].max() > 0:
        lab = fill_zero_inside(lab, inside)
    return relabel(lab)


def corrected_l_channel(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    if inside.sum() < 10:
        return L
    sigma = max(9.0, min(rgb.shape[:2]) / 7.0)
    bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corr = L - bg + np.median(L[inside])
    return cv2.bilateralFilter(corr.astype(np.float32), 5, 7, 7)


def channels_for(rgb: np.ndarray, inside: np.ndarray) -> Dict[str, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {
        "R": rgb[:, :, 0].astype(np.float32),
        "G": rgb[:, :, 1].astype(np.float32),
        "B": rgb[:, :, 2].astype(np.float32),
        "gray": gray,
        "LabL": lab[:, :, 0],
        "Lcorr": corrected_l_channel(rgb, inside),
        "Laba": lab[:, :, 1],
        "Labb": lab[:, :, 2],
    }


def robust_scale(ch: np.ndarray, inside: np.ndarray) -> float:
    vals = ch[inside]
    if len(vals) < 10:
        return 1.0
    q10, q90 = np.percentile(vals, [10, 90])
    return max(1.0, float(q90 - q10))


def gradients(channels: Dict[str, np.ndarray], inside: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, ch in channels.items():
        gx = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
        denom = float(np.percentile(g[inside], 75)) if inside.sum() else 1.0
        out[name] = g / max(1.0, denom)
    return out


def adjacent_edges(label: np.ndarray, valid: np.ndarray) -> List[Tuple[int, int, int, int, int, int]]:
    edges = []
    a = label[:, :-1]
    b = label[:, 1:]
    m = (a != b) & (a > 0) & (b > 0) & valid[:, :-1] & valid[:, 1:]
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        la, lb = int(label[y, x]), int(label[y, x + 1])
        edges.append((min(la, lb), max(la, lb), y, x, y, x + 1))
    a = label[:-1, :]
    b = label[1:, :]
    m = (a != b) & (a > 0) & (b > 0) & valid[:-1, :] & valid[1:, :]
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        la, lb = int(label[y, x]), int(label[y + 1, x])
        edges.append((min(la, lb), max(la, lb), y, x, y + 1, x))
    return edges


def pseudo_label_from_raw(ch: np.ndarray, inside: np.ndarray, classes: int) -> np.ndarray:
    vals = ch[inside]
    out = np.zeros_like(ch, np.int32)
    if len(vals) < 100 or classes <= 1:
        out[inside] = 1
        return out
    th = np.percentile(vals, np.linspace(0, 100, classes + 1)[1:-1])
    out[inside] = np.digitize(ch[inside], th) + 1
    return out


def summarize_boundary(
    image_key: str,
    label: np.ndarray,
    gra: np.ndarray,
    channels: Dict[str, np.ndarray],
    grad: Dict[str, np.ndarray],
    edge_list: Sequence[Tuple[int, int, int, int, int, int]],
    boundary_type: str,
    window_radius: int,
    gt_label: np.ndarray | None = None,
) -> List[Dict]:
    inside = gra > 0
    scale = {name: robust_scale(ch, inside) for name, ch in channels.items()}
    by_pair: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
    for a, b, y1, x1, y2, x2 in edge_list:
        by_pair[(a, b)].append((y1, x1, y2, x2))

    rows = []
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * window_radius + 1, 2 * window_radius + 1))
    for (a, b), pix_edges in by_pair.items():
        edge_mask = np.zeros_like(label, bool)
        y1 = np.array([e[0] for e in pix_edges])
        x1 = np.array([e[1] for e in pix_edges])
        y2 = np.array([e[2] for e in pix_edges])
        x2 = np.array([e[3] for e in pix_edges])
        edge_mask[y1, x1] = True
        edge_mask[y2, x2] = True
        band = binary_dilation(edge_mask, structure=dilate_kernel) & inside
        side_a = band & (label == a)
        side_b = band & (label == b)
        if side_a.sum() < 8 or side_b.sum() < 8:
            continue

        row = {
            "image": image_key,
            "boundary_type": boundary_type,
            "label_a": a,
            "label_b": b,
            "edge_count": len(pix_edges),
            "side_a_area": int(side_a.sum()),
            "side_b_area": int(side_b.sum()),
        }
        if gt_label is not None:
            gt_vals = gt_label[edge_mask & (gt_label > 0)]
            row["gt_labels_touched"] = int(len(set(int(x) for x in gt_vals)))
        side_max = 0.0
        edge_max = 0.0
        grad_max = 0.0
        rgb_side_max = 0.0
        for name, ch in channels.items():
            side = abs(float(np.median(ch[side_a])) - float(np.median(ch[side_b]))) / scale[name]
            edge = np.abs(ch[y1, x1] - ch[y2, x2]) / scale[name]
            edge_med = float(np.median(edge))
            g_med = float(np.median(np.r_[grad[name][y1, x1], grad[name][y2, x2]]))
            row[f"side_{name}"] = side
            row[f"edge_{name}"] = edge_med
            row[f"grad_{name}"] = g_med
            side_max = max(side_max, side)
            edge_max = max(edge_max, edge_med)
            grad_max = max(grad_max, g_med)
            if name in {"R", "G", "B", "gray", "LabL", "Lcorr"}:
                rgb_side_max = max(rgb_side_max, side)
        row["side_max_all"] = side_max
        row["side_max_raw_primary"] = rgb_side_max
        row["edge_max_all"] = edge_max
        row["grad_max_all"] = grad_max
        row["raw_boundary_support"] = rgb_side_max + 0.05 * min(4.0, max(0.0, grad_max - 1.0))
        rows.append(row)
    return rows


def quantiles(s: pd.Series) -> Dict[str, float]:
    qs = [5, 10, 25, 50, 75, 90, 95]
    return {f"q{q}": float(np.percentile(s.dropna(), q)) for q in qs if len(s.dropna())}


def threshold_table(pos: pd.Series, neg: pd.Series) -> pd.DataFrame:
    vals = np.unique(np.r_[np.percentile(pos, np.linspace(5, 95, 19)), np.percentile(neg, np.linspace(50, 99, 20))])
    rows = []
    for t in vals:
        tp = int((pos >= t).sum())
        fn = int((pos < t).sum())
        fp = int((neg >= t).sum())
        tn = int((neg < t).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        rows.append(dict(threshold=float(t), precision=precision, recall=recall, f1=f1, fp_rate=fp / max(1, fp + tn)))
    return pd.DataFrame(rows).sort_values(["f1", "threshold"], ascending=[False, True])


def add_title(img: np.ndarray, title: str, h: int = 28) -> np.ndarray:
    pil = Image.fromarray(img)
    can = Image.new("RGB", (pil.width, pil.height + h), "white")
    can.paste(pil, (0, h))
    d = ImageDraw.Draw(can)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = None
    d.text((5, 6), title, fill="black", font=font)
    return np.array(can)


def heatmap(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = score.copy()
    vals = s[mask > 0]
    if len(vals):
        lo, hi = np.percentile(vals, [5, 98])
        s = np.clip((s - lo) / max(1e-6, hi - lo), 0, 1)
    im = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im[mask == 0] = 255
    return im


def overlay_boundary(rgb: np.ndarray, label: np.ndarray, color=(255, 255, 0)) -> np.ndarray:
    out = rgb.copy()
    edge = np.zeros(label.shape, bool)
    for _, _, y1, x1, y2, x2 in adjacent_edges(label, label > 0):
        edge[y1, x1] = True
        edge[y2, x2] = True
    out[edge] = color
    return out


def main():
    args = parse_args()
    root, tmp = prepare_dataset(args)
    ann_path = find_ann(root)
    data = json.loads(ann_path.read_text())
    images = {int(im["id"]): im for im in data["images"]}
    anns_by: Dict[int, List[Dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by[int(ann["image_id"])].append(ann)
    cats = {c["name"]: int(c["id"]) for c in data.get("categories", [])}
    big_id = cats[args.big_category]
    sub_id = cats[args.subpart_category]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    panels = []
    for im in images.values():
        if args.raw_prefix not in im["file_name"]:
            continue
        rgb = read_rgb(image_path(root, ann_path, im["file_name"]))
        gra = union_masks(masks_for(im, anns_by, big_id), rgb.shape[:2])
        inside = gra > 0
        sub_masks = masks_for(im, anns_by, sub_id)
        gt = label_from_masks(sub_masks, inside)
        if int(gt.max()) <= 1:
            continue

        channels = channels_for(rgb, inside)
        grad = gradients(channels, inside)
        true_edges = adjacent_edges(gt, inside)
        rows.extend(summarize_boundary(im["file_name"], gt, gra, channels, grad, true_edges, "true_subpart_boundary", args.window_radius))

        pseudo = pseudo_label_from_raw(channels["Lcorr"], inside, args.pseudo_classes)
        gt_boundary = overlay_boundary(np.zeros_like(rgb), gt)[:, :, 0] > 0
        far_from_true = distance_transform_edt(~gt_boundary) > 4
        pseudo_edges = []
        for e in adjacent_edges(pseudo, inside & far_from_true):
            _, _, y1, x1, y2, x2 = e
            if gt[y1, x1] == gt[y2, x2]:
                pseudo_edges.append(e)
        rows.extend(summarize_boundary(im["file_name"], pseudo, gra, channels, grad, pseudo_edges, "within_subpart_pseudo_boundary", args.window_radius, gt))

        raw_support_map = np.maximum.reduce([grad[k] for k in ["R", "G", "B", "gray", "LabL", "Lcorr"]])
        panel = np.hstack([
            add_title(rgb, "raw"),
            add_title(overlay_boundary(rgb, gt), "GT subpart boundary"),
            add_title(heatmap(raw_support_map, gra), "raw gradient support"),
            add_title(overlay_boundary(rgb, pseudo, (255, 0, 0)), "raw pseudo boundaries"),
        ])
        panels.append(panel)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "raw_boundary_features.csv", index=False)

    report = {
        "window_radius": args.window_radius,
        "pseudo_classes": args.pseudo_classes,
        "rows": int(len(df)),
        "positive_boundaries": int((df["boundary_type"] == "true_subpart_boundary").sum()) if len(df) else 0,
        "negative_pseudo_boundaries": int((df["boundary_type"] == "within_subpart_pseudo_boundary").sum()) if len(df) else 0,
        "features": {},
        "recommended": {},
    }
    if len(df):
        pos = df[df["boundary_type"] == "true_subpart_boundary"]
        neg = df[df["boundary_type"] == "within_subpart_pseudo_boundary"]
        for feat in ["side_max_raw_primary", "edge_max_all", "grad_max_all", "raw_boundary_support", "side_R", "side_G", "side_B", "side_gray", "side_Lcorr"]:
            report["features"][feat] = {
                "positive": quantiles(pos[feat]),
                "negative": quantiles(neg[feat]) if len(neg) else {},
            }
            if len(pos) and len(neg):
                tt = threshold_table(pos[feat], neg[feat])
                tt.to_csv(args.out_dir / f"threshold_sweep_{feat}.csv", index=False)
                best = tt.iloc[0].to_dict()
                report["features"][feat]["best_f1_threshold"] = best
        if len(pos) and len(neg):
            support_sweep = threshold_table(pos["raw_boundary_support"], neg["raw_boundary_support"])
            side_sweep = threshold_table(pos["side_max_raw_primary"], neg["side_max_raw_primary"])
            grad_sweep = threshold_table(pos["grad_max_all"], neg["grad_max_all"])
            report["recommended"] = {
                "raw_boundary_support_best_f1": float(support_sweep.iloc[0]["threshold"]),
                "raw_boundary_support_weak_q10": float(np.percentile(pos["raw_boundary_support"], 10)),
                "raw_boundary_support_medium_q25": float(np.percentile(pos["raw_boundary_support"], 25)),
                "raw_boundary_support_strong_q50": float(np.percentile(pos["raw_boundary_support"], 50)),
                "side_contrast_best_f1": float(side_sweep.iloc[0]["threshold"]),
                "side_contrast_weak_q10": float(np.percentile(pos["side_max_raw_primary"], 10)),
                "side_contrast_medium_q25": float(np.percentile(pos["side_max_raw_primary"], 25)),
                "side_contrast_strong_q50": float(np.percentile(pos["side_max_raw_primary"], 50)),
                "gradient_ratio_best_f1": float(grad_sweep.iloc[0]["threshold"]),
                "gradient_ratio_weak_q10": float(np.percentile(pos["grad_max_all"], 10)),
                "gradient_ratio_medium_q25": float(np.percentile(pos["grad_max_all"], 25)),
                "gradient_ratio_strong_q50": float(np.percentile(pos["grad_max_all"], 50)),
                "min_boundary_edge_count": int(max(8, math.floor(np.percentile(pos["edge_count"], 10)))),
            }

    (args.out_dir / "raw_boundary_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if panels:
        width = max(p.shape[1] for p in panels)
        height = sum(p.shape[0] + 8 for p in panels)
        sheet = np.ones((height, width, 3), np.uint8) * 255
        y = 0
        for p in panels:
            sheet[y:y + p.shape[0], :p.shape[1]] = p
            y += p.shape[0] + 8
        Image.fromarray(sheet).save(args.out_dir / "raw_boundary_calibration_overview.jpg", quality=92)

    print(f"Saved {args.out_dir}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
