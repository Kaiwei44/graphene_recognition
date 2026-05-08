#!/usr/bin/env python3
"""v3 non-DL raw subpart segmentation.

Current mainline baseline:
Lab-L illumination correction + graph superpixels + conservative brightness class merge.

Inputs:
  - Roboflow/COCO zip or extracted COCO directory
  - COCO large-block annotations, usually category 'gra'
  - optional manual subpart annotations, usually category 'subparts'

Outputs:
  - per-block predicted masks and overlays
  - per-image label masks
  - evaluation metrics if GT subparts are available
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_multiotsu
from skimage.segmentation import felzenszwalb, find_boundaries
from sklearn.metrics import adjusted_rand_score


COLORS = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


@dataclass
class V3Params:
    # Main segmentation controls.
    gaussian_sigma_divisor: float = 7.0
    gaussian_sigma_min: float = 9.0
    bilateral_d: int = 5
    bilateral_sigma_color: float = 7.0
    bilateral_sigma_space: float = 7.0

    # Contrast thresholds on corrected Lab-L values. Conservative by design.
    contrast_low: float = 7.0
    contrast_high: float = 16.0
    max_classes: int = 3

    # Superpixel controls.
    felz_scale_small: float = 55.0
    felz_scale_large: float = 70.0
    felz_sigma: float = 0.45
    felz_min_size_area_divisor: float = 900.0
    felz_min_size_floor: int = 18
    large_area_threshold: int = 12000

    # Cleanup controls.
    min_component_area_fraction: float = 0.012
    min_component_area_floor: int = 25
    bad_pixel_dilate: int = 1


@dataclass
class BlockMetrics:
    image_id: int
    file_name: str
    block_index: int
    gt_regions: int
    pred_regions: int
    ari: float
    cover_gt: float
    cover_pred: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3 non-DL raw subpart segmentation")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-zip", type=Path, help="COCO/Roboflow zip file")
    src.add_argument("--dataset-dir", type=Path, help="Extracted COCO dataset directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--raw-prefix", default="raw_png", help="Only image filenames containing this substring are processed")
    parser.add_argument("--big-category", default="gra", help="COCO category name for upstream large block masks")
    parser.add_argument("--subpart-category", default="subparts", help="Optional COCO category name for manual subparts")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted zip contents in out-dir/extracted_dataset")
    parser.add_argument("--overview-width", type=int, default=1200, help="Width of each row in overview contact sheet")
    return parser.parse_args()


def find_annotation_file(root: Path) -> Path:
    candidates = list(root.rglob("_annotations.coco.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not find _annotations.coco.json under {root}")
    # Prefer train/_annotations.coco.json if present.
    candidates = sorted(candidates, key=lambda p: ("train" not in str(p.parent), len(str(p))))
    return candidates[0]


def prepare_dataset(args: argparse.Namespace) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if args.dataset_dir is not None:
        return args.dataset_dir, None
    tmp = tempfile.TemporaryDirectory(prefix="infra_v3_")
    with zipfile.ZipFile(args.input_zip) as zf:
        zf.extractall(tmp.name)
    root = Path(tmp.name)
    return root, tmp


def load_coco(ann_path: Path) -> Tuple[Dict, Dict[int, Dict], Dict[int, List[Dict]], Dict[str, int]]:
    data = json.loads(ann_path.read_text())
    images = {int(im["id"]): im for im in data["images"]}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    cat_name_to_id = {c["name"]: int(c["id"]) for c in data.get("categories", [])}
    return data, images, anns_by_image, cat_name_to_id


def image_path(dataset_root: Path, ann_path: Path, file_name: str) -> Path:
    p = ann_path.parent / file_name
    if p.exists():
        return p
    hits = list(dataset_root.rglob(file_name))
    if not hits:
        raise FileNotFoundError(f"Could not find image file {file_name}")
    return hits[0]


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def polygon_to_mask(height: int, width: int, ann: Dict) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    seg = ann.get("segmentation", [])
    if isinstance(seg, list):
        for poly in seg:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
    else:
        raise ValueError("Only polygon segmentations are supported in this lightweight script")
    return mask


def anns_of_category(anns: Sequence[Dict], category_id: Optional[int]) -> List[Dict]:
    if category_id is None:
        return []
    return [a for a in anns if int(a.get("category_id", -1)) == int(category_id)]


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return 0, 0, 1, 1
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def detect_bad_pixels(rgb: np.ndarray, params: V3Params) -> np.ndarray:
    """Detect red guide lines and black scale/text pixels to exclude from statistics."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    red = (r > 130) & (g < 105) & (b < 125) & ((r.astype(int) - g.astype(int)) > 35) & ((r.astype(int) - b.astype(int)) > 30)
    dark = (r < 36) & (g < 36) & (b < 36)
    bad = (red | dark).astype(np.uint8) * 255
    if params.bad_pixel_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bad = cv2.dilate(bad, k, iterations=params.bad_pixel_dilate)
    return bad > 0


def relabel_sequential(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label, dtype=np.int32)
    labs = [int(x) for x in np.unique(label) if x > 0]
    for i, lab in enumerate(labs, 1):
        out[label == lab] = i
    return out


def fill_zero_inside(label: np.ndarray, inside: np.ndarray) -> np.ndarray:
    out = label.copy().astype(np.int32)
    need = (out == 0) & inside
    if need.any() and (out > 0).any():
        _, inds = distance_transform_edt(out == 0, return_indices=True)
        nearest = out[tuple(inds)]
        out[need] = nearest[need]
    return out


def normalize_channel(ch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(ch[valid], [1, 99])
    denom = max(1.0, float(p99 - p1))
    return np.clip((ch - p1) / denom, 0.0, 1.0)


def choose_num_classes(vals: np.ndarray, params: V3Params) -> int:
    if len(vals) < 80:
        return 1
    p10, p90 = np.percentile(vals, [10, 90])
    contrast = float(p90 - p10)
    if contrast < params.contrast_low:
        return 1
    if contrast < params.contrast_high:
        return 2
    return min(3, params.max_classes)


def connected_cleanup(classmap: np.ndarray, inside: np.ndarray, min_area: int) -> np.ndarray:
    cm = classmap.astype(np.uint8).copy()
    cm[~inside] = 0
    out = np.zeros_like(cm, dtype=np.int32)
    next_id = 1
    for cls in [int(x) for x in np.unique(cm) if x > 0]:
        n, cc, stats, _ = cv2.connectedComponentsWithStats((cm == cls).astype(np.uint8), 8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = list(np.where(areas >= min_area)[0] + 1)
        if not keep and len(areas):
            keep = [int(np.argmax(areas) + 1)]
        for k in keep:
            out[cc == k] = next_id
            next_id += 1
    if out.max() == 0:
        out[inside] = 1
    else:
        out = fill_zero_inside(out, inside)
    return relabel_sequential(out)


def segment_block_v3(rgb: np.ndarray, block_mask: np.ndarray, params: V3Params) -> np.ndarray:
    """Split one large block into subparts using v3 graph-superpixel method."""
    inside = block_mask > 0
    h, w = inside.shape
    area = int(inside.sum())
    out_empty = np.zeros((h, w), dtype=np.int32)
    if area < 80:
        out_empty[inside] = 1
        return out_empty

    erode_radius = max(1, int(round(math.sqrt(area) / 85)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
    inner = cv2.erode(inside.astype(np.uint8), k, iterations=1).astype(bool)
    bad = detect_bad_pixels(rgb, params)
    valid = inner & (~bad)
    if valid.sum() < 80:
        valid = inside & (~bad)
    if valid.sum() < 80:
        valid = inside

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    sigma = max(params.gaussian_sigma_min, min(h, w) / params.gaussian_sigma_divisor)
    bg = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
    L_corr = L - bg + np.median(L[valid])
    L_smooth = cv2.bilateralFilter(
        L_corr.astype(np.float32),
        params.bilateral_d,
        params.bilateral_sigma_color,
        params.bilateral_sigma_space,
    )

    vals = L_smooth[valid]
    nclass = choose_num_classes(vals, params)
    if nclass == 1:
        out_empty[inside] = 1
        return out_empty

    fL = normalize_channel(L_smooth, valid)
    fa = normalize_channel(lab[:, :, 1], valid)
    fb = normalize_channel(lab[:, :, 2], valid)
    # Lab-L dominates. a,b are weak stabilizers only.
    feat = np.dstack([fL, 0.25 * fa, 0.25 * fb]).astype(np.float32)

    x1, y1, x2, y2 = mask_bbox(block_mask)
    pad = 4
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    feat_crop = feat[y1:y2, x1:x2].copy()
    mask_crop = inside[y1:y2, x1:x2]
    if mask_crop.sum() == 0:
        out_empty[inside] = 1
        return out_empty
    med = np.median(feat_crop[mask_crop], axis=0)
    feat_crop[~mask_crop] = med

    scale = params.felz_scale_small if area < params.large_area_threshold else params.felz_scale_large
    min_size = max(params.felz_min_size_floor, int(area / params.felz_min_size_area_divisor))
    sp_crop = felzenszwalb(feat_crop, scale=scale, sigma=params.felz_sigma, min_size=min_size).astype(np.int32) + 1
    sp = np.zeros((h, w), dtype=np.int32)
    sp[y1:y2, x1:x2] = sp_crop
    sp[~inside] = 0
    sp = relabel_sequential(sp)

    try:
        thresholds = threshold_multiotsu(vals, classes=nclass)
    except Exception:
        thresholds = np.array([np.median(vals)]) if nclass == 2 else np.percentile(vals, [33, 67])

    classmap = np.zeros((h, w), dtype=np.int32)
    for sid in [int(x) for x in np.unique(sp) if x > 0]:
        pix = (sp == sid) & inside
        vp = pix & (~bad)
        if vp.sum() < 3:
            vp = pix
        medv = float(np.median(L_smooth[vp]))
        cls = int(np.digitize(medv, thresholds) + 1)
        classmap[pix] = cls

    classmap = fill_zero_inside(classmap, inside)
    min_area = max(params.min_component_area_floor, int(params.min_component_area_fraction * area))
    return connected_cleanup(classmap, inside, min_area)


def make_gt_label_for_block(
    gt_masks: Sequence[np.ndarray],
    block_masks: Sequence[np.ndarray],
    block_index: int,
) -> Optional[np.ndarray]:
    """Assign each GT subpart to the large block with maximum overlap."""
    inside = block_masks[block_index]
    label = np.zeros_like(inside, dtype=np.int32)
    cur = 1
    for sm in gt_masks:
        overlaps = [int((sm & bm).sum()) for bm in block_masks]
        if not overlaps:
            continue
        assigned = int(np.argmax(overlaps))
        if assigned == block_index and overlaps[assigned] > 0:
            label[sm & inside] = cur
            cur += 1
    if label[inside].max() == 0:
        return None
    label = fill_zero_inside(label, inside)
    return relabel_sequential(label)


def covering_score(gt_label: np.ndarray, pred_label: np.ndarray, inside: np.ndarray, from_gt: bool = True) -> float:
    A = gt_label if from_gt else pred_label
    B = pred_label if from_gt else gt_label
    regs = [int(x) for x in np.unique(A[inside]) if x > 0]
    if not regs:
        return float("nan")
    total = 0
    acc = 0.0
    b_labs = [int(x) for x in np.unique(B[inside]) if x > 0]
    for r in regs:
        mask_a = (A == r) & inside
        a = int(mask_a.sum())
        total += a
        best = 0.0
        for s in b_labs:
            mask_b = (B == s) & inside
            inter = int((mask_a & mask_b).sum())
            if inter == 0:
                continue
            union = int((mask_a | mask_b).sum())
            best = max(best, inter / (union + 1e-9))
        acc += a * best
    return float(acc / (total + 1e-9))


def boundary_f1(gt_label: np.ndarray, pred_label: np.ndarray, inside: np.ndarray, tol: int = 2) -> Tuple[float, float, float]:
    gt_b = find_boundaries(gt_label, mode="outer") & inside
    pr_b = find_boundaries(pred_label, mode="outer") & inside
    if gt_b.sum() == 0 and pr_b.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt_b.sum() == 0 or pr_b.sum() == 0:
        return 0.0, 0.0, 0.0
    dt_g = distance_transform_edt(~gt_b)
    dt_p = distance_transform_edt(~pr_b)
    matched_p = pr_b & (dt_g <= tol)
    matched_g = gt_b & (dt_p <= tol)
    prec = float(matched_p.sum() / (pr_b.sum() + 1e-9))
    rec = float(matched_g.sum() / (gt_b.sum() + 1e-9))
    f1 = float(2 * prec * rec / (prec + rec + 1e-9))
    return prec, rec, f1


def evaluate_block(image_id: int, file_name: str, block_idx: int, gt_label: np.ndarray, pred_label: np.ndarray, inside: np.ndarray) -> BlockMetrics:
    y_true = gt_label[inside].ravel()
    y_pred = pred_label[inside].ravel()
    ari = float(adjusted_rand_score(y_true, y_pred))
    cg = covering_score(gt_label, pred_label, inside, True)
    cp = covering_score(gt_label, pred_label, inside, False)
    bp, br, bf = boundary_f1(gt_label, pred_label, inside, tol=2)
    return BlockMetrics(
        image_id=image_id,
        file_name=file_name,
        block_index=block_idx + 1,
        gt_regions=int(gt_label.max()),
        pred_regions=int(pred_label.max()),
        ari=ari,
        cover_gt=cg,
        cover_pred=cp,
        boundary_precision=bp,
        boundary_recall=br,
        boundary_f1=bf,
        area=int(inside.sum()),
    )


def overlay_label(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    for lab in [int(x) for x in np.unique(label) if x > 0]:
        color = COLORS[(lab - 1) % len(COLORS)].astype(np.float32)
        m = label == lab
        out[m] = out[m] * (1 - alpha) + color * alpha
    b = find_boundaries(label, mode="outer")
    out[b] = [255, 255, 0]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), width: int = 2) -> np.ndarray:
    out = img.copy()
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, color, width)
    return out


def add_title(img: np.ndarray, title: str, h: int = 30) -> np.ndarray:
    pil = Image.fromarray(img)
    out = Image.new("RGB", (pil.width, pil.height + h), "white")
    out.paste(pil, (0, h))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = None
    d.text((6, 7), title, fill="black", font=font)
    return np.array(out)


def resize_w(img: np.ndarray, width: int) -> np.ndarray:
    pil = Image.fromarray(img)
    h = max(1, int(round(pil.height * width / pil.width)))
    return np.array(pil.resize((width, h), Image.Resampling.LANCZOS))


def save_label_png(path: Path, label: np.ndarray) -> None:
    # Use uint16 to support many labels.
    Image.fromarray(label.astype(np.uint16)).save(path)


def make_panel(
    rgb: np.ndarray,
    block_mask: np.ndarray,
    pred_label: np.ndarray,
    gt_label: Optional[np.ndarray],
    metrics: Optional[BlockMetrics],
) -> np.ndarray:
    raw_view = draw_mask_contour(rgb, block_mask)
    pred_view = draw_mask_contour(overlay_label(rgb, pred_label), block_mask)
    if gt_label is not None:
        gt_view = draw_mask_contour(overlay_label(rgb, gt_label), block_mask)
        agree = rgb.copy()
        inside = block_mask > 0
        gb = find_boundaries(gt_label, mode="outer") & inside
        pb = find_boundaries(pred_label, mode="outer") & inside
        agree[gb] = [0, 255, 0]
        agree[pb] = [255, 0, 0]
        agree[gb & pb] = [255, 255, 0]
        title = f"ARI {metrics.ari:.3f} | CovGT {metrics.cover_gt:.3f} | BF1 {metrics.boundary_f1:.3f}"
        cols = [
            add_title(raw_view, "raw + big mask"),
            add_title(gt_view, f"GT subparts={int(gt_label.max())}"),
            add_title(pred_view, f"v3 pred={int(pred_label.max())}"),
            add_title(agree, title),
        ]
    else:
        cols = [add_title(raw_view, "raw + big mask"), add_title(pred_view, f"v3 pred={int(pred_label.max())}")]
    cols = [resize_w(c, 360) for c in cols]
    H = max(c.shape[0] for c in cols)
    padded = []
    for c in cols:
        p = np.ones((H, c.shape[1], 3), dtype=np.uint8) * 255
        p[: c.shape[0], : c.shape[1]] = c
        padded.append(p)
    return np.concatenate(padded, axis=1)


def build_contact_sheet(panels: Sequence[np.ndarray], out_path: Path, row_width: int) -> None:
    if not panels:
        return
    resized = [resize_w(p, row_width) for p in panels]
    W = max(p.shape[1] for p in resized)
    H = sum(p.shape[0] + 8 for p in resized)
    sheet = np.ones((H, W, 3), dtype=np.uint8) * 255
    y = 0
    for p in resized:
        sheet[y : y + p.shape[0], : p.shape[1]] = p
        y += p.shape[0] + 8
    Image.fromarray(sheet).save(out_path, quality=88)


def process_dataset(args: argparse.Namespace) -> None:
    params = V3Params()
    dataset_root, tmp = prepare_dataset(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_block_dir = args.out_dir / "per_block"
    per_image_dir = args.out_dir / "per_image_pred_labels"
    per_block_dir.mkdir(exist_ok=True)
    per_image_dir.mkdir(exist_ok=True)

    if args.keep_extracted and args.input_zip is not None:
        dst = args.out_dir / "extracted_dataset"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(dataset_root, dst)

    ann_path = find_annotation_file(dataset_root)
    _, images, anns_by_image, cat_name_to_id = load_coco(ann_path)
    big_cat_id = cat_name_to_id.get(args.big_category)
    subpart_cat_id = cat_name_to_id.get(args.subpart_category)
    if big_cat_id is None:
        raise ValueError(f"Could not find big category {args.big_category!r}. Available: {sorted(cat_name_to_id)}")

    metrics_rows: List[BlockMetrics] = []
    panels: List[np.ndarray] = []
    processed_images = 0
    processed_blocks = 0

    for image_id in sorted(images):
        im = images[image_id]
        file_name = im["file_name"]
        if args.raw_prefix and args.raw_prefix not in file_name:
            continue
        anns = anns_by_image.get(image_id, [])
        big_anns = anns_of_category(anns, big_cat_id)
        if not big_anns:
            continue
        img_path = image_path(dataset_root, ann_path, file_name)
        rgb = read_rgb(img_path)
        H, W = rgb.shape[:2]
        block_masks = [polygon_to_mask(H, W, a).astype(bool) for a in big_anns]
        gt_anns = anns_of_category(anns, subpart_cat_id)
        gt_masks = [polygon_to_mask(H, W, a).astype(bool) for a in gt_anns]

        full_label = np.zeros((H, W), dtype=np.int32)
        next_global = 1
        processed_images += 1
        for bidx, bm_bool in enumerate(block_masks):
            block_mask = bm_bool.astype(np.uint8) * 255
            pred = segment_block_v3(rgb, block_mask, params)
            # Global image-level labels.
            for lab in [int(x) for x in np.unique(pred) if x > 0]:
                full_label[pred == lab] = next_global
                next_global += 1

            gt_label = make_gt_label_for_block(gt_masks, block_masks, bidx) if gt_masks else None
            mrow = None
            if gt_label is not None:
                mrow = evaluate_block(image_id, file_name, bidx, gt_label, pred, bm_bool)
                metrics_rows.append(mrow)

            npz_payload = {
                "pred_label": pred.astype(np.int32),
                "block_mask": block_mask.astype(np.uint8),
                "image_id": np.array([image_id], dtype=np.int32),
                "block_index": np.array([bidx + 1], dtype=np.int32),
            }
            if gt_label is not None:
                npz_payload["gt_label"] = gt_label.astype(np.int32)
            np.savez_compressed(per_block_dir / f"image_{image_id}_block_{bidx+1}.npz", **npz_payload)

            panel = make_panel(rgb, block_mask, pred, gt_label, mrow)
            Image.fromarray(panel).save(per_block_dir / f"image_{image_id}_block_{bidx+1}.jpg", quality=95)
            panels.append(panel)
            processed_blocks += 1

        save_label_png(per_image_dir / f"image_{image_id}_pred_labels.png", full_label)
        Image.fromarray(overlay_label(rgb, full_label)).save(per_image_dir / f"image_{image_id}_pred_overlay.jpg", quality=95)

    build_contact_sheet(panels, args.out_dir / "v3_overview.jpg", args.overview_width)

    summary = {
        "method": "v3_graph_superpixel_lab_l",
        "processed_images": processed_images,
        "processed_blocks": processed_blocks,
        "params": asdict(params),
        "annotation_file": str(ann_path),
        "big_category": args.big_category,
        "subpart_category": args.subpart_category,
        "raw_prefix": args.raw_prefix,
    }

    if metrics_rows:
        df = pd.DataFrame([asdict(r) for r in metrics_rows])
        df.to_csv(args.out_dir / "v3_per_block.csv", index=False)
        img_df = df.groupby(["image_id", "file_name"], as_index=False).agg(
            blocks=("block_index", "count"),
            gt_regions=("gt_regions", "sum"),
            pred_regions=("pred_regions", "sum"),
            ari_mean=("ari", "mean"),
            cover_gt_mean=("cover_gt", "mean"),
            cover_pred_mean=("cover_pred", "mean"),
            boundary_f1_mean=("boundary_f1", "mean"),
        )
        img_df.to_csv(args.out_dir / "v3_per_image.csv", index=False)
        summary.update(
            {
                "metrics_available": True,
                "gt_regions_total": int(df["gt_regions"].sum()),
                "pred_regions_total": int(df["pred_regions"].sum()),
                "ari_mean": float(df["ari"].mean()),
                "ari_median": float(df["ari"].median()),
                "cover_gt_mean": float(df["cover_gt"].mean()),
                "cover_pred_mean": float(df["cover_pred"].mean()),
                "boundary_f1_mean": float(df["boundary_f1"].mean()),
                "count_exact_match_rate": float((df["gt_regions"] == df["pred_regions"]).mean()),
                "count_overseg_rate": float((df["pred_regions"] > df["gt_regions"]).mean()),
                "count_underseg_rate": float((df["pred_regions"] < df["gt_regions"]).mean()),
            }
        )
    else:
        summary["metrics_available"] = False

    (args.out_dir / "v3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if tmp is not None:
        tmp.cleanup()


def main() -> None:
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
