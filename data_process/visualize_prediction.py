import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None

from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models import M2F_model
from maskterial.modeling.segmentation_models.M2F import maskformer_model  # noqa: F401
from maskterial.utils.dataset_functions import setup_config

from flake_postprocess import GrapheneFlakePostprocessor, PostprocessParams


def build_cfg(config_file: str, weights: str, extra_opts: list[str]):
    args = SimpleNamespace(
        config_file=config_file,
        opts=["MODEL.WEIGHTS", weights, *extra_opts],
        resume=False,
        eval_only=True,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )
    return setup_config(args)


def clamp_detection_count(cfg):
    max_scores = (
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES * cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    )
    if cfg.TEST.DETECTIONS_PER_IMAGE <= max_scores:
        return

    cfg.defrost()
    cfg.TEST.DETECTIONS_PER_IMAGE = max_scores
    cfg.freeze()
    print(f"Clamped TEST.DETECTIONS_PER_IMAGE to {max_scores} available queries")


def flake_score(flake) -> float:
    return 1.0 - float(flake.false_positive_probability)


def flake_area_um2(flake) -> float:
    if flake.measurements is None:
        return 0.0
    return float(flake.measurements.area_um2)


def flake_shape_complexity(flake) -> float:
    mask = flake.mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter_px = sum(cv2.arcLength(contour, True) for contour in contours)
    area_px = max(int(np.count_nonzero(mask)), 1)
    return float((perimeter_px * perimeter_px) / (4.0 * np.pi * area_px))


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (0, 128, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 0, 255),
        (128, 255, 0),
        (255, 0, 128),
        (128, 0, 255),
    ]
    return palette[(index - 1) % len(palette)]


def format_flake_label(index: int, flake, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    if label_mode == "index":
        return f"#{index}"
    if label_mode == "score":
        return f"#{index} s={flake_score(flake):.2f}"
    if label_mode == "area":
        return f"#{index} a={flake_area_um2(flake):.1f}"
    return (
        f"#{index} s={flake_score(flake):.2f} "
        f"a={flake_area_um2(flake):.1f} c={flake_shape_complexity(flake):.1f}"
    )


def draw_text_box(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]):
    if not text:
        return

    image_h, image_w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = min(max(origin[0], 0), max(0, image_w - text_w - 8))
    y = min(max(origin[1], text_h + 8), max(text_h + 8, image_h - baseline - 4))
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_flake_labels(image, flakes, scale: float, label_mode: str):
    image = np.ascontiguousarray(image)
    for index, flake in enumerate(flakes, start=1):
        text = format_flake_label(index, flake, label_mode)
        center_x = int(flake.center[0] * scale)
        center_y = int(flake.center[1] * scale)
        draw_text_box(
            image,
            text,
            (center_x + 8, center_y - 8),
            color_for_index(index),
        )
    return image


def filter_flakes_by_area(flakes, min_area_um2: float):
    return [flake for flake in flakes if flake.measurements.area_um2 >= min_area_um2]


def sort_flakes(flakes, sort_by: str):
    if sort_by == "score":
        return sorted(flakes, key=flake_score, reverse=True)
    if sort_by == "area":
        return sorted(flakes, key=flake_area_um2, reverse=True)
    return flakes


def draw_flake_masks(image, flakes, scale: float):
    output = image.copy()
    overlay = image.copy()
    for index, flake in enumerate(flakes, start=1):
        mask = flake.mask.astype(np.uint8)
        if scale != 1.0:
            mask = cv2.resize(
                mask,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        color = color_for_index(index)
        overlay[mask.astype(bool)] = color
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)
    return output


def annotation_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = annotation.get("bbox", [0, 0, 0, 0])
        x0 = int(max(0, np.floor(x)))
        y0 = int(max(0, np.floor(y)))
        x1 = int(min(width, np.ceil(x + w)))
        y1 = int(min(height, np.ceil(y + h)))
        mask[y0:y1, x0:x1] = 1
        return mask.astype(bool)

    if mask_utils is None:
        raise ImportError("pycocotools is required to evaluate segmentation masks")

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        decoded = mask_utils.decode(rle)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def gt_masks_from_dataset_dict(dataset_dict: dict, height: int, width: int, min_area_px: int):
    gt_masks = []
    for annotation in dataset_dict.get("annotations", []):
        if int(annotation.get("iscrowd", 0)):
            continue
        mask = annotation_to_mask(annotation, height, width)
        if int(np.count_nonzero(mask)) < min_area_px:
            continue
        gt_masks.append(mask.astype(bool))
    return gt_masks


def pred_masks_from_flakes(flakes):
    return [(np.asarray(flake.mask) > 0).astype(bool) for flake in flakes]


def pairwise_intersection_iou(gt_masks, pred_masks):
    if not gt_masks or not pred_masks:
        return (
            np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32),
            np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32),
            np.asarray([int(np.count_nonzero(mask)) for mask in gt_masks], dtype=np.float32),
            np.asarray([int(np.count_nonzero(mask)) for mask in pred_masks], dtype=np.float32),
        )

    gt_areas = np.asarray([int(np.count_nonzero(mask)) for mask in gt_masks], dtype=np.float32)
    pred_areas = np.asarray([int(np.count_nonzero(mask)) for mask in pred_masks], dtype=np.float32)
    intersections = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            intersections[gt_index, pred_index] = int(np.count_nonzero(gt_mask & pred_mask))
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    ious = np.divide(
        intersections,
        np.maximum(unions, 1.0),
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    return intersections, ious, gt_areas, pred_areas


def gt_union_coverages(gt_masks, pred_masks, gt_areas):
    if not gt_masks:
        return np.zeros(0, dtype=np.float32)
    if not pred_masks:
        return np.zeros(len(gt_masks), dtype=np.float32)

    pred_union = np.zeros_like(gt_masks[0], dtype=bool)
    for pred_mask in pred_masks:
        pred_union |= pred_mask

    coverages = np.zeros(len(gt_masks), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        coverages[gt_index] = int(np.count_nonzero(gt_mask & pred_union)) / max(
            float(gt_areas[gt_index]),
            1.0,
        )
    return coverages


def match_instances_by_iou(ious: np.ndarray, iou_threshold: float) -> tuple[set[int], set[int]]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if ious.size == 0:
        return matched_gt, matched_pred

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-ious)
        pairs = zip(row_ind, col_ind)
    else:
        candidate_indices = np.argwhere(ious >= iou_threshold)
        order = np.argsort(-ious[candidate_indices[:, 0], candidate_indices[:, 1]])
        pairs = ((candidate_indices[i, 0], candidate_indices[i, 1]) for i in order)

    for gt_index, pred_index in pairs:
        gt_index = int(gt_index)
        pred_index = int(pred_index)
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        if ious[gt_index, pred_index] < iou_threshold:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
    return matched_gt, matched_pred


def evaluate_flake_predictions(
    dataset_dict: dict,
    flakes,
    image_shape: tuple[int, int],
    args,
) -> dict:
    height, width = image_shape
    gt_masks = gt_masks_from_dataset_dict(
        dataset_dict,
        height=height,
        width=width,
        min_area_px=args.eval_min_gt_area_px,
    )
    pred_masks = pred_masks_from_flakes(flakes)
    intersections, ious, gt_areas, pred_areas = pairwise_intersection_iou(gt_masks, pred_masks)

    matched_gt, matched_pred = match_instances_by_iou(ious, args.eval_iou_threshold)

    tp = len(matched_gt)
    fp = len(pred_masks) - len(matched_pred)
    fn = len(gt_masks) - len(matched_gt)

    if len(gt_masks) and len(pred_masks):
        hit_coverages = gt_union_coverages(gt_masks, pred_masks, gt_areas)
        hit_flags = hit_coverages >= args.eval_hit_coverage_threshold
        significant = intersections / np.maximum(
            np.minimum(gt_areas[:, None], pred_areas[None, :]),
            1.0,
        )
        fragment_counts = (significant >= args.eval_significant_overlap_threshold).sum(axis=1)
        gt_per_pred = (significant >= args.eval_significant_overlap_threshold).sum(axis=0)
    else:
        hit_coverages = np.zeros(len(gt_masks), dtype=np.float32)
        hit_flags = np.zeros(len(gt_masks), dtype=bool)
        fragment_counts = np.zeros(len(gt_masks), dtype=np.int64)
        gt_per_pred = np.zeros(len(pred_masks), dtype=np.int64)

    hit_count = int(np.count_nonzero(hit_flags))
    split_gt_count = int(np.count_nonzero((fragment_counts > 1) & hit_flags))
    fragment_sum_hit = int(fragment_counts[hit_flags].sum()) if hit_count else 0
    max_fragments = int(fragment_counts.max()) if len(fragment_counts) else 0
    merge_pred_count = int(np.count_nonzero(gt_per_pred > 1))
    max_gt_per_pred = int(gt_per_pred.max()) if len(gt_per_pred) else 0

    image_name = os.path.basename(dataset_dict["file_name"])
    return {
        "image": image_name,
        "gt": int(len(gt_masks)),
        "pred": int(len(pred_masks)),
        "strict_tp": int(tp),
        "strict_fp": int(fp),
        "strict_fn": int(fn),
        "hit_count": hit_count,
        "split_gt_count": split_gt_count,
        "fragment_sum_hit": fragment_sum_hit,
        "max_fragments_per_gt": max_fragments,
        "merge_pred_count": merge_pred_count,
        "max_gt_per_pred": max_gt_per_pred,
        "mean_hit_coverage": float(np.mean(hit_coverages)) if len(hit_coverages) else 0.0,
    }


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_eval_rows(rows: list[dict], args) -> dict:
    totals = {
        "images": len(rows),
        "gt": sum(int(row["gt"]) for row in rows),
        "pred": sum(int(row["pred"]) for row in rows),
        "strict_tp": sum(int(row["strict_tp"]) for row in rows),
        "strict_fp": sum(int(row["strict_fp"]) for row in rows),
        "strict_fn": sum(int(row["strict_fn"]) for row in rows),
        "hit_count": sum(int(row["hit_count"]) for row in rows),
        "split_gt_count": sum(int(row["split_gt_count"]) for row in rows),
        "fragment_sum_hit": sum(int(row["fragment_sum_hit"]) for row in rows),
        "merge_pred_count": sum(int(row["merge_pred_count"]) for row in rows),
        "max_fragments_per_gt": max([int(row["max_fragments_per_gt"]) for row in rows], default=0),
        "max_gt_per_pred": max([int(row["max_gt_per_pred"]) for row in rows], default=0),
    }
    precision = ratio(totals["strict_tp"], totals["strict_tp"] + totals["strict_fp"])
    recall = ratio(totals["strict_tp"], totals["strict_tp"] + totals["strict_fn"])
    f1 = ratio(2 * precision * recall, precision + recall)
    hit_recall = ratio(totals["hit_count"], totals["gt"])
    return {
        "eval_iou_threshold": float(args.eval_iou_threshold),
        "eval_hit_coverage_threshold": float(args.eval_hit_coverage_threshold),
        "eval_significant_overlap_threshold": float(args.eval_significant_overlap_threshold),
        "eval_min_gt_area_px": int(args.eval_min_gt_area_px),
        **totals,
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
        "hit_recall": hit_recall,
        "fp_per_image": ratio(totals["strict_fp"], totals["images"]),
        "split_rate_all_gt": ratio(totals["split_gt_count"], totals["gt"]),
        "split_rate_hit_gt": ratio(totals["split_gt_count"], totals["hit_count"]),
        "avg_fragments_per_hit": ratio(totals["fragment_sum_hit"], totals["hit_count"]),
        "merge_rate_pred": ratio(totals["merge_pred_count"], totals["pred"]),
    }


def write_eval_outputs(rows: list[dict], outdir: str, args):
    if not rows:
        return
    per_image_path = os.path.join(outdir, "evaluation_per_image.csv")
    with open(per_image_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = summarize_eval_rows(rows, args)
    metrics_path = os.path.join(outdir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"Evaluation metrics -> {metrics_path}")
    print(f"Per-image evaluation -> {per_image_path}")
    print(
        "Strict F1={strict_f1:.4f}, Hit Recall={hit_recall:.4f}, "
        "FP/image={fp_per_image:.4f}, Split Rate={split_rate_all_gt:.4f}".format(**metrics)
    )


def write_flake_rows(writer, dataset_dict, flakes):
    image_name = os.path.basename(dataset_dict["file_name"])
    for index, flake in enumerate(flakes, start=1):
        writer.writerow(
            {
                "image": image_name,
                "instance_id": index,
                "score": f"{flake_score(flake):.6f}",
                "area_um2": f"{flake_area_um2(flake):.6f}",
                "area_px": int(flake.measurements.area_px),
                "center_x": int(flake.center[0]),
                "center_y": int(flake.center[1]),
                "max_sidelength_px": f"{float(flake.max_sidelength):.3f}",
                "min_sidelength_px": f"{float(flake.min_sidelength):.3f}",
                "shape_complexity": f"{flake_shape_complexity(flake):.6f}",
            }
        )


def build_postprocess_params(args) -> PostprocessParams:
    return PostprocessParams(
        overlap_iou_threshold=args.pp_overlap_iou_threshold,
        overlap_containment_threshold=args.pp_overlap_containment_threshold,
        enable_bridge_merge=args.pp_enable_bridge_merge,
        grow_radius_px=args.pp_grow_radius_px,
        max_boundary_distance_px=args.pp_max_boundary_distance_px,
        lab_l_weight=args.pp_lab_l_weight,
        tau_grow=args.pp_tau_grow,
        tau_pair=args.pp_tau_pair,
        grow_sigma=args.pp_grow_sigma,
        min_bridge_area_px=args.pp_min_bridge_area_px,
        min_bridge_ratio=args.pp_min_bridge_ratio,
        final_min_area_um2=args.pp_final_min_area_um2,
        final_min_score=args.pp_final_min_score,
        final_max_shape_complexity=args.pp_final_max_shape_complexity,
        max_bridge_passes=args.pp_max_bridge_passes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--outdir", default="./vis_pred")
    parser.add_argument("--dataset-name", default="visualize_prediction_dataset")
    parser.add_argument("--class-name", default="gra")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--size-threshold-px", type=int, default=0)
    parser.add_argument("--min-area-um2", type=float, default=0.0)
    parser.add_argument(
        "--label-mode",
        choices=["full", "area", "score", "index", "none"],
        default="full",
    )
    parser.add_argument("--sort-by", choices=["score", "area", "none"], default="score")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Deprecated compatibility flag. Postprocessing is enabled by default.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable graphene flake postprocessing and visualize raw model predictions.",
    )
    parser.add_argument("--postprocess-vis-dir", default=None)
    parser.add_argument("--pp-overlap-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pp-overlap-containment-threshold", type=float, default=0.8)
    parser.add_argument("--pp-enable-bridge-merge", action="store_true")
    parser.add_argument("--pp-grow-radius-px", type=int, default=3)
    parser.add_argument("--pp-max-boundary-distance-px", type=int, default=3)
    parser.add_argument("--pp-lab-l-weight", type=float, default=0.5)
    parser.add_argument("--pp-tau-grow", type=float, default=12.0)
    parser.add_argument("--pp-tau-pair", type=float, default=12.0)
    parser.add_argument("--pp-grow-sigma", type=float, default=1.5)
    parser.add_argument("--pp-min-bridge-area-px", type=float, default=15.0)
    parser.add_argument("--pp-min-bridge-ratio", type=float, default=0.3)
    parser.add_argument("--pp-final-min-area-um2", type=float, default=100.0)
    parser.add_argument("--pp-final-min-score", type=float, default=0.015)
    parser.add_argument("--pp-final-max-shape-complexity", type=float, default=5.0)
    parser.add_argument("--pp-max-bridge-passes", type=int, default=5)
    parser.add_argument(
        "--no-eval-metrics",
        action="store_true",
        help="Disable segmentation evaluation CSV/JSON generation.",
    )
    parser.add_argument(
        "--eval-iou-threshold",
        type=float,
        default=0.5,
        help="One-to-one IoU threshold for strict TP/FP/FN metrics.",
    )
    parser.add_argument(
        "--eval-hit-coverage-threshold",
        type=float,
        default=0.5,
        help="GT coverage threshold for hit recall, using all predicted masks together.",
    )
    parser.add_argument(
        "--eval-significant-overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap ratio used to count split and merge fragments.",
    )
    parser.add_argument(
        "--eval-min-gt-area-px",
        type=int,
        default=0,
        help="Ignore GT masks smaller than this pixel area when computing metrics.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.postprocess_vis_dir is not None:
        os.makedirs(args.postprocess_vis_dir, exist_ok=True)
    random.seed(args.seed)

    cfg = build_cfg(args.config_file, args.weights, args.opts)
    clamp_detection_count(cfg)

    register_coco_instances(args.dataset_name, {}, args.ann, args.image_root)
    MetadataCatalog.get(args.dataset_name).set(thing_classes=[args.class_name])
    meta = MetadataCatalog.get(args.dataset_name)
    predictor = DefaultPredictor(cfg)
    seg_model = M2F_model(
        model=predictor.model,
        config=cfg,
        device=torch.device(cfg.MODEL.DEVICE),
    )
    maskterial = MaskTerial(
        segmentation_model=seg_model,
        score_threshold=args.score_threshold,
        min_class_occupancy=0.0,
        size_threshold=args.size_threshold_px,
        device=torch.device(cfg.MODEL.DEVICE),
    )
    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    if args.all_images or args.num_samples == -1:
        samples = dataset_dicts
    else:
        samples = random.sample(dataset_dicts, min(args.num_samples, len(dataset_dicts)))

    use_postprocess = not args.no_postprocess
    if args.no_postprocess and args.postprocess_vis_dir is not None:
        print("[WARN] --postprocess-vis-dir is ignored because --no-postprocess was set.")
    postprocessor = None
    if use_postprocess:
        postprocessor = GrapheneFlakePostprocessor(build_postprocess_params(args))

    csv_path = args.summary_csv or os.path.join(args.outdir, "predictions.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "image",
            "instance_id",
            "score",
            "area_um2",
            "area_px",
            "center_x",
            "center_y",
            "max_sidelength_px",
            "min_sidelength_px",
            "shape_complexity",
        ],
    )
    csv_writer.writeheader()
    eval_rows = []

    for dataset_dict in samples:
        img = cv2.imread(dataset_dict["file_name"])
        if img is None:
            print(f"Skipping unreadable image: {dataset_dict['file_name']}")
            continue

        raw_flakes = maskterial.predict(img)
        base = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
        if use_postprocess:
            postprocess_result = postprocessor.run(
                image_bgr=img,
                raw_flakes=raw_flakes,
                debug_vis_dir=args.postprocess_vis_dir,
                image_stem=base,
            )
            flakes = postprocess_result.final_flakes
        else:
            flakes = filter_flakes_by_area(raw_flakes, args.min_area_um2)

        flakes = sort_flakes(flakes, args.sort_by)
        write_flake_rows(csv_writer, dataset_dict, flakes)
        if not args.no_eval_metrics:
            eval_rows.append(
                evaluate_flake_predictions(
                    dataset_dict=dataset_dict,
                    flakes=flakes,
                    image_shape=img.shape[:2],
                    args=args,
                )
            )

        pred_img = img
        if args.scale != 1.0:
            pred_img = cv2.resize(
                pred_img,
                dsize=None,
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_LINEAR,
            )
        pred_img = draw_flake_masks(pred_img, flakes, args.scale)
        if use_postprocess:
            pred_img = draw_flake_labels(pred_img, flakes, args.scale, args.label_mode)

        if use_postprocess:
            print(
                f"{base}: raw {len(raw_flakes)}, final {len(flakes)} "
                f"(postprocess score>={args.pp_final_min_score}, "
                f"area>={args.pp_final_min_area_um2} um^2, "
                f"shape<={args.pp_final_max_shape_complexity})"
            )
        else:
            print(
                f"{base}: predicted {len(raw_flakes)}, drew {len(flakes)} "
                f"(score>={args.score_threshold}, size>{args.size_threshold_px}px, "
                f"area>={args.min_area_um2} um^2)"
            )
        pred_path = os.path.join(args.outdir, f"{base}_pred.jpg")
        cv2.imwrite(pred_path, pred_img)

        if args.draw_gt:
            gt_visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=meta,
                scale=args.scale,
                instance_mode=ColorMode.IMAGE,
            )
            gt_img = gt_visualizer.draw_dataset_dict(dataset_dict).get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(args.outdir, f"{base}_gt.jpg"), gt_img)

        print(f"saved: {pred_path}")

    csv_file.close()
    if not args.no_eval_metrics:
        write_eval_outputs(eval_rows, args.outdir, args)
    print(f"All done. -> {args.outdir}")
    print(f"Prediction table -> {csv_path}")


if __name__ == "__main__":
    main()
