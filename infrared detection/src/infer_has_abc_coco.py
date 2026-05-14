#!/usr/bin/env python3
"""Run final subpart segmentation plus ABC/stacking detection and export COCO.

Input:
  - A dataset folder containing raw/infra images and `_annotations.coco.json`.
  - The COCO annotations contain the original RAW `gra` segmentations.

Output:
  - A new COCO JSON containing only original RAW `gra` segmentations whose
    inferred subparts have at least one accepted ABC/stacking shade candidate.
  - Output annotations keep the original segmentation geometry and are relabeled
    to category `has_ABC`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
PIPELINE_PATH = HERE / "selective_hybrid_full_pipeline_v4_smooth.py"
ABC_PATH = HERE / "experiment_subpart_binary_kmeans.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


pipe = load_module(PIPELINE_PATH, "infra_subpart_pipeline")
abc = load_module(ABC_PATH, "infra_abc_binary_kmeans")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder with raw/infra images and _annotations.coco.json.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output folder for has_ABC COCO and CSV reports.")
    ap.add_argument("--output-name", default="has_ABC_annotations.coco.json", help="Output COCO file name.")
    ap.add_argument("--work-dir", type=Path, default=None, help="Intermediate subpart output folder.")
    ap.add_argument("--reuse-segmentation", action="store_true", help="Reuse --work-dir if it already has pipeline outputs.")
    ap.add_argument("--pair-csv", type=Path, default=None, help="Optional CSV with raw_file,infra_file columns.")
    ap.add_argument("--pair-mode", choices=["filename", "order", "shape"], default="filename")
    ap.add_argument("--raw-prefix", default="raw_png")
    ap.add_argument("--infra-prefix", default="infra_png")
    ap.add_argument("--big-category", default="gra")
    ap.add_argument("--has-abc-category", default="has_ABC")
    ap.add_argument("--save-segmentation-overview", action="store_true")
    ap.add_argument(
        "--no-kmeans-visualizations",
        action="store_true",
        help="Do not save full-pair KMeans visualization images.",
    )

    # ABC detector defaults mirror experiment_subpart_binary_kmeans.py.
    ap.add_argument("--erode-px", type=int, default=2)
    ap.add_argument("--trim-low", type=float, default=5.0)
    ap.add_argument("--trim-high", type=float, default=95.0)
    ap.add_argument("--gaussian-sigma", type=float, default=0.5)
    ap.add_argument("--max-gray-delta", type=float, default=55.0)
    ap.add_argument("--min-gray-delta", type=float, default=0.0)
    ap.add_argument("--min-minority-frac", type=float, default=0.25)
    ap.add_argument("--min-effective-px", type=int, default=20)
    ap.add_argument(
        "--minority-frac-erode-px",
        type=int,
        default=2,
        help="Erode the final effective mask before computing minority class fraction for acceptance.",
    )
    ap.add_argument(
        "--boundary-prefilter-delta",
        type=float,
        default=80.0,
        help="Run raw k=2 KMeans before erode/trim. If raw median delta is above this, remove the minority class as boundary artifact.",
    )
    ap.add_argument("--disable-boundary-prefilter", action="store_true")

    # Attribution from accepted INFRA subpart back to original RAW gra annotation.
    ap.add_argument("--min-subpart-overlap-frac", type=float, default=0.20)
    ap.add_argument("--min-overlap-px", type=int, default=20)
    ap.add_argument(
        "--min-subpart-parent-frac",
        type=float,
        default=0.10,
        help="Skip ABC screening when a subpart is smaller than this fraction of its parent graphene area.",
    )
    return ap.parse_args()


def default_work_dir(output_coco: Path) -> Path:
    return output_coco.parent / f"{output_coco.stem}_subpart_work"


def run_subpart_pipeline(args: argparse.Namespace, work_dir: Path) -> None:
    decisions = work_dir / "pair_registration_decisions.csv"
    per_pair = work_dir / "per_pair"
    if args.reuse_segmentation and decisions.exists() and per_pair.exists():
        return

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PIPELINE_PATH),
        "--dataset-dir",
        str(args.input_dir),
        "--out-dir",
        str(work_dir),
        "--raw-prefix",
        args.raw_prefix,
        "--infra-prefix",
        args.infra_prefix,
        "--big-category",
        args.big_category,
    ]
    if args.pair_csv is not None:
        cmd += ["--pair-mode", "csv", "--pair-csv", str(args.pair_csv)]
    else:
        cmd += ["--pair-mode", args.pair_mode]
    if not args.save_segmentation_overview:
        cmd.append("--skip-overview")

    subprocess.run(cmd, check=True)


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_gray_smooth(gray: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)


def run_abc_for_subpart(
    gray: np.ndarray,
    gray_smooth: np.ndarray,
    final_label: np.ndarray,
    label_id: int,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[bool, str, Optional[dict], dict]:
    mask = final_label == label_id
    if args.disable_boundary_prefilter:
        boundary_info = {
            "boundary_prefilter_applied": 0,
            "boundary_prefilter_delta": np.nan,
            "boundary_prefilter_removed_frac": 0.0,
            "boundary_prefilter_removed_class": "",
            "boundary_prefilter_effective_px": int(mask.sum()),
        }
        inner = abc.erode_mask(mask, args.erode_px, args.min_effective_px)
    else:
        prefiltered_mask, boundary_info = abc.raw_boundary_prefilter(
            gray,
            mask,
            args.boundary_prefilter_delta,
            args.min_effective_px,
            seed=seed + 8009,
        )
        inner = (
            prefiltered_mask
            if int(boundary_info["boundary_prefilter_applied"]) == 1
            else abc.erode_mask(prefiltered_mask, args.erode_px, args.min_effective_px)
        )
    result = abc.run_binary_kmeans(
        gray,
        gray_smooth,
        inner,
        args.trim_low,
        args.trim_high,
        args.min_effective_px,
        seed=seed,
        minority_frac_erode_px=args.minority_frac_erode_px,
    )
    if result is not None:
        result.update(boundary_info)
    accepted, reason = abc.acceptance_reason(
        result,
        args.min_gray_delta,
        args.max_gray_delta,
        args.min_minority_frac,
    )
    return accepted, reason, result, boundary_info


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def pair_kmeans_visualization(
    gray: np.ndarray,
    final_label: np.ndarray,
    ann_masks: Dict[int, np.ndarray],
    higher_gray_map: np.ndarray,
) -> np.ndarray:
    """Draw full-frame registered segmentation, final subparts, and accepted high-gray KMeans pixels."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    higher_gray_map = higher_gray_map.astype(bool)
    if np.any(higher_gray_map):
        out[higher_gray_map] = (
            0.10 * out[higher_gray_map] + 0.90 * np.array([0, 255, 0], dtype=np.uint8)
        ).astype(np.uint8)

    # Yellow: original RAW graphene segmentation after registration to the INFRA frame.
    for ann_mask in ann_masks.values():
        abc.draw_contours(out, ann_mask, (0, 255, 255), 2)

    # Cyan: final subpart boundaries on the INFRA frame.
    for label_id in [int(x) for x in np.unique(final_label) if int(x) > 0]:
        subpart_mask = final_label == label_id
        abc.draw_contours(out, subpart_mask, (255, 255, 0), 1)
        ys, xs = np.where(subpart_mask)
        if xs.size:
            cv2.putText(
                out,
                f"L{label_id}",
                (int(np.median(xs)), int(np.median(ys))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return out


def source_gra_annotations(raw_im: dict, anns_by: Dict[int, List[dict]], big_id: int) -> List[dict]:
    return [a for a in anns_by.get(int(raw_im["id"]), []) if int(a.get("category_id", -1)) == int(big_id)]


def warped_annotation_masks(
    raw_im: dict,
    annotations: List[dict],
    matrix: np.ndarray,
    out_shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    masks = {}
    for ann in annotations:
        raw_mask = pipe.poly_mask(int(raw_im["height"]), int(raw_im["width"]), ann)
        masks[int(ann["id"])] = pipe.warp_mask(raw_mask, matrix, out_shape) > 0
    return masks


def assign_subpart_to_annotation(
    subpart_mask: np.ndarray,
    ann_masks: Dict[int, np.ndarray],
    min_subpart_overlap_frac: float,
    min_overlap_px: int,
) -> Tuple[Optional[int], int, float]:
    label_area = int(subpart_mask.sum())
    if label_area == 0 or not ann_masks:
        return None, 0, 0.0
    best_ann = None
    best_overlap = 0
    best_frac = 0.0
    for ann_id, ann_mask in ann_masks.items():
        overlap = int((subpart_mask & ann_mask).sum())
        frac = float(overlap / max(1, label_area))
        if overlap > best_overlap:
            best_ann = ann_id
            best_overlap = overlap
            best_frac = frac
    if best_ann is None:
        return None, best_overlap, best_frac
    if len(ann_masks) == 1 and best_overlap >= min(5, min_overlap_px):
        return best_ann, best_overlap, best_frac
    if best_overlap >= min_overlap_px and best_frac >= min_subpart_overlap_frac:
        return best_ann, best_overlap, best_frac
    return None, best_overlap, best_frac


def copy_selected_annotation(
    ann: dict,
    new_id: int,
    has_abc_category_id: int,
    details: List[dict],
    infra_file: str,
) -> dict:
    out = copy.deepcopy(ann)
    out["id"] = new_id
    out["category_id"] = has_abc_category_id
    out["source_annotation_id"] = ann.get("id")
    out["abc_infra_file"] = infra_file
    out["abc_subpart_labels"] = [int(d["subpart_label"]) for d in details]
    out["abc_subpart_count"] = len(details)
    out["abc_max_gray_delta"] = max(float(d["raw_gray_median_delta"]) for d in details)
    out["abc_max_minority_frac"] = max(float(d["minority_frac"]) for d in details)
    return out


def build_output_coco(
    input_data: dict,
    images_by_id: Dict[int, dict],
    selected_by_ann: Dict[int, dict],
    has_abc_category_name: str,
) -> dict:
    selected_image_ids = sorted({int(item["annotation"]["image_id"]) for item in selected_by_ann.values()})
    selected_images = [copy.deepcopy(images_by_id[i]) for i in selected_image_ids]

    has_abc_category_id = 1
    out_annotations = []
    for new_id, ann_id in enumerate(sorted(selected_by_ann), start=1):
        item = selected_by_ann[ann_id]
        out_annotations.append(
            copy_selected_annotation(
                item["annotation"],
                new_id,
                has_abc_category_id,
                item["details"],
                item["infra_file"],
            )
        )

    info = copy.deepcopy(input_data.get("info", {}))
    info["generated_by"] = "infer_has_abc_coco.py"
    info["description"] = "Original RAW gra segmentations relabeled as has_ABC when inferred subparts pass ABC/stacking detection."

    return {
        "info": info,
        "licenses": copy.deepcopy(input_data.get("licenses", [])),
        "images": selected_images,
        "annotations": out_annotations,
        "categories": [
            {
                "id": has_abc_category_id,
                "name": has_abc_category_name,
                "supercategory": "stacking",
            }
        ],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_coco = args.output_dir / args.output_name
    work_dir = args.work_dir if args.work_dir is not None else default_work_dir(output_coco)

    run_subpart_pipeline(args, work_dir)

    ann_path = pipe.find_ann(args.input_dir)
    input_data, images, anns_by, cats = pipe.load_coco(ann_path)
    big_id = cats.get(args.big_category)
    if big_id is None:
        raise ValueError(f"category not found: {args.big_category}")
    images_by_id = {int(k): v for k, v in images.items()}
    images_by_file = {im["file_name"]: im for im in images.values()}

    decisions = read_csv(work_dir / "pair_registration_decisions.csv")
    selected_by_ann: Dict[int, dict] = {}
    subpart_rows: List[dict] = []
    selected_rows: List[dict] = []
    kmeans_vis_dir = args.output_dir / "kmeans_visualizations"
    kmeans_visualization_paths: List[str] = []
    if not args.no_kmeans_visualizations:
        kmeans_vis_dir.mkdir(parents=True, exist_ok=True)

    for row in decisions:
        raw_file = row["raw_file"]
        infra_file = row["infra_file"]
        raw_id = int(row["raw_id"])
        infra_id = int(row["infra_id"])
        if raw_file not in images_by_file or infra_file not in images_by_file:
            continue
        raw_im = images_by_file[raw_file]
        gra_anns = source_gra_annotations(raw_im, anns_by, big_id)
        if not gra_anns:
            continue

        npz_path = work_dir / "per_pair" / f"raw{raw_id}_infra{infra_id}_masks.npz"
        z = np.load(npz_path)
        final_label = z["final_label"].astype(np.int32)
        matrix = z["M"].astype(np.float32)
        gray = abc.read_gray(pipe.image_path(args.input_dir, ann_path, infra_file))
        if final_label.shape != gray.shape:
            final_label = cv2.resize(final_label, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        gray_smooth = make_gray_smooth(gray, args.gaussian_sigma)
        ann_masks = warped_annotation_masks(raw_im, gra_anns, matrix, final_label.shape)
        anns_by_id = {int(a["id"]): a for a in gra_anns}
        higher_gray_map = np.zeros(gray.shape, dtype=bool)
        pair_subpart_rows: List[dict] = []

        for label_id in [int(x) for x in np.unique(final_label) if int(x) > 0]:
            subpart_mask = final_label == label_id
            assigned_ann_id, overlap_px, overlap_frac = assign_subpart_to_annotation(
                subpart_mask,
                ann_masks,
                args.min_subpart_overlap_frac,
                args.min_overlap_px,
            )
            subpart_area_px = int(subpart_mask.sum())
            parent_graphene_area_px = int(ann_masks[assigned_ann_id].sum()) if assigned_ann_id is not None else 0
            subpart_parent_area_frac = (
                float(subpart_area_px / max(1, parent_graphene_area_px))
                if parent_graphene_area_px > 0
                else 0.0
            )
            boundary_info = {
                "boundary_prefilter_applied": 0,
                "boundary_prefilter_delta": np.nan,
                "boundary_prefilter_removed_frac": 0.0,
                "boundary_prefilter_removed_class": "",
                "boundary_prefilter_effective_px": subpart_area_px,
            }
            if subpart_parent_area_frac < args.min_subpart_parent_frac:
                accepted = False
                reason = "subpart_area_below_graphene_fraction"
                result = None
            else:
                accepted, reason, result, boundary_info = run_abc_for_subpart(
                    gray,
                    gray_smooth,
                    final_label,
                    label_id,
                    args,
                    seed=1301 + raw_id * 97 + infra_id * 13 + label_id,
                )

            summary = {
                "raw_file": raw_file,
                "infra_file": infra_file,
                "raw_id": raw_id,
                "infra_id": infra_id,
                "raw_annotation_id": assigned_ann_id if assigned_ann_id is not None else "",
                "subpart_label": label_id,
                "subpart_area_px": subpart_area_px,
                "parent_graphene_area_px": parent_graphene_area_px,
                "subpart_parent_area_frac": subpart_parent_area_frac,
                "accepted": int(accepted),
                "reason": reason,
                "assignment_overlap_px": overlap_px,
                "assignment_overlap_frac": overlap_frac,
                "raw_gray_median_delta": np.nan if result is None else result["gray_median_delta"],
                "smooth_gray_median_delta": np.nan if result is None else result["smooth_median_delta"],
                "minority_class": "" if result is None else result["minority_class"],
                "minority_frac": np.nan if result is None else result["minority_frac"],
                "minority_class_before_area_erode": "" if result is None else result["minority_class_before_area_erode"],
                "minority_frac_before_area_erode": np.nan if result is None else result["minority_frac_before_area_erode"],
                "minority_area_effective_px": 0 if result is None else result["minority_area_effective_px"],
                "minority_frac_erode_px": args.minority_frac_erode_px,
                "higher_gray_class": "",
                "boundary_prefilter_applied": boundary_info["boundary_prefilter_applied"],
                "boundary_prefilter_delta": boundary_info["boundary_prefilter_delta"],
                "boundary_prefilter_removed_frac": boundary_info["boundary_prefilter_removed_frac"],
                "boundary_prefilter_removed_class": boundary_info["boundary_prefilter_removed_class"],
                "effective_px": 0 if result is None else result["effective_px"],
                "decision": row.get("decision", ""),
                "final_reason": row.get("final_reason", ""),
                "kmeans_visualization": "",
            }
            subpart_rows.append(summary)
            pair_subpart_rows.append(summary)

            if not args.no_kmeans_visualizations and result is not None:
                higher_gray_class = int(
                    max(result["class_rows"], key=lambda class_row: class_row["raw_median"])["class_id"]
                )
                if accepted:
                    higher_gray_map |= result["class_map"] == higher_gray_class
                summary["higher_gray_class"] = higher_gray_class

            if not accepted or assigned_ann_id is None:
                continue
            if assigned_ann_id not in selected_by_ann:
                selected_by_ann[assigned_ann_id] = {
                    "annotation": anns_by_id[assigned_ann_id],
                    "details": [],
                    "infra_file": infra_file,
                }
            selected_by_ann[assigned_ann_id]["details"].append(summary)

        if not args.no_kmeans_visualizations:
            pair = abc.infer_pair_name(raw_file)
            vis_name = safe_name(f"{pair}_raw{raw_id}_infra{infra_id}_registered_subparts_kmeans.jpg")
            vis_rel = str(Path("kmeans_visualizations") / vis_name)
            vis = pair_kmeans_visualization(gray, final_label, ann_masks, higher_gray_map)
            if not cv2.imwrite(str(kmeans_vis_dir / vis_name), vis):
                raise RuntimeError(f"failed to write KMeans visualization: {kmeans_vis_dir / vis_name}")
            kmeans_visualization_paths.append(vis_rel)
            for summary in pair_subpart_rows:
                summary["kmeans_visualization"] = vis_rel

    output_data = build_output_coco(input_data, images_by_id, selected_by_ann, args.has_abc_category)
    output_coco.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    for ann_id, item in sorted(selected_by_ann.items()):
        ann = item["annotation"]
        selected_rows.append(
            {
                "raw_file": images_by_id[int(ann["image_id"])]["file_name"],
                "raw_annotation_id": ann_id,
                "category": args.has_abc_category,
                "abc_subpart_count": len(item["details"]),
                "abc_subpart_labels": ",".join(str(int(d["subpart_label"])) for d in item["details"]),
                "max_gray_delta": max(float(d["raw_gray_median_delta"]) for d in item["details"]),
                "max_minority_frac": max(float(d["minority_frac"]) for d in item["details"]),
                "infra_file": item["infra_file"],
                "kmeans_visualizations": ";".join(
                    dict.fromkeys(d["kmeans_visualization"] for d in item["details"] if d.get("kmeans_visualization"))
                ),
            }
        )

    summary_path = output_coco.with_name(f"{output_coco.stem}_subpart_abc_summary.csv")
    selected_path = output_coco.with_name(f"{output_coco.stem}_selected_annotations.csv")
    config_path = output_coco.with_name(f"{output_coco.stem}_run_config.json")
    write_csv(summary_path, subpart_rows)
    write_csv(selected_path, selected_rows)
    config_path.write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "input_annotation": str(ann_path),
                "segmentation_work_dir": str(work_dir),
                "output_coco": str(output_coco),
                "big_category": args.big_category,
                "has_abc_category": args.has_abc_category,
                "abc_params": {
                    "erode_px": args.erode_px,
                    "trim_low": args.trim_low,
                    "trim_high": args.trim_high,
                    "gaussian_sigma": args.gaussian_sigma,
                    "min_gray_delta": args.min_gray_delta,
                    "max_gray_delta": args.max_gray_delta,
                    "min_minority_frac": args.min_minority_frac,
                    "min_effective_px": args.min_effective_px,
                    "minority_frac_erode_px": args.minority_frac_erode_px,
                    "boundary_prefilter_delta": args.boundary_prefilter_delta,
                    "disable_boundary_prefilter": args.disable_boundary_prefilter,
                },
                "assignment_params": {
                    "min_subpart_overlap_frac": args.min_subpart_overlap_frac,
                    "min_overlap_px": args.min_overlap_px,
                    "min_subpart_parent_frac": args.min_subpart_parent_frac,
                },
                "counts": {
                    "subparts_checked": len(subpart_rows),
                    "selected_annotations": len(selected_rows),
                    "output_images": len(output_data["images"]),
                    "kmeans_visualizations": len(kmeans_visualization_paths),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(output_coco)
    print(
        f"subparts_checked={len(subpart_rows)} selected_annotations={len(selected_rows)} "
        f"output_images={len(output_data['images'])}"
    )


if __name__ == "__main__":
    main()
