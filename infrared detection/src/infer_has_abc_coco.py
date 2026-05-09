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

    # ABC detector defaults mirror experiment_subpart_binary_kmeans.py.
    ap.add_argument("--erode-px", type=int, default=2)
    ap.add_argument("--trim-low", type=float, default=5.0)
    ap.add_argument("--trim-high", type=float, default=95.0)
    ap.add_argument("--gaussian-sigma", type=float, default=0.5)
    ap.add_argument("--max-gray-delta", type=float, default=55.0)
    ap.add_argument("--min-gray-delta", type=float, default=0.0)
    ap.add_argument("--min-minority-frac", type=float, default=0.25)
    ap.add_argument("--min-effective-px", type=int, default=20)

    # Attribution from accepted INFRA subpart back to original RAW gra annotation.
    ap.add_argument("--min-subpart-overlap-frac", type=float, default=0.20)
    ap.add_argument("--min-overlap-px", type=int, default=20)
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
) -> Tuple[bool, str, Optional[dict]]:
    mask = final_label == label_id
    inner = abc.erode_mask(mask, args.erode_px, args.min_effective_px)
    result = abc.run_binary_kmeans(
        gray,
        gray_smooth,
        inner,
        args.trim_low,
        args.trim_high,
        args.min_effective_px,
        seed=seed,
    )
    accepted, reason = abc.acceptance_reason(
        result,
        args.min_gray_delta,
        args.max_gray_delta,
        args.min_minority_frac,
    )
    return accepted, reason, result


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

        for label_id in [int(x) for x in np.unique(final_label) if int(x) > 0]:
            accepted, reason, result = run_abc_for_subpart(
                gray,
                gray_smooth,
                final_label,
                label_id,
                args,
                seed=1301 + raw_id * 97 + infra_id * 13 + label_id,
            )
            subpart_mask = final_label == label_id
            assigned_ann_id, overlap_px, overlap_frac = assign_subpart_to_annotation(
                subpart_mask,
                ann_masks,
                args.min_subpart_overlap_frac,
                args.min_overlap_px,
            )

            summary = {
                "raw_file": raw_file,
                "infra_file": infra_file,
                "raw_id": raw_id,
                "infra_id": infra_id,
                "raw_annotation_id": assigned_ann_id if assigned_ann_id is not None else "",
                "subpart_label": label_id,
                "accepted": int(accepted),
                "reason": reason,
                "assignment_overlap_px": overlap_px,
                "assignment_overlap_frac": overlap_frac,
                "raw_gray_median_delta": np.nan if result is None else result["gray_median_delta"],
                "smooth_gray_median_delta": np.nan if result is None else result["smooth_median_delta"],
                "minority_class": "" if result is None else result["minority_class"],
                "minority_frac": np.nan if result is None else result["minority_frac"],
                "effective_px": 0 if result is None else result["effective_px"],
                "decision": row.get("decision", ""),
                "final_reason": row.get("final_reason", ""),
            }
            subpart_rows.append(summary)

            if not accepted or assigned_ann_id is None:
                continue
            if assigned_ann_id not in selected_by_ann:
                selected_by_ann[assigned_ann_id] = {
                    "annotation": anns_by_id[assigned_ann_id],
                    "details": [],
                    "infra_file": infra_file,
                }
            selected_by_ann[assigned_ann_id]["details"].append(summary)

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
                },
                "assignment_params": {
                    "min_subpart_overlap_frac": args.min_subpart_overlap_frac,
                    "min_overlap_px": args.min_overlap_px,
                },
                "counts": {
                    "subparts_checked": len(subpart_rows),
                    "selected_annotations": len(selected_rows),
                    "output_images": len(output_data["images"]),
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
