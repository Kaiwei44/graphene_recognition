#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


COARSE_CATEGORIES = [
    {"id": 1, "name": "low_1_5", "supercategory": "graphene_thickness"},
    {"id": 2, "name": "mid_6_8", "supercategory": "graphene_thickness"},
    {"id": 3, "name": "thick_9_plus", "supercategory": "graphene_thickness"},
]


def parse_layer_from_category(name: Any) -> int | None:
    text = str(name).lower()
    nums = re.findall(r"(?<!\d)(\d+)(?!\d)", text)
    if nums:
        return int(nums[0])
    try:
        return int(float(text))
    except Exception:
        return None


def layer_to_coarse_id(layer: int, low_max: int, mid_max: int) -> int | None:
    if 1 <= layer <= low_max:
        return 1
    if low_max < layer <= mid_max:
        return 2
    if layer > mid_max:
        return 3
    return None


def output_path_for(annotation_path: Path, out_suffix: str) -> Path:
    if annotation_path.name == "_annotations.coco.json":
        return annotation_path.with_name("_annotations" + out_suffix)
    return annotation_path.with_name(annotation_path.stem + out_suffix)


def remap_annotation_file(
    annotation_path: Path,
    output_path: Path,
    low_max: int,
    mid_max: int,
    keep_empty_images: bool,
) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_to_layer: dict[int, int] = {}
    for category in coco.get("categories", []):
        layer = parse_layer_from_category(category.get("name", ""))
        if layer is not None:
            category_to_layer[int(category["id"])] = layer

    kept_annotations = []
    kept_image_ids = set()
    layer_counts: Counter[int] = Counter()
    coarse_counts: Counter[int] = Counter()
    dropped_counts: Counter[str] = Counter()

    for annotation in coco.get("annotations", []):
        original_category_id = int(annotation.get("category_id", -1))
        layer = category_to_layer.get(original_category_id)
        if layer is None:
            dropped_counts[f"unknown_category_{original_category_id}"] += 1
            continue

        coarse_id = layer_to_coarse_id(layer, low_max=low_max, mid_max=mid_max)
        if coarse_id is None:
            dropped_counts[f"unmapped_layer_{layer}"] += 1
            continue

        new_annotation = dict(annotation)
        new_annotation["category_id"] = coarse_id
        kept_annotations.append(new_annotation)
        kept_image_ids.add(int(new_annotation["image_id"]))
        layer_counts[layer] += 1
        coarse_counts[coarse_id] += 1

    output_coco = dict(coco)
    output_coco["categories"] = COARSE_CATEGORIES
    output_coco["annotations"] = kept_annotations
    if not keep_empty_images:
        output_coco["images"] = [
            image for image in coco.get("images", []) if int(image["id"]) in kept_image_ids
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_coco, f, ensure_ascii=False)

    summary = {
        "input": str(annotation_path),
        "output": str(output_path),
        "num_images": len(output_coco.get("images", [])),
        "num_annotations": len(kept_annotations),
        "layer_counts": dict(sorted(layer_counts.items())),
        "coarse_counts": {
            category["name"]: int(coarse_counts[category["id"]])
            for category in COARSE_CATEGORIES
        },
        "dropped_counts": dict(sorted(dropped_counts.items())),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def find_annotation_paths(dataset_dir: Path) -> list[Path]:
    paths = []
    for split in ("train", "valid", "val", "test"):
        path = dataset_dir / split / "_annotations.coco.json"
        if path.exists():
            paths.append(path)
    if paths:
        return paths
    return sorted(dataset_dir.rglob("_annotations.coco.json"))


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\n[OK] {summary['input']} -> {summary['output']}")
    print(f"  images: {summary['num_images']}")
    print(f"  annotations: {summary['num_annotations']}")
    print(f"  layer_counts: {summary['layer_counts']}")
    print(f"  coarse_counts: {summary['coarse_counts']}")
    if summary["dropped_counts"]:
        print(f"  dropped_counts: {summary['dropped_counts']}")
    if 5 not in summary["layer_counts"]:
        print("  warning: no layer-5 annotations in this split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert layer COCO annotations to AMM coarse thickness labels."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Roboflow COCO dataset root containing train/test folders.",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=None,
        help="Convert one COCO json file instead of scanning a dataset directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output json path. Only valid with --annotation-path.",
    )
    parser.add_argument(
        "--out-suffix",
        default=".amm_coarse.coco.json",
        help="Suffix used when converting a full dataset directory.",
    )
    parser.add_argument("--low-max", type=int, default=5)
    parser.add_argument("--mid-max", type=int, default=8)
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Drop images that have no remapped foreground annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.annotation_path is None and args.dataset_dir is None:
        raise SystemExit("Provide either --dataset-dir or --annotation-path")
    if args.annotation_path is not None and args.output_path is None:
        raise SystemExit("--output-path is required when using --annotation-path")
    if args.low_max >= args.mid_max:
        raise SystemExit("--low-max must be smaller than --mid-max")

    if args.annotation_path is not None:
        annotation_paths = [args.annotation_path]
    else:
        annotation_paths = find_annotation_paths(args.dataset_dir)
        if not annotation_paths:
            raise SystemExit(f"No _annotations.coco.json found under {args.dataset_dir}")

    for annotation_path in annotation_paths:
        output_path = args.output_path or output_path_for(annotation_path, args.out_suffix)
        summary = remap_annotation_file(
            annotation_path=annotation_path,
            output_path=output_path,
            low_max=args.low_max,
            mid_max=args.mid_max,
            keep_empty_images=not args.drop_empty_images,
        )
        print_summary(summary)


if __name__ == "__main__":
    main()
