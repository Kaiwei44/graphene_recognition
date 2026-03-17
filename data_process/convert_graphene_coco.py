#!/usr/bin/env python3
"""
Normalize a COCO annotation file for MaskTerial training.

This script is intended for datasets like ``graphene.v4i.coco`` where:
- masks are stored as polygons instead of COCO RLE
- categories are duplicated or otherwise inconsistent

It converts every annotation mask to COCO RLE, remaps every annotation to a
single foreground class, and rewrites the dataset categories accordingly.

Examples
--------
Convert one file:
    python3 data_process/convert_graphene_coco.py \
      --src graphene.v4i.coco/train/_annotations.coco.json \
      --dst graphene.v4i.coco/train/_annotations_rle_1class.coco.json

Convert common Roboflow splits in-place into new files:
    python3 data_process/convert_graphene_coco.py \
      --dataset-dir graphene.v4i.coco
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

try:
    import pycocotools.mask as mask_util
except ImportError:  # pragma: no cover - fallback for lighter environments
    mask_util = None


DEFAULT_TARGET_CLASS = {"id": 1, "name": "gra", "supercategory": "gra"}


def _point_on_segment(px, py, ax, ay, bx, by):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-9:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= 1e-9


def _point_in_polygon(px, py, points):
    inside = False
    total = len(points)
    for index in range(total):
        ax, ay = points[index]
        bx, by = points[(index + 1) % total]

        if _point_on_segment(px, py, ax, ay, bx, by):
            return True

        intersects = ((ay > py) != (by > py)) and (
            px < (bx - ax) * (py - ay) / ((by - ay) or 1e-12) + ax
        )
        if intersects:
            inside = not inside
    return inside


def _filled_indices_from_polygon(points, height, width):
    min_x = max(0, math.floor(min(x for x, _ in points)))
    max_x = min(width - 1, math.ceil(max(x for x, _ in points)))
    min_y = max(0, math.floor(min(y for _, y in points)))
    max_y = min(height - 1, math.ceil(max(y for _, y in points)))

    indices = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if _point_in_polygon(x + 0.5, y + 0.5, points):
                indices.add(y + x * height)
    return indices


def _indices_to_uncompressed_rle(indices, height, width):
    total_pixels = height * width
    if not indices:
        return {"size": [height, width], "counts": [total_pixels]}

    sorted_indices = sorted(indices)
    runs = []

    start = sorted_indices[0]
    end = start
    for index in sorted_indices[1:]:
        if index == end + 1:
            end = index
            continue
        runs.append((start, end))
        start = index
        end = index
    runs.append((start, end))

    counts = []
    cursor = 0
    for start, end in runs:
        counts.append(start - cursor)
        counts.append(end - start + 1)
        cursor = end + 1
    counts.append(total_pixels - cursor)
    return {"size": [height, width], "counts": counts}


def _polygon_to_uncompressed_rle(segmentation, height, width):
    filled_indices = set()
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon), 2)
        ]
        if len(points) < 3:
            continue
        filled_indices.update(_filled_indices_from_polygon(points, height, width))
    return _indices_to_uncompressed_rle(filled_indices, height, width)


def _encode_rle(segmentation, height, width):
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle["counts"] = counts.decode("utf-8")
        return rle

    if not isinstance(segmentation, list):
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")

    if mask_util is None:
        return _polygon_to_uncompressed_rle(segmentation, height, width)

    # COCO polygon annotations may contain one or more polygons per instance.
    rles = mask_util.frPyObjects(segmentation, height, width)
    rle = mask_util.merge(rles) if isinstance(rles, list) else rles

    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return rle


def convert_coco_file(
    src_path: Path,
    dst_path: Path,
    *,
    target_class: dict,
    drop_empty_images: bool,
):
    with src_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    id_to_name = {category["id"]: category.get("name", "") for category in categories}
    seg_types_before = Counter(type(ann.get("segmentation")).__name__ for ann in annotations)
    categories_before = Counter(
        id_to_name.get(ann.get("category_id"), str(ann.get("category_id")))
        for ann in annotations
    )

    images_by_id = {image["id"]: image for image in images}
    converted_annotations = []
    used_image_ids = set()

    for annotation in annotations:
        image = images_by_id.get(annotation["image_id"])
        if image is None:
            raise KeyError(
                f"Annotation {annotation.get('id')} references missing image_id "
                f"{annotation.get('image_id')}"
            )

        converted = dict(annotation)
        converted["category_id"] = target_class["id"]
        converted["segmentation"] = _encode_rle(
            annotation["segmentation"],
            image["height"],
            image["width"],
        )
        converted_annotations.append(converted)
        used_image_ids.add(converted["image_id"])

    converted_images = images
    if drop_empty_images:
        converted_images = [
            image for image in images if image.get("id") in used_image_ids
        ]

    converted_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": converted_images,
        "annotations": converted_annotations,
        "categories": [target_class],
    }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as handle:
        json.dump(converted_coco, handle)

    print(f"Converted: {src_path} -> {dst_path}")
    print(f"  images: {len(images)} -> {len(converted_images)}")
    print(f"  annotations: {len(annotations)} -> {len(converted_annotations)}")
    print(f"  categories before: {dict(categories_before)}")
    print(f"  segmentation types before: {dict(seg_types_before)}")
    print(f"  categories after: {[target_class]}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert Roboflow-style COCO polygons into single-class RLE COCO."
    )
    parser.add_argument("--src", type=Path, help="Input COCO annotation file.")
    parser.add_argument("--dst", type=Path, help="Output COCO annotation file.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Dataset directory containing train/test/valid subdirectories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "valid"],
        help="Splits to process when --dataset-dir is used.",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=DEFAULT_TARGET_CLASS["id"],
        help="Foreground class id in the output COCO.",
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET_CLASS["name"],
        help="Foreground class name in the output COCO.",
    )
    parser.add_argument(
        "--target-supercategory",
        default=DEFAULT_TARGET_CLASS["supercategory"],
        help="Foreground class supercategory in the output COCO.",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Drop images that do not have any remaining annotations.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    target_class = {
        "id": args.target_id,
        "name": args.target_name,
        "supercategory": args.target_supercategory,
    }

    if args.dataset_dir is not None:
        for split in args.splits:
            src_path = args.dataset_dir / split / "_annotations.coco.json"
            if not src_path.exists():
                continue
            dst_path = args.dataset_dir / split / "_annotations_rle_1class.coco.json"
            convert_coco_file(
                src_path,
                dst_path,
                target_class=target_class,
                drop_empty_images=args.drop_empty_images,
            )
        return

    if args.src is None or args.dst is None:
        raise SystemExit("Use either --dataset-dir or both --src and --dst.")

    convert_coco_file(
        args.src,
        args.dst,
        target_class=target_class,
        drop_empty_images=args.drop_empty_images,
    )


if __name__ == "__main__":
    main()
