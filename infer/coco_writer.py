from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


SEGMENTATION_CATEGORIES = [
    {"id": 1, "name": "flake", "supercategory": "flake"},
]

LAYER_CATEGORIES = [
    {"id": 1, "name": "layer_1", "supercategory": "layer"},
    {"id": 2, "name": "layer_2", "supercategory": "layer"},
    {"id": 3, "name": "layer_3", "supercategory": "layer"},
    {"id": 4, "name": "layer_4", "supercategory": "layer"},
    {"id": 5, "name": "layer_5", "supercategory": "layer"},
    {"id": 6, "name": "layer_6", "supercategory": "layer"},
    {"id": 7, "name": "layer_7", "supercategory": "layer"},
    {"id": 8, "name": "gt_7", "supercategory": "layer"},
]

LAYER_CATEGORY_ID = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    ">7": 8,
}

CATEGORY_NAME_BY_ID = {category["id"]: category["name"] for category in LAYER_CATEGORIES}


def image_record(image_id: int, file_name: str, width: int, height: int) -> dict[str, Any]:
    return {
        "id": int(image_id),
        "file_name": str(file_name),
        "width": int(width),
        "height": int(height),
    }


def encode_mask(mask: np.ndarray) -> tuple[dict[str, Any], float, list[float]]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask_u8))
    area = float(mask_utils.area(rle))
    bbox = [float(value) for value in mask_utils.toBbox(rle).tolist()]
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"]}, area, bbox


def annotation_record(
    annotation_id: int,
    image_id: int,
    mask: np.ndarray,
    category_id: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    segmentation, area, bbox = encode_mask(mask)
    record = {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }
    if extra:
        record.update(_jsonify(extra))
    return record


def write_coco(
    path: str | Path,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    info: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": _jsonify(info or {}),
        "licenses": [],
        "images": _jsonify(images),
        "annotations": _jsonify(annotations),
        "categories": _jsonify(categories),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = []
    known = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in known:
                known.add(key)
                extra_fields.append(key)
    columns = list(fieldnames) + extra_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonify(row))


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
