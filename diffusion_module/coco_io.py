"""COCO dataset utilities for diffusion-powered augmentation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class CocoSample:
	"""Container representing one COCO image entry and its annotations."""

	image_info: Dict
	annotations: List[Dict]
	image_path: Path

	@property
	def width(self) -> int:
		return int(self.image_info["width"])

	@property
	def height(self) -> int:
		return int(self.image_info["height"])


class CocoDataset:
	"""Utility class to read, update, and persist COCO annotations."""

	def __init__(self, ann_path: Path):
		self.path = ann_path
		with open(ann_path, "r", encoding="utf-8") as handle:
			self.data = json.load(handle)

		self.images = {img["id"]: img for img in self.data.get("images", [])}
		self.annotations: List[Dict] = list(self.data.get("annotations", []))
		self.categories = {cat["id"]: cat for cat in self.data.get("categories", [])}

		self.anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
		for ann in self.annotations:
			self.anns_by_image[ann["image_id"]].append(ann)

		self.next_image_id = (max(self.images) + 1) if self.images else 1
		self.next_annotation_id = (
			max(ann["id"] for ann in self.annotations) + 1 if self.annotations else 1
		)

	def iter_samples(self, image_root: Path) -> Iterable[CocoSample]:
		for image_id, image_info in self.images.items():
			file_name = image_info["file_name"]
			image_path = (image_root / file_name).expanduser().resolve()
			annotations = self.anns_by_image.get(image_id, [])
			yield CocoSample(image_info=image_info, annotations=annotations, image_path=image_path)

	@staticmethod
	def annotation_mask(ann: Dict, height: int, width: int) -> np.ndarray:
		seg = ann.get("segmentation")
		if seg is None:
			return np.zeros((height, width), dtype=np.uint8)
		rle = _segmentation_to_rle(seg, height, width)
		mask = mask_utils.decode(rle)
		if mask.ndim == 3:
			mask = mask[..., 0]
		return mask.astype(np.uint8)

	def add_image(self, file_name: str, width: int, height: int) -> int:
		image_id = self.next_image_id
		self.next_image_id += 1
		entry = {
			"id": image_id,
			"file_name": file_name,
			"width": int(width),
			"height": int(height),
		}
		self.data.setdefault("images", []).append(entry)
		self.images[image_id] = entry
		return image_id

	def add_annotation(
		self,
		image_id: int,
		category_id: int,
		mask: np.ndarray,
	) -> int:
		rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
		if isinstance(rle["counts"], bytes):
			rle["counts"] = rle["counts"].decode("ascii")
		bbox = mask_utils.toBbox(rle).tolist()
		area = float(mask_utils.area(rle))

		ann_id = self.next_annotation_id
		self.next_annotation_id += 1
		annotation = {
			"id": ann_id,
			"image_id": image_id,
			"category_id": category_id,
			"segmentation": rle,
			"area": area,
			"bbox": bbox,
			"iscrowd": 0,
		}
		self.data.setdefault("annotations", []).append(annotation)
		self.annotations.append(annotation)
		self.anns_by_image[image_id].append(annotation)
		return ann_id

	def save(self, output_path: Path) -> None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, "w", encoding="utf-8") as handle:
			json.dump(self.data, handle, indent=2)


def _segmentation_to_rle(segmentation, height: int, width: int):
	if isinstance(segmentation, list):
		rles = mask_utils.frPyObjects(segmentation, height, width)
		return mask_utils.merge(rles)
	if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
		rles = mask_utils.frPyObjects(segmentation, height, width)
		return mask_utils.merge(rles)
	return segmentation
