"""CLI for COCO-native diffusion + SAM augmentation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from diffusion_module.coco_io import CocoDataset, CocoSample
from diffusion_module.diffusion_model import (
	AugmentationConfig,
	DiffusionAugmentor,
	_configure_logging,
)

LOGGER = logging.getLogger("maskterial.diffusion.coco")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Augment COCO datasets via Stable Diffusion + SAM")
	parser.add_argument("--image-root", type=Path, required=True, help="Folder containing training images")
	parser.add_argument("--ann", type=Path, required=True, help="COCO annotation JSON")
	parser.add_argument(
		"--output-ann",
		type=Path,
		help="Path to write the updated COCO JSON (defaults to <ann> with _aug suffix)",
	)
	parser.add_argument(
		"--output-image-dir",
		type=Path,
		help="Directory for augmented images (defaults to --image-root)",
	)
	parser.add_argument(
		"--metadata-path",
		type=Path,
		help="Optional JSON file to log per-sample provenance",
	)

	parser.add_argument("--sd-model-id", default="runwayml/stable-diffusion-v1-5")
	parser.add_argument("--sam-checkpoint", type=Path, required=True)
	parser.add_argument(
		"--sam-model-type",
		default="vit_h",
		choices=["default", "vit_h", "vit_l", "vit_b"],
	)
	parser.add_argument("--prompt", default=AugmentationConfig.prompt)
	parser.add_argument("--negative-prompt", default=None)
	parser.add_argument("--strength", type=float, default=0.35)
	parser.add_argument("--guidance-scale", type=float, default=7.5)
	parser.add_argument("--num-inference-steps", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--bbox-padding", type=int, default=8, help="Pixels to expand annotation bboxes")
	parser.add_argument("--max-samples", type=int)
	parser.add_argument("--output-suffix", default="_aug")
	parser.add_argument("--image-ext", default=".png")
	parser.add_argument("--device", default=None, help="Force cuda or cpu")
	parser.add_argument("--keep-safety-checker", action="store_true")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	_configure_logging(args.verbose)

	image_root = args.image_root.expanduser().resolve()
	output_image_dir = (args.output_image_dir or image_root).expanduser().resolve()
	output_image_dir.mkdir(parents=True, exist_ok=True)

	output_ann = args.output_ann
	if output_ann is None:
		output_ann = args.ann.with_name(f"{args.ann.stem}_aug.coco.json")
	output_ann = output_ann.expanduser().resolve()

	metadata_path = args.metadata_path
	if metadata_path is None:
		metadata_path = output_ann.with_suffix(".metadata.json")
	metadata_path = metadata_path.expanduser().resolve()

	if args.device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	config = AugmentationConfig(
		sd_model_id=args.sd_model_id,
		sam_model_type=args.sam_model_type,
		sam_checkpoint=args.sam_checkpoint,
		prompt=args.prompt,
		negative_prompt=args.negative_prompt,
		strength=args.strength,
		guidance_scale=args.guidance_scale,
		num_inference_steps=args.num_inference_steps,
		seed=args.seed,
		bbox_padding=args.bbox_padding,
		output_suffix=args.output_suffix,
		image_ext=args.image_ext,
		mask_ext=args.image_ext,
		max_samples=args.max_samples,
		skip_existing=args.skip_existing,
		device=device,
		disable_safety_checker=not args.keep_safety_checker,
	)

	coco_ds = CocoDataset(args.ann.expanduser().resolve())
	augmentor = DiffusionAugmentor(config)

	metadata_entries = []
	processed = 0
	for sample in coco_ds.iter_samples(image_root):
		if config.max_samples is not None and processed >= config.max_samples:
			break
		if not sample.image_path.exists():
			LOGGER.warning("Image %s not found; skipping", sample.image_path)
			continue
		boxes, cat_ids = _boxes_from_annotations(sample, padding=config.bbox_padding)
		if not boxes:
			LOGGER.info("Image %s has no annotations; skipping", sample.image_path.name)
			continue

		image = Image.open(sample.image_path).convert("RGB")
		try:
			augmented_image, sam_masks = augmentor.generate_with_boxes(image, boxes)
		except Exception as exc:
			LOGGER.exception("Failed to augment %s: %s", sample.image_path, exc)
			continue

		output_stem = augmentor._next_output_stem(output_image_dir, Path(sample.image_info["file_name"]).stem)
		valid_masks: List[np.ndarray] = []
		valid_categories: List[int] = []
		for category_id, sam_mask in zip(cat_ids, sam_masks):
			mask_binary = (sam_mask > 0).astype(np.uint8)
			if mask_binary.sum() == 0:
				continue
			valid_masks.append(mask_binary)
			valid_categories.append(category_id)

		if not valid_masks:
			LOGGER.warning("SAM predicted empty masks for %s; skipping save", sample.image_path.name)
			continue

		output_filename = f"{output_stem}{config.image_ext}"
		output_path = output_image_dir / output_filename
		augmented_image.save(output_path)

		file_name = output_filename
		try:
			file_name = output_path.relative_to(image_root).as_posix()
		except ValueError:
			pass

		new_image_id = coco_ds.add_image(
			file_name=file_name,
			width=augmented_image.width,
			height=augmented_image.height,
		)

		for category_id, mask_binary in zip(valid_categories, valid_masks):
			coco_ds.add_annotation(new_image_id, category_id, mask_binary)

		processed += 1
		metadata_entries.append(
			{
				"source_image_id": sample.image_info["id"],
				"source_file": str(sample.image_path),
				"augmented_file": str(output_path),
				"new_image_id": new_image_id,
				"num_annotations": len(valid_masks),
			}
		)

	LOGGER.info("Augmented %d images", processed)
	coco_ds.save(output_ann)
	LOGGER.info("Wrote updated COCO file to %s", output_ann)

	with open(metadata_path, "w", encoding="utf-8") as handle:
		json.dump(metadata_entries, handle, indent=2)
	LOGGER.info("Wrote metadata to %s", metadata_path)


def _boxes_from_annotations(sample: CocoSample, padding: int) -> Tuple[List[np.ndarray], List[int]]:
	boxes: List[np.ndarray] = []
	cat_ids: List[int] = []
	width, height = sample.width, sample.height
	for ann in sample.annotations:
		x, y, w, h = ann["bbox"]
		xmin = max(0, x - padding)
		ymin = max(0, y - padding)
		xmax = min(width - 1, x + w + padding)
		ymax = min(height - 1, y + h + padding)
		if xmax <= xmin or ymax <= ymin:
			continue
		boxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.float32))
		cat_ids.append(int(ann["category_id"]))
	return boxes, cat_ids


if __name__ == "__main__":
	main()
