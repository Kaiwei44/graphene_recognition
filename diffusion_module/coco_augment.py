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


def get_centered_crop(
	image: Image.Image,
	bbox: List[float],
	crop_size: int = 512,
) -> Tuple[Image.Image, Tuple[int, int]]:
	"""Crop a square patch centered on a COCO bbox.
	
	Args:
		image: PIL Image object (original large image)
		bbox: COCO bbox in [x, y, w, h] format
		crop_size: Size of the square crop (default: 512)
	
	Returns:
		cropped_image: The cropped patch
		crop_origin: (x1, y1) coordinates of the crop window in the original image
	"""
	x, y, w, h = bbox
	img_width, img_height = image.size
	
	# Compute center of the bbox
	cx = x + w / 2
	cy = y + h / 2
	
	# Compute top-left of crop window
	x1 = cx - crop_size / 2
	y1 = cy - crop_size / 2
	
	# Boundary clamping: ensure crop window is within image bounds
	x1 = max(0, min(x1, img_width - crop_size))
	y1 = max(0, min(y1, img_height - crop_size))
	
	# If image is smaller than crop_size, clamp to image bounds
	x1 = max(0, x1)
	y1 = max(0, y1)
	x2 = min(img_width, x1 + crop_size)
	y2 = min(img_height, y1 + crop_size)
	
	# Perform the crop
	cropped_image = image.crop((x1, y1, x2, y2))
	
	# Return crop origin as integers
	crop_origin = (int(x1), int(y1))
	
	return cropped_image, crop_origin



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
		"--output-dir",
		type=Path,
		help="Root directory for all outputs (images and annotations). Overrides --output-image-dir and --output-ann.",
	)
	parser.add_argument(
		"--output-image-dir",
		type=Path,
		help="Directory for augmented images (defaults to <output-dir> or --image-root)",
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
	parser.add_argument("--prompt", action="append", help="Prompt(s) to use. Can be specified multiple times.")
	parser.add_argument("--prompt-file", type=Path, help="File containing prompts (one per line)")
	parser.add_argument("--prompt-strategy", default="random", choices=["random", "cycle"])
	parser.add_argument("--negative-prompt", default=AugmentationConfig.negative_prompt)
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
	
	if args.output_dir:
		output_dir = args.output_dir.expanduser().resolve()
		output_dir.mkdir(parents=True, exist_ok=True)
		# Default images to output_dir if not specified
		output_image_dir = (args.output_image_dir or output_dir).expanduser().resolve()
		
		# Default annotation to output_dir/filename
		if args.output_ann:
			output_ann = args.output_ann.expanduser().resolve()
		else:
			# If output_dir is same as input dir, use _aug suffix to avoid overwrite
			if output_dir == args.ann.parent.expanduser().resolve():
				output_ann = output_dir / f"{args.ann.stem}_aug.coco.json"
			else:
				# If saving to a new directory, keep the original filename (clean copy)
				output_ann = output_dir / args.ann.name
	else:
		# Legacy behavior
		output_image_dir = (args.output_image_dir or image_root).expanduser().resolve()
		output_ann = args.output_ann
		if output_ann is None:
			output_ann = args.ann.with_name(f"{args.ann.stem}_aug.coco.json")
		output_ann = output_ann.expanduser().resolve()

	output_image_dir.mkdir(parents=True, exist_ok=True)

	metadata_path = args.metadata_path
	if metadata_path is None:
		metadata_path = output_ann.with_suffix(".metadata.json")
	metadata_path = metadata_path.expanduser().resolve()

	if args.device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	prompts = []
	if args.prompt:
		prompts.extend(args.prompt)
	if args.prompt_file:
		with open(args.prompt_file, "r", encoding="utf-8") as f:
			prompts.extend([line.strip() for line in f if line.strip()])
	
	if not prompts:
		prompts = list(AugmentationConfig.prompts)

	config = AugmentationConfig(
		sd_model_id=args.sd_model_id,
		sam_model_type=args.sam_model_type,
		sam_checkpoint=args.sam_checkpoint,
		prompts=prompts,
		prompt_strategy=args.prompt_strategy,
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
	patches_generated = 0
	
	for sample in coco_ds.iter_samples(image_root):
		if config.max_samples is not None and processed >= config.max_samples:
			break
		if not sample.image_path.exists():
			LOGGER.warning("Image %s not found; skipping", sample.image_path)
			continue
		
		annotations = sample.annotations
		if not annotations:
			LOGGER.info("Image %s has no annotations; skipping", sample.image_path.name)
			continue
		
		# Load the original image once
		try:
			original_image = Image.open(sample.image_path).convert("RGB")
		except Exception as exc:
			LOGGER.exception("Failed to load image %s: %s", sample.image_path, exc)
			continue
		
		# Random sampling strategy: select up to 3 annotations per image
		max_patches_per_image = 3
		target_anns = np.random.choice(
			annotations,
			size=min(len(annotations), max_patches_per_image),
			replace=False
		)
		
		processed += 1
		
		for ann in target_anns:
			ann_id = ann["id"]
			category_id = int(ann["category_id"])
			global_bbox = ann["bbox"]  # [x, y, w, h]
			
			# 1. Perform centered cropping
			try:
				cropped_img, crop_origin = get_centered_crop(original_image, global_bbox, crop_size=512)
			except Exception as exc:
				LOGGER.warning("Failed to crop annotation %d: %s", ann_id, exc)
				continue
			
			# Ensure cropped image is exactly 512x512 (pad if necessary for edge cases)
			if cropped_img.size != (512, 512):
				LOGGER.debug(
					"Cropped image for ann %d is %dx%d instead of 512x512; applying reflection padding",
					ann_id, cropped_img.size[0], cropped_img.size[1]
				)
				# Convert PIL to numpy for reflection padding
				img_array = np.array(cropped_img)
				curr_h, curr_w = img_array.shape[:2]
				
				# Calculate padding needed (only on right and bottom)
				h_pad = 512 - curr_h
				w_pad = 512 - curr_w
				
				# Apply reflection padding: ((top, bottom), (left, right), (channels))
				# Critical: (0, h_pad) means "no padding on top, pad on bottom"
				# This maintains the (0,0) anchor point for coordinate mapping
				padded_array = np.pad(
					img_array,
					pad_width=((0, h_pad), (0, w_pad), (0, 0)),
					mode='reflect'
				)
				
				# Convert back to PIL
				cropped_img = Image.fromarray(padded_array)
			
			# 2. Coordinate mapping: global bbox -> local bbox on 512x512 patch
			local_x1 = global_bbox[0] - crop_origin[0]
			local_y1 = global_bbox[1] - crop_origin[1]
			local_x2 = local_x1 + global_bbox[2]
			local_y2 = local_y1 + global_bbox[3]
			
			# Clamp to [0, 512]
			local_x1 = max(0.0, min(512.0, local_x1))
			local_y1 = max(0.0, min(512.0, local_y1))
			local_x2 = max(0.0, min(512.0, local_x2))
			local_y2 = max(0.0, min(512.0, local_y2))
			
			# Construct SAM prompt box
			input_box = np.array([local_x1, local_y1, local_x2, local_y2], dtype=np.float32)
			
			# 3. Validity assertion: skip if box area is too small
			box_area = (local_x2 - local_x1) * (local_y2 - local_y1)
			if box_area < 50:
				LOGGER.debug(
					"Skipping ann %d: box area %.1f px² is too small (< 50 px²)",
					ann_id, box_area
				)
				continue
			
			# 4. Call generation pipeline
			try:
				prompt = augmentor.get_next_prompt()
				augmented_image, sam_masks = augmentor.generate_with_boxes(
					cropped_img, [input_box], prompt
				)
			except Exception as exc:
				LOGGER.exception("Failed to augment ann %d from %s: %s", ann_id, sample.image_path, exc)
				continue
			
			# 5. Validate SAM results
			if not sam_masks or len(sam_masks) == 0:
				LOGGER.warning("SAM returned no masks for ann %d; skipping", ann_id)
				continue
			
			# 6. Prepare and validate mask
			sam_mask = sam_masks[0]
			mask_binary = (sam_mask > 0).astype(np.uint8)
			
			if np.sum(mask_binary) == 0:
				LOGGER.warning("SAM predicted empty mask for ann %d; skipping", ann_id)
				continue
			
			# Verify mask shape matches expected 512x512
			if mask_binary.shape != (512, 512):
				LOGGER.error(
					"Mask shape mismatch for ann %d: expected (512, 512), got %s",
					ann_id, mask_binary.shape
				)
				continue
			
			# Log mask statistics for verification
			LOGGER.debug(
				"Mask for ann %d: shape=%s, nonzero_pixels=%d, dtype=%s",
				ann_id, mask_binary.shape, np.sum(mask_binary), mask_binary.dtype
			)
			
			# 7. Save the cropped augmented patch
			# Generate unique filename
			import random
			import string
			random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
			orig_file_name = Path(sample.image_info["file_name"]).stem
			output_filename = f"{orig_file_name}_crop_{ann_id}_{random_suffix}{config.image_ext}"
			output_path = output_image_dir / output_filename
			
			augmented_image.save(output_path)
			LOGGER.debug("Saved augmented patch to: %s", output_path)
			
			# 8. Register new COCO entries
			# Determine the file_name for COCO JSON
			# Strategy: Try to make it relative to image_root if possible,
			# otherwise use absolute path for clarity
			try:
				# Try to compute relative path from image_root
				file_name = output_path.relative_to(image_root).as_posix()
				LOGGER.debug("Using relative path for COCO: %s", file_name)
			except ValueError:
				# output_image_dir is not under image_root
				# Use absolute path to avoid confusion
				file_name = str(output_path)
				LOGGER.warning(
					"Output directory (%s) is not under image_root (%s). "
					"Using absolute path in COCO JSON: %s. "
					"You may need to update the COCO JSON or image_root when training.",
					output_image_dir, image_root, file_name
				)
			
			# Add image entry with hard-coded 512x512 dimensions
			new_image_id = coco_ds.add_image(
				file_name=file_name,
				width=512,
				height=512,
			)
			
			# Add annotation with SAM-generated mask
			# bbox and area will be recomputed from mask inside add_annotation
			new_ann_id = coco_ds.add_annotation(new_image_id, category_id, mask_binary)
			
			# Verify the annotation was added correctly
			LOGGER.debug(
				"Added annotation %d for image %d (file: %s, category: %d)",
				new_ann_id, new_image_id, file_name, category_id
			)
			
			patches_generated += 1
			LOGGER.debug("Generated patch %d from ann %d (image %s)", patches_generated, ann_id, sample.image_path.name)
			
			metadata_entries.append(
				{
					"source_image_id": sample.image_info["id"],
					"source_annotation_id": ann_id,
					"source_file": str(sample.image_path),
					"crop_origin": crop_origin,
					"augmented_file": str(output_path),
					"new_image_id": new_image_id,
					"category_id": category_id,
				}
			)
	
	LOGGER.info("Processed %d images, generated %d augmented patches", processed, patches_generated)
	coco_ds.save(output_ann)
	LOGGER.info("Wrote updated COCO file to %s", output_ann)
	
	with open(metadata_path, "w", encoding="utf-8") as handle:
		json.dump(metadata_entries, handle, indent=2)
	LOGGER.info("Wrote metadata to %s", metadata_path)
	
	# Provide guidance on using the generated dataset
	LOGGER.info("=" * 60)
	LOGGER.info("DATASET GENERATION COMPLETE")
	LOGGER.info("=" * 60)
	LOGGER.info("Generated %d augmented patches from %d source images", patches_generated, processed)
	LOGGER.info("Output images: %s", output_image_dir)
	LOGGER.info("Output COCO JSON: %s", output_ann)
	
	# Check if output_image_dir is under image_root
	try:
		output_image_dir.relative_to(image_root)
		LOGGER.info("✓ Output images are under image_root, ready for training")
		LOGGER.info("  Use this command for training:")
		LOGGER.info("    --image-root %s", image_root)
		LOGGER.info("    --ann %s", output_ann)
	except ValueError:
		LOGGER.warning("⚠ Output images are NOT under image_root!")
		LOGGER.warning("  Option 1: Use output_image_dir as the new image_root:")
		LOGGER.warning("    --image-root %s", output_image_dir)
		LOGGER.warning("    --ann %s", output_ann)
		LOGGER.warning("  Option 2: Move/copy images to be under image_root:")
		LOGGER.warning("    cp -r %s/* %s/", output_image_dir, image_root)
	LOGGER.info("=" * 60)


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
