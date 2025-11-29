"""CLI for COCO-native diffusion + SAM augmentation."""

from __future__ import annotations

import argparse
import json
import logging
import random
import string
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


def find_visible_annotations(
	crop_rect: Tuple[int, int, int, int],
	annotations: List[dict],
) -> List[dict]:
	"""Find all annotations that intersect with the crop region.
	
	Args:
		crop_rect: Crop bounding box as [x1, y1, x2, y2]
		annotations: List of COCO annotation dicts with 'bbox' field [x, y, w, h]
	
	Returns:
		List of annotations that have any overlap with crop_rect
	"""
	crop_x1, crop_y1, crop_x2, crop_y2 = crop_rect
	visible_anns = []
	
	for ann in annotations:
		bbox = ann["bbox"]  # [x, y, w, h]
		ann_x1 = bbox[0]
		ann_y1 = bbox[1]
		ann_x2 = bbox[0] + bbox[2]
		ann_y2 = bbox[1] + bbox[3]
		
		# Check for intersection
		# Two rectangles intersect if they overlap in both X and Y axes
		x_overlap = max(0, min(crop_x2, ann_x2) - max(crop_x1, ann_x1))
		y_overlap = max(0, min(crop_y2, ann_y2) - max(crop_y1, ann_y1))
		
		intersection_area = x_overlap * y_overlap
		
		if intersection_area > 0:
			visible_anns.append(ann)
	
	return visible_anns


def map_annotations_to_crop(
	visible_annotations: List[dict],
	crop_origin: Tuple[int, int],
	crop_size: int = 512,
	min_box_size: int = 5,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
	"""Map global annotations to local crop coordinates.
	
	Args:
		visible_annotations: List of annotation dicts with 'bbox' and 'category_id' fields
		crop_origin: (x1, y1) of crop window in original image
		crop_size: Size of the crop (default: 512)
		min_box_size: Minimum width/height after clamping (prevents SAM errors)
	
	Returns:
		Tuple of (boxes_xyxy, category_ids, annotation_ids):
			- boxes_xyxy: List of np.array([x1, y1, x2, y2]) in local coordinates
			- category_ids: List of category IDs
			- annotation_ids: List of original annotation IDs
	"""
	boxes = []
	category_ids = []
	annotation_ids = []
	
	crop_x1, crop_y1 = crop_origin
	
	for ann in visible_annotations:
		global_bbox = ann["bbox"]  # [x, y, w, h]
		
		# Map to local coordinates
		local_x1 = global_bbox[0] - crop_x1
		local_y1 = global_bbox[1] - crop_y1
		local_x2 = local_x1 + global_bbox[2]
		local_y2 = local_y1 + global_bbox[3]
		
		# Clamp to [0, crop_size]
		local_x1 = max(0.0, min(float(crop_size), local_x1))
		local_y1 = max(0.0, min(float(crop_size), local_y1))
		local_x2 = max(0.0, min(float(crop_size), local_x2))
		local_y2 = max(0.0, min(float(crop_size), local_y2))
		
		# Validity check: skip if box is too small after clamping
		width = local_x2 - local_x1
		height = local_y2 - local_y1
		
		if width < min_box_size or height < min_box_size:
			LOGGER.debug(
				"Skipping ann %d: box size %.1f×%.1f is too small (< %dpx)",
				ann["id"], width, height, min_box_size
			)
			continue
		
		# Create box in xyxy format
		box = np.array([local_x1, local_y1, local_x2, local_y2], dtype=np.float32)
		boxes.append(box)
		category_ids.append(int(ann["category_id"]))
		annotation_ids.append(ann["id"])
	
	return boxes, category_ids, annotation_ids




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
	parser.add_argument("--batch-size", type=int, default=4, help="Batch size for parallel GPU processing (default: 4)")
	parser.add_argument("--max-samples", type=int)
	parser.add_argument("--output-suffix", default="_aug")
	parser.add_argument("--image-ext", default=".png")
	parser.add_argument("--device", default=None, help="Force cuda or cpu")
	parser.add_argument("--keep-safety-checker", action="store_true")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	return parser.parse_args()


def process_batch(
	batch_buffer: List[dict],
	augmentor: DiffusionAugmentor,
	coco_ds: CocoDataset,
	output_image_dir: Path,
	image_root: Path,
	config: AugmentationConfig,
) -> Tuple[int, List[dict]]:
	"""Process a batch of cropped images in parallel using batch GPU processing.
	
	Args:
		batch_buffer: List of dicts with 'image', 'boxes', 'category_ids', 'prompt', etc.
		augmentor: DiffusionAugmentor instance
		coco_ds: CocoDataset instance
		output_image_dir: Output directory for images
		image_root: Root directory for images
		config: AugmentationConfig
	
	Returns:
		Tuple of (num_successful_patches, metadata_entries)
	"""
	if not batch_buffer:
		return 0, []
	
	# 1. Unpack batch buffer
	images_batch = [item['image'] for item in batch_buffer]
	prompts_batch = [item['prompt'] for item in batch_buffer]
	
	LOGGER.info("Processing batch of %d images", len(batch_buffer))
	
	# 2. Batch generation (parallel processing on GPU)
	try:
		augmented_images = augmentor.generate_batch(images_batch, prompts_batch)
	except Exception as exc:
		LOGGER.error("Batch generation failed: %s", exc)
		return 0, []
	
	# 3. SAM post-processing and registration (serial, but uses batch SAM)
	num_successful_patches = 0
	num_successful_annotations = 0
	metadata_entries = []
	
	for i, (aug_img, item) in enumerate(zip(augmented_images, batch_buffer)):
		try:
			# Run batch SAM on the augmented image with all boxes
			boxes = item['boxes']
			category_ids = item['category_ids']
			
			if not boxes:
				LOGGER.warning("No valid boxes for patch %d; skipping", i)
				continue
			
			LOGGER.debug(
				"Running batch SAM with %d boxes for patch from ann %s",
				len(boxes), item.get('source_ann_ids', 'unknown')
			)
			
			# Batch SAM prediction
			masks = augmentor.run_sam_batch(aug_img, boxes)
			
			if len(masks) != len(category_ids):
				LOGGER.error(
					"Mask count mismatch: got %d masks for %d boxes; skipping patch",
					len(masks), len(boxes)
				)
				continue
			
			# Save the augmented patch once
			random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
			output_filename = f"{item['source_filename']}_crop_{item['crop_id']}_{random_suffix}{config.image_ext}"
			output_path = output_image_dir / output_filename
			
			aug_img.save(output_path)
			LOGGER.debug("Saved augmented patch to: %s", output_path)
			
			# Register in COCO (one image, multiple annotations)
			try:
				file_name = output_path.relative_to(image_root).as_posix()
				LOGGER.debug("Using relative path for COCO: %s", file_name)
			except ValueError:
				file_name = str(output_path)
				LOGGER.warning(
					"Output directory (%s) is not under image_root (%s). "
					"Using absolute path in COCO JSON: %s. "
					"You may need to update the COCO JSON or image_root when training.",
					output_image_dir, image_root, file_name
				)
			
			# Register the image once
			new_image_id = coco_ds.add_image(file_name=file_name, width=512, height=512)
			
			# Register all annotations for this image
			valid_mask_count = 0
			for mask, cat_id in zip(masks, category_ids):
				# Validate mask
				if np.sum(mask) == 0:
					LOGGER.debug("Skipping empty mask for category %d in image %d", cat_id, new_image_id)
					continue
				
				# Verify mask shape
				if mask.shape != (512, 512):
					LOGGER.error(
						"Mask shape mismatch for category %d: expected (512, 512), got %s",
						cat_id, mask.shape
					)
					continue
				
				# Add annotation to COCO
				new_ann_id = coco_ds.add_annotation(new_image_id, cat_id, mask)
				valid_mask_count += 1
				
				LOGGER.debug(
					"Added annotation %d for image %d (category: %d)",
					new_ann_id, new_image_id, cat_id
				)
			
			if valid_mask_count == 0:
				LOGGER.warning("No valid masks for patch %d; image saved but no annotations", i)
			else:
				num_successful_patches += 1
				num_successful_annotations += valid_mask_count
				
				LOGGER.info(
					"Successfully processed patch %d: %d/%d annotations valid",
					i, valid_mask_count, len(masks)
				)
			
			metadata_entries.append({
				"source_image_id": item.get('source_image_id'),
				"source_annotation_ids": item.get('source_ann_ids', []),
				"source_file": item.get('source_file'),
				"crop_origin": item.get('crop_origin'),
				"augmented_file": str(output_path),
				"new_image_id": new_image_id,
				"category_ids": category_ids,
				"num_valid_annotations": valid_mask_count,
			})
			
		except Exception as exc:
			LOGGER.exception("Failed to process batch item %d: %s", i, exc)
			continue
	
	LOGGER.info(
		"Batch complete: %d patches generated, %d total annotations",
		num_successful_patches, num_successful_annotations
	)
	
	return num_successful_patches, metadata_entries


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
	batch_buffer = []  # Initialize batch buffer
	
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
		
		# Generate crop ID counter for this image
		crop_id_counter = 0
		
		for center_ann in target_anns:
			ann_id = center_ann["id"]
			global_bbox = center_ann["bbox"]  # [x, y, w, h]
			
			# 1. Perform centered cropping based on the center annotation
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
			
			# 2. Find ALL annotations visible in this crop (not just the center one)
			crop_x1, crop_y1 = crop_origin
			crop_rect = (crop_x1, crop_y1, crop_x1 + 512, crop_y1 + 512)
			
			visible_annotations = find_visible_annotations(crop_rect, annotations)
			
			LOGGER.debug(
				"Crop centered on ann %d has %d visible annotations (of %d total in image)",
				ann_id, len(visible_annotations), len(annotations)
			)
			
			# 3. Map all visible annotations to local coordinates
			boxes, category_ids, ann_ids = map_annotations_to_crop(
				visible_annotations,
				crop_origin,
				crop_size=512,
				min_box_size=5
			)
			
			if not boxes:
				LOGGER.debug(
					"No valid boxes after coordinate mapping for crop centered on ann %d; skipping",
					ann_id
				)
				continue
			
			LOGGER.debug(
				"Crop centered on ann %d: %d/%d visible annotations passed validation",
				ann_id, len(boxes), len(visible_annotations)
			)
			
			# 4. Add to batch buffer with ALL visible objects
			prompt = augmentor.get_next_prompt()
			batch_buffer.append({
				'image': cropped_img,
				'boxes': boxes,  # List of boxes (multi-object)
				'category_ids': category_ids,  # List of category IDs
				'prompt': prompt,
				'crop_id': crop_id_counter,
				'source_ann_ids': ann_ids,  # List of source annotation IDs
				'source_filename': Path(sample.image_info["file_name"]).stem,
				'source_image_id': sample.image_info["id"],
				'source_file': str(sample.image_path),
				'crop_origin': crop_origin,
			})
			
			crop_id_counter += 1
			
			# 5. When buffer is full, process batch
			if len(batch_buffer) >= args.batch_size:
				num_successful, batch_metadata = process_batch(
					batch_buffer, augmentor, coco_ds, output_image_dir, image_root, config
				)
				patches_generated += num_successful
				metadata_entries.extend(batch_metadata)
				batch_buffer = []  # Reset buffer
	
	# 6. Process remaining items in buffer
	if batch_buffer:
		LOGGER.info("Processing remaining %d items in batch buffer", len(batch_buffer))
		num_successful, batch_metadata = process_batch(
			batch_buffer, augmentor, coco_ds, output_image_dir, image_root, config
		)
		patches_generated += num_successful
		metadata_entries.extend(batch_metadata)
	
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
