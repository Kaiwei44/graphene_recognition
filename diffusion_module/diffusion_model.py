"""Diffusion-powered microscopy data augmentation.

This module turns pairs of real microscope images and masks into new, stylized
samples by running Stable Diffusion (img2img) followed by Segment Anything
(SAM) to re-project the mask onto the generated image.  It is designed to be
invoked as a CLI tool but can also be reused as a Python module.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from segment_anything import SamPredictor, sam_model_registry


LOGGER = logging.getLogger("maskterial.diffusion")


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class DatasetItem:
	image_path: Path
	mask_path: Path


@dataclass
class AugmentationConfig:
	sd_model_id: str = "runwayml/stable-diffusion-v1-5"
	sam_model_type: str = "vit_h"
	sam_checkpoint: Path = Path("sam_vit_h_4b8939.pth")
	prompt: str = (
		"A microscope image of 2D material flakes on a substrate, high resolution, "
		"scientific photography."
	)
	negative_prompt: Optional[str] = None
	strength: float = 0.35
	guidance_scale: float = 7.5
	num_inference_steps: int = 50
	seed: int = 42
	bbox_padding: int = 8
	mask_threshold: int = 0
	output_suffix: str = "_aug"
	image_ext: str = ".png"
	mask_ext: str = ".png"
	max_samples: Optional[int] = None
	skip_existing: bool = False
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	disable_safety_checker: bool = True


def _configure_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="[%(asctime)s] %(levelname)s - %(message)s",
	)


def _resolve_path(path_str: str, root: Optional[Path]) -> Path:
	path = Path(path_str)
	if not path.is_absolute() and root:
		path = root / path
	return path.expanduser().resolve()


def load_dataset_from_manifest(
	manifest_path: Path,
	image_root: Optional[Path] = None,
	mask_root: Optional[Path] = None,
) -> List[DatasetItem]:
	with open(manifest_path, "r", encoding="utf-8") as handle:
		data = json.load(handle)

	if not isinstance(data, list):
		raise ValueError("Manifest must be a JSON list of {image_path, mask_path} objects")

	dataset: List[DatasetItem] = []
	for entry in data:
		image_path = _resolve_path(entry["image_path"], image_root)
		mask_path = _resolve_path(entry["mask_path"], mask_root)
		dataset.append(DatasetItem(image_path=image_path, mask_path=mask_path))

	return dataset


def auto_discover_dataset(image_dir: Path, mask_dir: Path) -> List[DatasetItem]:
	images = {
		path.stem: path
		for path in image_dir.rglob("*")
		if path.suffix.lower() in SUPPORTED_IMAGE_EXTS
	}
	masks = {
		path.stem: path
		for path in mask_dir.rglob("*")
		if path.suffix.lower() in SUPPORTED_IMAGE_EXTS
	}

	shared = sorted(images.keys() & masks.keys())
	dataset = [DatasetItem(images[stem], masks[stem]) for stem in shared]
	LOGGER.info("Auto-discovered %d paired samples", len(dataset))
	missing = set(images) - set(masks)
	if missing:
		LOGGER.warning("Skipping %d images with no matching mask", len(missing))
	return dataset


def _load_mask(mask_path: Path, threshold: int) -> np.ndarray:
	mask = Image.open(mask_path).convert("L")
	mask_np = np.array(mask)
	return (mask_np > threshold).astype(np.uint8)


def _component_boxes(
	mask: np.ndarray, padding: int, image_shape: Tuple[int, int]
) -> List[np.ndarray]:
	labeled, num_features = ndimage.label(mask)
	if num_features == 0:
		return []

	slices = ndimage.find_objects(labeled)
	boxes: List[np.ndarray] = []
	height, width = image_shape
	for slc in slices:
		if slc is None:
			continue
		ymin, ymax = slc[0].start, slc[0].stop
		xmin, xmax = slc[1].start, slc[1].stop
		xmin = max(0, xmin - padding)
		ymin = max(0, ymin - padding)
		xmax = min(width - 1, xmax + padding)
		ymax = min(height - 1, ymax + padding)
		boxes.append(np.array([xmin, ymin, xmax, ymax]))
	return boxes


class DiffusionAugmentor:
	def __init__(self, config: AugmentationConfig):
		self.config = config
		dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
		LOGGER.info("Loading Stable Diffusion model: %s", config.sd_model_id)
		pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
			config.sd_model_id,
			torch_dtype=dtype,
		)
		if config.disable_safety_checker:
			pipe.safety_checker = lambda images, clip_input: (images, False)
		pipe.enable_attention_slicing()
		pipe = pipe.to(config.device)
		self.pipe = pipe
		self.generator = torch.Generator(device=config.device).manual_seed(config.seed)

		LOGGER.info("Loading SAM checkpoint: %s", config.sam_checkpoint)
		sam_model = sam_model_registry[config.sam_model_type](checkpoint=str(config.sam_checkpoint))
		sam_model.to(device=config.device)
		self.sam_predictor = SamPredictor(sam_model)

	def generate_with_boxes(
		self,
		original_image: Image.Image,
		boxes_xyxy: Sequence[np.ndarray],
	) -> Tuple[Image.Image, List[np.ndarray]]:
		if not boxes_xyxy:
			raise ValueError("At least one box is required for augmentation")
		augmented = self.pipe(
			prompt=self.config.prompt,
			negative_prompt=self.config.negative_prompt,
			image=original_image,
			strength=self.config.strength,
			guidance_scale=self.config.guidance_scale,
			num_inference_steps=self.config.num_inference_steps,
			generator=self.generator,
		).images[0]
		augmented_np = np.array(augmented)
		self.sam_predictor.set_image(augmented_np)

		masks: List[np.ndarray] = []
		for box in boxes_xyxy:
			masks_pred, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
			masks.append(masks_pred[0].astype(np.uint8))
		return augmented, masks

	def augment_dataset(
		self,
		dataset: Sequence[DatasetItem],
		output_dir: Path,
	) -> List[dict]:
		images_dir = output_dir / "images"
		masks_dir = output_dir / "masks"
		images_dir.mkdir(parents=True, exist_ok=True)
		masks_dir.mkdir(parents=True, exist_ok=True)

		metadata: List[dict] = []
		for idx, item in enumerate(dataset):
			if self.config.max_samples is not None and idx >= self.config.max_samples:
				LOGGER.info("Reached max_samples=%d, stopping", self.config.max_samples)
				break
			LOGGER.info("[%d/%d] Processing %s", idx + 1, len(dataset), item.image_path.name)
			try:
				meta = self._augment_single(item, images_dir, masks_dir)
				if meta is not None:
					metadata.append(meta)
			except Exception as exc:  # pragma: no cover - defensive logging
				LOGGER.exception("Failed to process %s: %s", item.image_path, exc)
		return metadata

	def _augment_single(
		self,
		item: DatasetItem,
		images_dir: Path,
		masks_dir: Path,
	) -> Optional[dict]:
		original_image = Image.open(item.image_path).convert("RGB")
		mask = _load_mask(item.mask_path, self.config.mask_threshold)
		boxes = _component_boxes(mask, self.config.bbox_padding, original_image.size[::-1])
		if not boxes:
			LOGGER.warning("Mask %s is empty; skipping", item.mask_path)
			return None

		augmented, sam_masks = self.generate_with_boxes(original_image, boxes)
		combined_mask = self._merge_masks(sam_masks)

		output_stem = self._next_output_stem(images_dir, item.image_path.stem)
		image_path = images_dir / f"{output_stem}{self.config.image_ext}"
		mask_path = masks_dir / f"{output_stem}{self.config.mask_ext}"
		augmented.save(image_path)
		Image.fromarray(combined_mask).save(mask_path)

		LOGGER.info("Saved augmented pair: %s", output_stem)
		return {
			"source_image": str(item.image_path),
			"source_mask": str(item.mask_path),
			"augmented_image": str(image_path),
			"augmented_mask": str(mask_path),
		}

	def _merge_masks(self, masks: Sequence[np.ndarray]) -> np.ndarray:
		combined = np.zeros(masks[0].shape, dtype=np.uint8)
		for mask in masks:
			combined |= (mask > 0).astype(np.uint8)
		return combined.astype(np.uint8) * 255

	def _next_output_stem(self, images_dir: Path, base_stem: str) -> str:
		suffix = self.config.output_suffix
		candidate = f"{base_stem}{suffix}"
		if self.config.skip_existing:
			image_path = images_dir / f"{candidate}{self.config.image_ext}"
			if not image_path.exists():
				return candidate
		index = 1
		while True:
			candidate_idx = f"{candidate}_{index:03d}"
			image_path = images_dir / f"{candidate_idx}{self.config.image_ext}"
			if not image_path.exists():
				return candidate_idx
			index += 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Stable Diffusion + SAM augmentation pipeline")
	dataset_group = parser.add_mutually_exclusive_group(required=True)
	dataset_group.add_argument(
		"--dataset-manifest",
		type=Path,
		help="JSON file with a list of {image_path, mask_path} entries",
	)
	dataset_group.add_argument(
		"--image-dir",
		type=Path,
		help="Directory containing the real microscope images (paired by filename)",
	)
	parser.add_argument(
		"--mask-dir",
		type=Path,
		help="Directory containing masks (required when using --image-dir)",
	)
	parser.add_argument("--output-dir", type=Path, required=True, help="Where augmented data goes")
	parser.add_argument("--image-root", type=Path, help="Root to prepend to manifest image paths")
	parser.add_argument("--mask-root", type=Path, help="Root to prepend to manifest mask paths")

	parser.add_argument("--sam-checkpoint", type=Path, required=True, help="Path to SAM checkpoint")
	parser.add_argument(
		"--sam-model-type",
		default="vit_h",
		choices=["default", "vit_h", "vit_l", "vit_b"],
		help="SAM backbone variant",
	)
	parser.add_argument("--sd-model-id", default="runwayml/stable-diffusion-v1-5")
	parser.add_argument("--prompt", default=AugmentationConfig.prompt)
	parser.add_argument("--negative-prompt", default=None)
	parser.add_argument("--strength", type=float, default=0.35)
	parser.add_argument("--guidance-scale", type=float, default=7.5)
	parser.add_argument("--num-inference-steps", type=int, default=50)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--bbox-padding", type=int, default=8)
	parser.add_argument("--mask-threshold", type=int, default=0)
	parser.add_argument("--max-samples", type=int)
	parser.add_argument("--output-suffix", default="_aug")
	parser.add_argument("--image-ext", default=".png")
	parser.add_argument("--mask-ext", default=".png")
	parser.add_argument("--device", default=None, help="Override device (cuda or cpu)")
	parser.add_argument("--keep-safety-checker", action="store_true")
	parser.add_argument("--skip-existing", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
	args = parse_args(argv)
	_configure_logging(args.verbose)

	if args.image_dir and not args.mask_dir:
		raise ValueError("--mask-dir is required when using --image-dir mode")

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
		mask_threshold=args.mask_threshold,
		output_suffix=args.output_suffix,
		image_ext=args.image_ext,
		mask_ext=args.mask_ext,
		max_samples=args.max_samples,
		skip_existing=args.skip_existing,
		device=device,
		disable_safety_checker=not args.keep_safety_checker,
	)

	if args.dataset_manifest:
		dataset = load_dataset_from_manifest(
			args.dataset_manifest,
			image_root=args.image_root,
			mask_root=args.mask_root,
		)
	else:
		dataset = auto_discover_dataset(args.image_dir, args.mask_dir)

	if not dataset:
		LOGGER.error("No dataset entries found; exiting")
		sys.exit(1)

	augmentor = DiffusionAugmentor(config)
	metadata = augmentor.augment_dataset(dataset, args.output_dir)

	metadata_path = args.output_dir / "metadata.json"
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	with open(metadata_path, "w", encoding="utf-8") as handle:
		json.dump(metadata, handle, indent=2)
	LOGGER.info("Wrote metadata for %d augmented samples to %s", len(metadata), metadata_path)


if __name__ == "__main__":
	main()
