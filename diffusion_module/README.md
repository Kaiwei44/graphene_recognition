# Diffusion-Augmented Data Generator

This utility turns your lab images + masks into fresh training pairs by running Stable Diffusion (img2img) followed by Segment Anything (SAM) to re-estimate the mask on the synthesized image.

## 1. Requirements

Install the new dependencies (alongside the project requirements):

```bash
pip install -r requirements.txt
```

Download a SAM checkpoint (e.g. `sam_vit_h_4b8939.pth`) and place it somewhere accessible.

## 2. Prepare the dataset description

Create a JSON manifest containing objects with `image_path` and `mask_path`. Paths can be absolute or relative to the optional `--image-root` / `--mask-root` arguments.

```json
[
  {"image_path": "real_data/img_001.jpg", "mask_path": "real_data/mask_001.png"},
  {"image_path": "real_data/img_002.jpg", "mask_path": "real_data/mask_002.png"}
]
```

Alternatively, point the script at `--image-dir` and `--mask-dir` and it will pair files that share the same stem.

### COCO-native workflow

If your training loop already consumes COCO annotations, use the dedicated CLI that augments images **and** appends the new samples directly to your COCO JSON:

```bash
python diffusion_module/coco_augment.py \
  --image-root ~/Data/hBN/test \
  --ann ~/Data/hBN/test/_annotations_1class.coco.json \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sd-model-id runwayml/stable-diffusion-v1-5 \
  --strength 0.35 \
  --guidance-scale 7.5 \
  --num-inference-steps 50
```

What it does:
- Reads every image + annotation straight from the COCO file.
- Runs Stable Diffusion/SAM, writes augmented images back into `--image-root` (override with `--output-image-dir` if desired).
- Appends new `images` / `annotations` entries to a fresh JSON (defaults to `<ann>_aug.coco.json`).
- Records provenance in `<output-ann>.metadata.json` so you can trace the synthetic samples later.

## 3. Run the pipeline

```bash
python diffusion_module/diffusion_model.py \
  --dataset-manifest data/real_dataset.json \
  --image-root /path/to/data \
  --mask-root /path/to/data \
  --output-dir augmented_data \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --prompt "A microscope image of 2D material flakes on a substrate, high resolution, scientific photography." \
  --strength 0.35 \
  --bbox-padding 8 \
  --max-samples 500
```

Key outputs:
- `augmented_data/images/*.png`: Synthesized images
- `augmented_data/masks/*.png`: SAM-predicted masks aligned with each synthesized image
- `augmented_data/metadata.json`: Audit trail linking each new sample to its source pair

### Useful flags
- `--image-dir / --mask-dir`: Auto-discover pairs without a JSON manifest
- `--negative-prompt`: Suppress unwanted artifacts
- `--keep-safety-checker`: Keep the Diffusers safety checker enabled (disabled by default)
- `--skip-existing`: Skip regeneration when an augmented file already exists
- `--device cpu`: Run on CPU (orders of magnitude slower; GPU recommended)

## 4. Tips for best results
- Tune `--strength` to balance fidelity vs. diversity (0.25–0.45 works well for most microscopes)
- Use `--guidance-scale` (6–9) to control how strongly the prompt influences the output
- Consider multiple seeds (via repeated runs with `--seed`) to broaden the generated distribution
