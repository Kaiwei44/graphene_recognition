# Infra/raw v3 non-DL subpart segmentation

This folder contains the current best non-deep-learning v3 pipeline for splitting large raw-image blocks into thickness-consistent subparts by color/depth appearance.

The method is intentionally classical and interpretable:

1. Read COCO segmentation annotations.
2. Select raw images and large block masks, usually category `gra`.
3. For each large block:
   - remove red guide lines / black scale text from statistics,
   - convert RGB to Lab,
   - correct illumination on Lab-L with a large-scale Gaussian background,
   - smooth with a bilateral filter,
   - generate graph superpixels with Felzenszwalb segmentation,
   - assign each superpixel to a corrected-brightness class using a conservative contrast rule and multi-Otsu,
   - clean connected components and small fragments.
4. Save predicted label masks and visual overlays.
5. If manual `subparts` annotations exist, evaluate with ARI, covering score, and boundary F1.

This is the v3 version that was more stable than the later profile-split experiment. It is preferred as the current mainline baseline.

## Folder structure

```text
infra_v3_code/
  README.md
  requirements.txt
  src/
    v3_raw_subparts.py
  examples/
    run_on_uploaded_zip.sh
```

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run directly on a Roboflow/COCO zip:

```bash
python src/v3_raw_subparts.py \
  --input-zip /path/to/infrared.v3-infra-with-subparts-v2.coco-segmentation.zip \
  --out-dir /path/to/out_v3 \
  --raw-prefix raw_png \
  --big-category gra \
  --subpart-category subparts
```

Or run on an extracted COCO directory:

```bash
python src/v3_raw_subparts.py \
  --dataset-dir /path/to/dataset_root \
  --out-dir /path/to/out_v3
```

The script searches for `_annotations.coco.json` automatically under `dataset-dir`.

## Outputs

The output directory contains:

```text
per_block/                         visual panels and per-block npz files
per_image_pred_labels/             predicted integer label masks as PNG
v3_overview.jpg                    visual contact sheet
v3_per_block.csv                   per-block metrics if GT exists
v3_per_image.csv                   per-image aggregate if GT exists
v3_summary.json                    overall summary and run config
```

Each `.npz` in `per_block/` includes:

```text
pred_label        integer label mask for the current block
block_mask        the large-block mask
image_id          COCO image id
gt_label          only present when manual subparts exist
```

## Notes

- This code does not use deep learning.
- It assumes the upstream large-block mask is already available in COCO annotations.
- If no manual `subparts` category is present, the script still runs prediction and visualization, but metrics will be omitted.
- For production, keep this v3 method as the default and apply more aggressive splitters only to blocks detected as complex/under-split.
