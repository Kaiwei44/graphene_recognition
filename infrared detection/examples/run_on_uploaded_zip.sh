#!/usr/bin/env bash
set -euo pipefail
python src/v3_raw_subparts.py \
  --input-zip /mnt/data/infrared.v3-infra-with-subparts-v2.coco-segmentation.zip \
  --out-dir /mnt/data/v3_run_from_code \
  --raw-prefix raw_png \
  --big-category gra \
  --subpart-category subparts
