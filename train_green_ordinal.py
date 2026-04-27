from __future__ import annotations

import argparse
import os
from pathlib import Path

from layer_recognition.green_ordinal import (
    DEFAULT_FEATURE_COLS,
    ExtractConfig,
    class_counts,
    coerce_rows,
    expand_path,
    extract_features,
    read_csv,
    rows_to_dicts,
    run_cv,
    run_final,
    run_roboflow_test,
    select_feature_cols,
    split_counts,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train robust green-channel ordinal calibration model for graphene layers 1/2/3/4/6/7/8.")
    parser.add_argument("--coco-dir", type=str, default=None, help="Dataset root containing train/test/_annotations.coco.json")
    parser.add_argument("--train-image-dir", type=str, default=None)
    parser.add_argument("--train-coco", type=str, default=None)
    parser.add_argument("--valid-image-dir", type=str, default=None)
    parser.add_argument("--valid-coco", type=str, default=None)
    parser.add_argument("--test-image-dir", type=str, default=None)
    parser.add_argument("--test-coco", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["cv", "roboflow_test", "final"], default="roboflow_test")
    parser.add_argument("--feature-cache", type=str, default=None)
    parser.add_argument("--force-extract", action="store_true")

    parser.add_argument("--regressor", choices=["huber", "ridge"], default="huber")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--huber-alpha", type=float, default=0.0001)
    parser.add_argument("--huber-epsilon", type=float, default=1.35)
    parser.add_argument("--boundary-margin", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--roi-scale", type=float, default=3.0)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--bg-min-pixels", type=int, default=100)
    parser.add_argument("--edge-dilate-px", type=int, default=5)
    parser.add_argument("--bg-sample-max", type=int, default=20000)
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURE_COLS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cache = expand_path(args.feature_cache) if args.feature_cache else out_dir / "features.csv"
    if feature_cache.exists() and not args.force_extract:
        print(f"[INFO] Loading cached features: {feature_cache}")
        rows = coerce_rows(read_csv(feature_cache))
    else:
        cfg = ExtractConfig(
            roi_scale=args.roi_scale,
            min_area_px=args.min_area_px,
            bg_min_pixels=args.bg_min_pixels,
            bg_sample_max=args.bg_sample_max,
            edge_dilate_px=args.edge_dilate_px,
        )
        rows = rows_to_dicts(
            extract_features(
                cfg=cfg,
                coco_dir=args.coco_dir,
                train_image_dir=args.train_image_dir,
                train_coco=args.train_coco,
                valid_image_dir=args.valid_image_dir,
                valid_coco=args.valid_coco,
                test_image_dir=args.test_image_dir,
                test_coco=args.test_coco,
            )
        )
        write_csv(rows, feature_cache)
        print(f"[INFO] Saved features: {feature_cache}")

    print("[INFO] Layer counts by split:")
    for split, counts in split_counts(rows).items():
        print(f"  {split}: {counts}")
    if "5" not in class_counts(rows):
        print("[WARN] No layer-5 samples found. Scores in 4.5-5.5 are treated as review/missing-5 zone.")

    requested_features = [name.strip() for name in args.features.split(",") if name.strip()]
    feature_cols = select_feature_cols(rows, requested_features)
    print(f"[INFO] Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}")

    payload = vars(args).copy()
    payload["feature_cols"] = feature_cols
    common = dict(
        rows=rows,
        feature_cols=feature_cols,
        out_dir=out_dir,
    )
    if args.mode == "roboflow_test":
        run_roboflow_test(
            **common,
            regressor=args.regressor,
            degree=args.degree,
            ridge_alpha=args.ridge_alpha,
            huber_alpha=args.huber_alpha,
            huber_epsilon=args.huber_epsilon,
            boundary_margin=args.boundary_margin,
            args_payload=payload,
        )
    elif args.mode == "cv":
        run_cv(
            **common,
            folds=args.folds,
            seed=args.seed,
            regressor=args.regressor,
            degree=args.degree,
            ridge_alpha=args.ridge_alpha,
            huber_alpha=args.huber_alpha,
            huber_epsilon=args.huber_epsilon,
            boundary_margin=args.boundary_margin,
        )
    elif args.mode == "final":
        run_final(
            **common,
            regressor=args.regressor,
            degree=args.degree,
            ridge_alpha=args.ridge_alpha,
            huber_alpha=args.huber_alpha,
            huber_epsilon=args.huber_epsilon,
            boundary_margin=args.boundary_margin,
            args_payload=payload,
        )
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()

