from __future__ import annotations

import argparse

from layer_recognition.green_ordinal import (
    COMPACT_BASE_FEATURES,
    COMPACT_PHI_FEATURES,
    ExtractConfig,
    class_counts,
    coerce_rows,
    expand_path,
    extract_features,
    parse_alpha_grid,
    read_csv,
    rows_to_dicts,
    run_cv,
    run_final,
    run_roboflow_test,
    split_counts,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Compact Green Ordinal Ridge model for graphene layers 1/2/3/4/6/7/8."
    )

    parser.add_argument("--coco-dir", type=str, default=None, help="Dataset root containing train/test/_annotations.coco.json")
    parser.add_argument("--train-image-dir", type=str, default=None)
    parser.add_argument("--train-coco", type=str, default=None)
    parser.add_argument("--valid-image-dir", type=str, default=None)
    parser.add_argument("--valid-coco", type=str, default=None)
    parser.add_argument("--test-image-dir", type=str, default=None)
    parser.add_argument("--test-coco", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["cv", "roboflow_test", "final", "extract_only"], default="roboflow_test")
    parser.add_argument("--feature-cache", type=str, default=None)
    parser.add_argument("--force-extract", action="store_true")

    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument("--tune", dest="tune", action="store_true", help="Tune ridge alpha on the train split only")
    tune_group.add_argument("--no-tune", dest="tune", action="store_false", help="Use --alpha directly")
    parser.set_defaults(tune=True)

    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--alpha-grid", type=str, default="0.3,1,3,10,30,100")
    parser.add_argument("--boundary-margin", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--roi-scale", type=float, default=3.0)
    parser.add_argument("--min-area-px", type=int, default=20)
    parser.add_argument("--bg-min-pixels", type=int, default=100)
    parser.add_argument("--edge-dilate-px", type=int, default=5)
    parser.add_argument("--bg-sample-max", type=int, default=20000)
    parser.add_argument("--trim-low", type=float, default=10.0)
    parser.add_argument("--trim-high", type=float, default=90.0)

    # Compatibility with the old train script. Accepted but ignored.
    parser.add_argument("--regressor", default=None, help="Ignored. This version always uses CompactOrdinalRidge.")
    parser.add_argument("--degree", type=int, default=None, help="Ignored. This version uses a fixed 7-term map.")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Compatibility alias for --alpha.")
    parser.add_argument("--huber-alpha", type=float, default=None, help="Ignored.")
    parser.add_argument("--huber-epsilon", type=float, default=None, help="Ignored.")
    parser.add_argument("--features", type=str, default=None, help="Ignored. Feature set is fixed to Version B.")

    args = parser.parse_args()
    if args.ridge_alpha is not None:
        args.alpha = float(args.ridge_alpha)
    return args


def main() -> None:
    args = parse_args()
    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regressor is not None:
        print(f"[INFO] Ignoring --regressor {args.regressor!r}; using CompactOrdinalRidge.")
    if args.degree is not None:
        print(f"[INFO] Ignoring --degree {args.degree!r}; using fixed 7-term feature map.")
    if args.features is not None:
        print("[INFO] Ignoring --features; using fixed Version B features: ndg, iqr_ndg, wb1, wb2.")

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
            trim_low=args.trim_low,
            trim_high=args.trim_high,
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
        rows = coerce_rows(rows)
        write_csv(rows, feature_cache)
        print(f"[INFO] Saved features: {feature_cache}")

    rows = [row for row in rows if int(row["layer"]) in set([1, 2, 3, 4, 6, 7, 8])]
    if not rows:
        raise RuntimeError("No usable rows after filtering layers 1/2/3/4/6/7/8")

    print("[INFO] Layer counts by split:")
    for split, counts in split_counts(rows).items():
        print(f"  {split}: {counts}")
    if "5" not in class_counts(rows):
        print("[WARN] No layer-5 samples found. Scores in 4.5-5.5 are treated as review/missing-5 zone.")

    print(f"[INFO] Model base features ({len(COMPACT_BASE_FEATURES)}): {', '.join(COMPACT_BASE_FEATURES)}")
    print(f"[INFO] Model phi features ({len(COMPACT_PHI_FEATURES)}): {', '.join(COMPACT_PHI_FEATURES)}")

    payload = vars(args).copy()
    payload["feature_cache"] = str(feature_cache)
    payload["model_base_features"] = COMPACT_BASE_FEATURES
    payload["model_phi_features"] = COMPACT_PHI_FEATURES

    if args.mode == "extract_only":
        print("[INFO] extract_only complete.")
        return

    alpha_grid = parse_alpha_grid(args.alpha_grid)
    common = dict(
        rows=rows,
        out_dir=out_dir,
        tune=args.tune,
        alpha=args.alpha,
        alpha_grid=alpha_grid,
        folds=args.folds,
        seed=args.seed,
        boundary_margin=args.boundary_margin,
    )

    if args.mode == "roboflow_test":
        run_roboflow_test(**common, args_payload=payload)
    elif args.mode == "cv":
        run_cv(**common)
    elif args.mode == "final":
        run_final(**common, args_payload=payload)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
