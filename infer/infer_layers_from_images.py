from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from coco_writer import (  # noqa: E402
    CATEGORY_NAME_BY_ID,
    LAYER_CATEGORIES,
    LAYER_CATEGORY_ID,
    SEGMENTATION_CATEGORIES,
    annotation_record,
    image_record,
    write_coco,
    write_csv,
)
from layer_runner import DEFAULT_LAYER_MODEL_DIR, LayerRecognitionRunner  # noqa: E402
from segmentation_runner import (  # noqa: E402
    DEFAULT_SEG_WEIGHTS,
    SegmentationRunner,
    build_postprocess_params,
    expand_path,
    flake_summary,
    postprocess_params_dict,
    resolve_seg_config,
)


IMAGE_FIELDNAMES = [
    "image",
    "image_id",
    "segmentation_ann_id",
    "flake_index",
    "status",
    "category_id",
    "category_name",
    "pred_layer",
    "raw_pred_layer",
    "layer_source",
    "layer_score",
    "review",
    "wb1",
    "wb2",
    "ndg",
    "iqr_ndg",
    "delta_g",
    "g_flake_rep",
    "g_bg_plane_median",
    "flake_rep_source",
    "roi_bg_pixels",
    "seg_score",
    "area_px",
    "area_um2",
    "center_x",
    "center_y",
    "max_sidelength_px",
    "min_sidelength_px",
    "shape_complexity",
]

FAILURE_FIELDNAMES = [
    "image",
    "image_id",
    "segmentation_ann_id",
    "flake_index",
    "status",
    "reason",
    "wb1",
    "wb2",
    "seg_score",
    "area_px",
    "area_um2",
    "center_x",
    "center_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer graphene flake masks and layer labels from an image folder.")
    parser.add_argument("--image-dir", required=True, help="Folder containing images to infer.")
    parser.add_argument("--out-dir", default=None, help="Output folder. Defaults to --image-dir.")
    parser.add_argument("--seg-config", default=None, help="M2F config.yaml. Defaults to config.yaml next to --seg-weights.")
    parser.add_argument("--seg-weights", default=DEFAULT_SEG_WEIGHTS, help="M2F checkpoint path.")
    parser.add_argument("--layer-model-dir", default=DEFAULT_LAYER_MODEL_DIR, help="Green ordinal layer model directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="Comma-separated image suffixes.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--size-threshold-px", type=int, default=0)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--postprocess-vis-dir", default=None)

    parser.add_argument("--pp-overlap-iou-threshold", type=float, default=0.5)
    parser.add_argument("--pp-overlap-containment-threshold", type=float, default=0.8)
    parser.add_argument("--pp-enable-bridge-merge", action="store_true")
    parser.add_argument("--pp-grow-radius-px", type=int, default=3)
    parser.add_argument("--pp-max-boundary-distance-px", type=int, default=3)
    parser.add_argument("--pp-lab-l-weight", type=float, default=0.5)
    parser.add_argument("--pp-tau-grow", type=float, default=12.0)
    parser.add_argument("--pp-tau-pair", type=float, default=12.0)
    parser.add_argument("--pp-grow-sigma", type=float, default=1.5)
    parser.add_argument("--pp-min-bridge-area-px", type=float, default=15.0)
    parser.add_argument("--pp-min-bridge-ratio", type=float, default=0.3)
    parser.add_argument("--pp-final-min-area-um2", type=float, default=100.0)
    parser.add_argument("--pp-final-min-score", type=float, default=0.015)
    parser.add_argument("--pp-final-max-shape-complexity", type=float, default=5.0)
    parser.add_argument("--pp-max-bridge-passes", type=int, default=5)
    return parser.parse_args()


def list_images(image_dir: Path, extensions_text: str, max_images: int | None) -> list[Path]:
    extensions = {item.strip().lower() for item in extensions_text.split(",") if item.strip()}
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    if max_images is not None:
        images = images[: max(0, int(max_images))]
    return images


def make_info(args: argparse.Namespace, seg_config: Path, layer_runner: LayerRecognitionRunner, postprocess_params) -> dict:
    return {
        "description": "MaskTerial segmentation followed by green ordinal layer recognition",
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": "infer/infer_layers_from_images.py",
        "script_version": 1,
        "image_dir": str(expand_path(args.image_dir)),
        "segmentation_config": str(seg_config),
        "segmentation_weights": str(expand_path(args.seg_weights)),
        "layer_model_dir": str(expand_path(args.layer_model_dir)),
        "layer_thresholds": layer_runner.threshold_info(),
        "postprocess_enabled": not args.no_postprocess,
        "postprocess_params": postprocess_params_dict(postprocess_params),
    }


def merge_rows(common: dict, layer_row: dict, summary: dict) -> dict:
    row = dict(common)
    row.update(layer_row)
    row.update(summary)
    return row


def main() -> None:
    args = parse_args()
    image_dir = expand_path(args.image_dir)
    out_dir = expand_path(args.out_dir or args.image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_weights = expand_path(args.seg_weights)
    seg_config = resolve_seg_config(seg_weights, args.seg_config)
    postprocess_params = build_postprocess_params(args)

    print(f"[INFO] Loading segmentation model: {seg_weights}")
    print(f"[INFO] Segmentation config: {seg_config}")
    seg_runner = SegmentationRunner(
        config_path=seg_config,
        weights_path=seg_weights,
        device=args.device,
        score_threshold=args.score_threshold,
        size_threshold_px=args.size_threshold_px,
        postprocess_params=postprocess_params,
        use_postprocess=not args.no_postprocess,
    )

    print(f"[INFO] Loading layer model: {args.layer_model_dir}")
    layer_runner = LayerRecognitionRunner(args.layer_model_dir)
    info = make_info(args, seg_config, layer_runner, postprocess_params)

    image_paths = list_images(image_dir, args.extensions, args.max_images)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"[INFO] Found {len(image_paths)} images")

    images = []
    segmentation_annotations = []
    layer_annotations = []
    prediction_rows = []
    background_rows = []
    failure_rows = []
    next_ann_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Skipping unreadable image: {image_path}")
            continue

        height, width = image_bgr.shape[:2]
        file_name = image_path.name
        images.append(image_record(image_id, file_name, width, height))

        print(f"[INFO] Processing {image_id}/{len(image_paths)}: {file_name}")
        debug_vis_dir = None
        if args.postprocess_vis_dir:
            debug_vis_dir = expand_path(args.postprocess_vis_dir)
        seg_result = seg_runner.run_image(image_bgr, image_path.stem, debug_vis_dir=debug_vis_dir)
        layer_results = layer_runner.predict_image(
            image_bgr=image_bgr,
            filename=file_name,
            image_path=str(image_path),
            image_id=image_id,
            flakes=seg_result.final_flakes,
        )

        for flake_index, flake in enumerate(seg_result.final_flakes, start=1):
            seg_ann_id = next_ann_id
            next_ann_id += 1
            summary = flake_summary(flake)
            common = {
                "image": file_name,
                "image_id": image_id,
                "segmentation_ann_id": seg_ann_id,
                "flake_index": flake_index,
            }

            segmentation_annotations.append(
                annotation_record(
                    annotation_id=seg_ann_id,
                    image_id=image_id,
                    mask=flake.mask,
                    category_id=1,
                    extra={
                        "seg_score": summary["seg_score"],
                        "area_um2": summary["area_um2"],
                        "center": [summary["center_x"], summary["center_y"]],
                        "flake_index": flake_index,
                    },
                )
            )

            if flake_index > len(layer_results):
                row = {
                    **common,
                    "status": "feature_failed",
                    "reason": "missing layer result for flake",
                    "wb1": "",
                    "wb2": "",
                }
                failure_rows.append(merge_rows(row, {}, summary))
                continue

            layer_result = layer_results[flake_index - 1]
            layer_row = layer_result.csv_row()

            if layer_result.status == "layer":
                category_id = LAYER_CATEGORY_ID[str(layer_result.category_key)]
                category_name = CATEGORY_NAME_BY_ID[category_id]
                accepted_row = {
                    **common,
                    "status": "layer",
                    "category_id": category_id,
                    "category_name": category_name,
                }
                prediction_rows.append(merge_rows(accepted_row, layer_row, summary))
                layer_annotations.append(
                    annotation_record(
                        annotation_id=seg_ann_id,
                        image_id=image_id,
                        mask=flake.mask,
                        category_id=category_id,
                        extra={
                            "segmentation_annotation_id": seg_ann_id,
                            "seg_score": summary["seg_score"],
                            "layer_score": layer_result.layer_score,
                            "pred_layer": layer_result.final_label,
                            "raw_pred_layer": layer_result.raw_pred_layer,
                            "layer_source": layer_result.feature_values.get("layer_source", ""),
                            "review": layer_result.review,
                            "ndg": layer_result.feature_values.get("ndg", ""),
                            "iqr_ndg": layer_result.feature_values.get("iqr_ndg", ""),
                            "wb1": layer_result.wb1,
                            "wb2": layer_result.wb2,
                            "flake_rep_source": layer_result.feature_values.get("flake_rep_source", ""),
                        },
                    )
                )
            elif layer_result.status == "background":
                dropped_row = {
                    **common,
                    "status": "background",
                    "category_id": "",
                    "category_name": "background_dropped",
                    "background_score_threshold": layer_runner.background_score_threshold,
                }
                background_rows.append(merge_rows(dropped_row, layer_row, summary))
            else:
                failed_row = {
                    **common,
                    "status": "feature_failed",
                    "reason": layer_result.reason,
                }
                failure_rows.append(merge_rows(failed_row, layer_row, summary))

        print(
            f"       raw={len(seg_result.raw_flakes)} final={len(seg_result.final_flakes)} "
            f"layered={sum(1 for item in layer_results if item.status == 'layer')} "
            f"background={sum(1 for item in layer_results if item.status == 'background')} "
            f"failed={sum(1 for item in layer_results if item.status == 'feature_failed')}"
        )

    segmentation_coco_path = out_dir / "_predicted_segmentation.coco.json"
    layer_coco_path = out_dir / "_predicted_layers.coco.json"
    write_coco(segmentation_coco_path, images, segmentation_annotations, SEGMENTATION_CATEGORIES, info)
    write_coco(layer_coco_path, images, layer_annotations, LAYER_CATEGORIES, info)
    write_csv(out_dir / "_layer_predictions.csv", prediction_rows, IMAGE_FIELDNAMES)
    write_csv(out_dir / "_background_dropped.csv", background_rows, IMAGE_FIELDNAMES)
    write_csv(out_dir / "_layer_failures.csv", failure_rows, FAILURE_FIELDNAMES)

    print(f"[INFO] Wrote segmentation COCO: {segmentation_coco_path}")
    print(f"[INFO] Wrote layer COCO: {layer_coco_path}")
    print(f"[INFO] Layer annotations: {len(layer_annotations)}")
    print(f"[INFO] Background dropped: {len(background_rows)}")
    print(f"[INFO] Feature failures: {len(failure_rows)}")


if __name__ == "__main__":
    main()
