import argparse
import csv
import os
import random
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch

from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models import M2F_model
from maskterial.modeling.segmentation_models.M2F import maskformer_model  # noqa: F401
from maskterial.utils.dataset_functions import setup_config

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from flake_postprocess import GrapheneFlakePostprocessor, PostprocessParams


def build_cfg(config_file: str, weights: str, extra_opts: list[str]):
    args = SimpleNamespace(
        config_file=config_file,
        opts=["MODEL.WEIGHTS", weights, *extra_opts],
        resume=False,
        eval_only=True,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )
    return setup_config(args)


def clamp_detection_count(cfg):
    max_scores = (
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES * cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    )
    if cfg.TEST.DETECTIONS_PER_IMAGE <= max_scores:
        return

    cfg.defrost()
    cfg.TEST.DETECTIONS_PER_IMAGE = max_scores
    cfg.freeze()
    print(f"Clamped TEST.DETECTIONS_PER_IMAGE to {max_scores} available queries")


def flake_score(flake) -> float:
    return 1.0 - float(flake.false_positive_probability)


def flake_area_um2(flake) -> float:
    if flake.measurements is None:
        return 0.0
    return float(flake.measurements.area_um2)


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (0, 128, 255),
        (0, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 0, 255),
        (128, 255, 0),
        (255, 0, 128),
        (128, 0, 255),
    ]
    return palette[(index - 1) % len(palette)]


def format_flake_label(index: int, flake, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    if label_mode == "index":
        return f"#{index}"
    if label_mode == "score":
        return f"#{index} s={flake_score(flake):.2f}"
    if label_mode == "area":
        return f"#{index} a={flake_area_um2(flake):.1f}"
    return f"#{index} s={flake_score(flake):.2f} a={flake_area_um2(flake):.1f}"


def draw_text_box(image, text: str, origin: tuple[int, int], color: tuple[int, int, int]):
    if not text:
        return

    image_h, image_w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = min(max(origin[0], 0), max(0, image_w - text_w - 8))
    y = min(max(origin[1], text_h + 8), max(text_h + 8, image_h - baseline - 4))
    cv2.rectangle(
        image,
        (x - 3, y - text_h - 6),
        (x + text_w + 6, y + baseline + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_flake_labels(image, flakes, scale: float, label_mode: str):
    image = np.ascontiguousarray(image)
    for index, flake in enumerate(flakes, start=1):
        text = format_flake_label(index, flake, label_mode)
        center_x = int(flake.center[0] * scale)
        center_y = int(flake.center[1] * scale)
        draw_text_box(
            image,
            text,
            (center_x + 8, center_y - 8),
            color_for_index(index),
        )
    return image


def filter_flakes_by_area(flakes, min_area_um2: float):
    return [flake for flake in flakes if flake.measurements.area_um2 >= min_area_um2]


def sort_flakes(flakes, sort_by: str):
    if sort_by == "score":
        return sorted(flakes, key=flake_score, reverse=True)
    if sort_by == "area":
        return sorted(flakes, key=flake_area_um2, reverse=True)
    return flakes


def draw_flake_masks(image, flakes, scale: float):
    output = image.copy()
    overlay = image.copy()
    for index, flake in enumerate(flakes, start=1):
        mask = flake.mask.astype(np.uint8)
        if scale != 1.0:
            mask = cv2.resize(
                mask,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        color = color_for_index(index)
        overlay[mask.astype(bool)] = color
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)
    return output


def write_flake_rows(writer, dataset_dict, flakes):
    image_name = os.path.basename(dataset_dict["file_name"])
    for index, flake in enumerate(flakes, start=1):
        writer.writerow(
            {
                "image": image_name,
                "instance_id": index,
                "score": f"{flake_score(flake):.6f}",
                "area_um2": f"{flake_area_um2(flake):.6f}",
                "area_px": int(flake.measurements.area_px),
                "center_x": int(flake.center[0]),
                "center_y": int(flake.center[1]),
                "max_sidelength_px": f"{float(flake.max_sidelength):.3f}",
                "min_sidelength_px": f"{float(flake.min_sidelength):.3f}",
            }
        )


def build_postprocess_params(args) -> PostprocessParams:
    return PostprocessParams(
        overlap_iou_threshold=args.pp_overlap_iou_threshold,
        overlap_containment_threshold=args.pp_overlap_containment_threshold,
        enable_bridge_merge=args.pp_enable_bridge_merge,
        grow_radius_px=args.pp_grow_radius_px,
        max_boundary_distance_px=args.pp_max_boundary_distance_px,
        lab_l_weight=args.pp_lab_l_weight,
        tau_grow=args.pp_tau_grow,
        tau_pair=args.pp_tau_pair,
        grow_sigma=args.pp_grow_sigma,
        min_bridge_area_px=args.pp_min_bridge_area_px,
        min_bridge_ratio=args.pp_min_bridge_ratio,
        final_min_area_um2=args.pp_final_min_area_um2,
        max_bridge_passes=args.pp_max_bridge_passes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--outdir", default="./vis_pred")
    parser.add_argument("--dataset-name", default="visualize_prediction_dataset")
    parser.add_argument("--class-name", default="gra")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--size-threshold-px", type=int, default=0)
    parser.add_argument("--min-area-um2", type=float, default=0.0)
    parser.add_argument(
        "--label-mode",
        choices=["full", "area", "score", "index", "none"],
        default="full",
    )
    parser.add_argument("--sort-by", choices=["score", "area", "none"], default="score")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
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
    parser.add_argument("--pp-max-bridge-passes", type=int, default=5)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.postprocess_vis_dir is not None:
        os.makedirs(args.postprocess_vis_dir, exist_ok=True)
    random.seed(args.seed)

    cfg = build_cfg(args.config_file, args.weights, args.opts)
    clamp_detection_count(cfg)

    register_coco_instances(args.dataset_name, {}, args.ann, args.image_root)
    MetadataCatalog.get(args.dataset_name).set(thing_classes=[args.class_name])
    meta = MetadataCatalog.get(args.dataset_name)

    predictor = DefaultPredictor(cfg)
    seg_model = M2F_model(
        model=predictor.model,
        config=cfg,
        device=torch.device(cfg.MODEL.DEVICE),
    )
    maskterial = MaskTerial(
        segmentation_model=seg_model,
        score_threshold=args.score_threshold,
        min_class_occupancy=0.0,
        size_threshold=args.size_threshold_px,
        device=torch.device(cfg.MODEL.DEVICE),
    )
    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    if args.all_images or args.num_samples == -1:
        samples = dataset_dicts
    else:
        samples = random.sample(dataset_dicts, min(args.num_samples, len(dataset_dicts)))

    use_postprocess = args.postprocess or args.postprocess_vis_dir is not None
    postprocessor = None
    if use_postprocess:
        postprocessor = GrapheneFlakePostprocessor(build_postprocess_params(args))

    csv_path = args.summary_csv or os.path.join(args.outdir, "predictions.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "image",
            "instance_id",
            "score",
            "area_um2",
            "area_px",
            "center_x",
            "center_y",
            "max_sidelength_px",
            "min_sidelength_px",
        ],
    )
    csv_writer.writeheader()

    for dataset_dict in samples:
        img = cv2.imread(dataset_dict["file_name"])
        if img is None:
            print(f"Skipping unreadable image: {dataset_dict['file_name']}")
            continue

        raw_flakes = maskterial.predict(img)
        base = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
        if use_postprocess:
            postprocess_result = postprocessor.run(
                image_bgr=img,
                raw_flakes=raw_flakes,
                debug_vis_dir=args.postprocess_vis_dir,
                image_stem=base,
            )
            flakes = postprocess_result.final_flakes
        else:
            flakes = filter_flakes_by_area(raw_flakes, args.min_area_um2)

        flakes = sort_flakes(flakes, args.sort_by)
        write_flake_rows(csv_writer, dataset_dict, flakes)

        pred_img = img
        if args.scale != 1.0:
            pred_img = cv2.resize(
                pred_img,
                dsize=None,
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_LINEAR,
            )
        pred_img = draw_flake_masks(pred_img, flakes, args.scale)
        pred_img = draw_flake_labels(pred_img, flakes, args.scale, args.label_mode)

        if use_postprocess:
            print(
                f"{base}: raw {len(raw_flakes)}, final {len(flakes)} "
                f"(postprocess area>={args.pp_final_min_area_um2} um^2)"
            )
        else:
            print(
                f"{base}: predicted {len(raw_flakes)}, drew {len(flakes)} "
                f"(score>={args.score_threshold}, size>{args.size_threshold_px}px, "
                f"area>={args.min_area_um2} um^2)"
            )
        pred_path = os.path.join(args.outdir, f"{base}_pred.jpg")
        cv2.imwrite(pred_path, pred_img)

        if args.draw_gt:
            gt_visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=meta,
                scale=args.scale,
                instance_mode=ColorMode.IMAGE,
            )
            gt_img = gt_visualizer.draw_dataset_dict(dataset_dict).get_image()[
                :, :, ::-1
            ]
            cv2.imwrite(os.path.join(args.outdir, f"{base}_gt.jpg"), gt_img)

        print(f"saved: {pred_path}")

    csv_file.close()
    print(f"All done. -> {args.outdir}")
    print(f"Prediction table -> {csv_path}")


if __name__ == "__main__":
    main()
