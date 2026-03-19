import argparse
import os
import random
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


def draw_area_labels(image, flakes, scale: float):
    image = np.ascontiguousarray(image)
    for flake in flakes:
        text = f"{flake.measurements.area_um2:.1f} um^2"
        center_x = int(flake.center[0] * scale)
        center_y = int(flake.center[1] * scale)
        text_origin = (center_x + 8, center_y - 8)
        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        x0, y0 = text_origin
        x1 = x0 + text_size[0] + 8
        y1 = y0 - text_size[1] - 8
        cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.rectangle(
            image,
            (x0 - 4, y1),
            (x1, y0 + baseline + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def filter_flakes_by_area(flakes, min_area_um2: float):
    return [flake for flake in flakes if flake.measurements.area_um2 >= min_area_um2]


def draw_flake_masks(image, flakes, scale: float):
    output = image.copy()
    overlay = image.copy()
    for flake in flakes:
        mask = flake.mask.astype(np.uint8)
        if scale != 1.0:
            mask = cv2.resize(
                mask,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        overlay[mask.astype(bool)] = (0, 255, 255)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (0, 200, 255), 2, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--outdir", default="./vis_pred")
    parser.add_argument("--dataset-name", default="visualize_prediction_dataset")
    parser.add_argument("--class-name", default="gra")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area-um2", type=float, default=100.0)
    parser.add_argument("--draw-gt", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)

    cfg = build_cfg(args.config_file, args.weights, args.opts)

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
        score_threshold=0.0,
        min_class_occupancy=0.0,
        size_threshold=0,
        device=torch.device(cfg.MODEL.DEVICE),
    )
    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    if args.num_samples == -1:
        samples = dataset_dicts
    else:
        samples = random.sample(dataset_dicts, min(args.num_samples, len(dataset_dicts)))

    for dataset_dict in samples:
        img = cv2.imread(dataset_dict["file_name"])
        if img is None:
            print(f"Skipping unreadable image: {dataset_dict['file_name']}")
            continue

        flakes = maskterial.predict(img)
        flakes = filter_flakes_by_area(flakes, args.min_area_um2)
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
        pred_img = draw_area_labels(pred_img, flakes, args.scale)

        base = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
        print(f"{base}: kept {len(flakes)} flakes >= {args.min_area_um2} um^2")
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

    print(f"All done. -> {args.outdir}")


if __name__ == "__main__":
    main()
