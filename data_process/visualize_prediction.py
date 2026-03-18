import argparse
import os
import random
from types import SimpleNamespace

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

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

        outputs = predictor(img)

        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=meta,
            scale=args.scale,
            instance_mode=ColorMode.IMAGE,
        )
        pred_img = visualizer.draw_instance_predictions(
            outputs["instances"].to("cpu")
        ).get_image()[:, :, ::-1]

        base = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
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
