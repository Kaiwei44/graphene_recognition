import json
import os

from detectron2.data.datasets import register_coco_instances

from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models.M2F import maskformer_model  # noqa: F401
from maskterial.utils.argparser import parse_eval_args
from maskterial.utils.evaluator import evaluate_on_dataset
from maskterial.utils.loader_functions import load_models


def main():
    args = parse_eval_args()

    if not os.path.exists(args.annotation_path):
        raise ValueError(f"Annotation path {args.annotation_path} does not exist.")
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory {args.image_dir} does not exist.")

    register_coco_instances(
        args.dataset_name,
        {},
        args.annotation_path,
        args.image_dir,
    )

    seg_model, cls_model, pp_model = load_models(**vars(args))

    maskterial = MaskTerial(
        segmentation_model=seg_model,
        classification_model=cls_model,
        postprocessing_model=pp_model,
        score_threshold=0.0,
        min_class_occupancy=0.0,
        size_threshold=200,
        device=args.device,
    )

    results = evaluate_on_dataset(
        model=maskterial,
        dataset_name=args.dataset_name,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
