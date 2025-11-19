import json
import os

from detectron2.data.datasets import register_coco_instances

from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models.M2F import maskformer_model  # noqa: F401
from maskterial.utils.argparser import parse_eval_args
from maskterial.utils.evaluator import evaluate_on_dataset
from maskterial.utils.loader_functions import load_models

if __name__ == "__main__":
    args = parse_eval_args()
    OUT_DIR = args.output_dir
    DATASET_NAME = args.dataset_name
    ANN_PATH = args.annotation_path
    IMAGE_DIR = args.image_dir
    DEVICE = args.device

    if not os.path.exists(ANN_PATH):
        raise ValueError(f"Annotation path {ANN_PATH} does not exist.")
    if not os.path.exists(IMAGE_DIR):
        raise ValueError(f"Image directory {IMAGE_DIR} does not exist.")

    register_coco_instances(DATASET_NAME, {}, ANN_PATH, IMAGE_DIR)

    seg_model, cls_model, pp_model = load_models(args)

    print("Loading Model...")
    maskterial = MaskTerial(
        segmentation_model=seg_model,
        classification_model=cls_model,
        postprocessing_model=pp_model,
        score_threshold=0.0,
        min_class_occupancy=0.0,
        size_threshold=200,
        device=DEVICE,
    )

    results = evaluate_on_dataset(
        model=maskterial,
        dataset_name=DATASET_NAME,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
