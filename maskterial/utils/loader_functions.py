import json
import os

import cv2
import numpy as np

from ..modeling.classification_models import AMM_head, GMM_head, MLP_head
from ..modeling.postprocessing_models import L2_model
from ..modeling.segmentation_models import M2F_model, MRCNN_model


def load_and_partition_data(data_path):
    mask_paths = {}
    annotations = {}

    # Loading the Data
    solidities = {"train": [], "test": []}
    arcareas = {"train": [], "test": []}
    labels = {"train": [], "test": []}

    # fmt: off
    for split in ["train", "test"]:
        print(f"Processing {split} split")
        annotations[split] = json.load(
            open(os.path.join(data_path, split, "annotations.json"), "r")
        )
        mask_paths[split] = [
            os.path.join(data_path, split, "masks", image_name + ".png")
            for image_name in annotations[split].keys()
        ]
        labels[split] = [
            annotations[split][image_name] for image_name in annotations[split].keys()
        ]
        for idx, mask_path in enumerate(mask_paths[split]):
            print(f"{idx}/{len(mask_paths[split])}", end="\r")

            mask = cv2.imread(mask_path, 0)

            contours, _ = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]

            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            arclength = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            solidity = float(area) / convex_hull_area

            solidities[split].append(solidity)
            arcareas[split].append(arclength / area**0.5)
    # fmt: on

    X_train = np.array([arcareas["train"], solidities["train"]]).T
    y_train = np.array(labels["train"])

    X_test = np.array([arcareas["test"], solidities["test"]]).T
    y_test = np.array(labels["test"])

    return X_train, y_train, X_test, y_test


def load_models(
    cls_model_type: str = None,
    cls_model_root: str = None,
    seg_model_type: str = None,
    seg_model_root: str = None,
    pp_model_type: str = None,
    pp_model_root: str = None,
    device: str = "cuda",
    **kwargs,
) -> tuple:
    seg_model = None
    cls_model = None
    pp_model = None

    if cls_model_type is not None:
        if cls_model_type == "AMM":
            cls_model = AMM_head
        elif cls_model_type == "GMM":
            cls_model = GMM_head
        elif cls_model_type == "MLP":
            cls_model = MLP_head
        else:
            raise ValueError("Unknown cls model type")

        cls_model = cls_model.from_pretrained(cls_model_root, device=device)

    if seg_model_type is not None:
        if seg_model_type == "M2F":
            seg_model = M2F_model
        elif seg_model_type == "MRCNN":
            seg_model = MRCNN_model
        else:
            raise ValueError("Unknown seg model type")

        seg_model = seg_model.from_pretrained(seg_model_root, device=device)

    if pp_model_type is not None:
        if pp_model_type == "L2":
            pp_model = L2_model
        else:
            raise ValueError("Unknown pp model type")

        pp_model = pp_model.from_pretrained(pp_model_root, device=device)

    if seg_model is None and cls_model is None and pp_model is None:
        raise ValueError("No model is loaded. Please check the model type and root.")

    return seg_model, cls_model, pp_model
