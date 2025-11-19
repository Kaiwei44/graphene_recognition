import os
import sys
from functools import wraps
from typing import Literal

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from fastapi import HTTPException, UploadFile

from ..modeling.classification_models import AMM_head, GMM_head, MLP_head
from ..maskterial import MaskTerial
from ..modeling.postprocessing_models import L2_model
from ..modeling.segmentation_models import M2F_model, MRCNN_model


def convert_coco_polygon_to_rle(coco_file: dict) -> dict:
    coco = coco_file.copy()

    imgs = coco["images"]
    anns = coco["annotations"]
    for idx, ann in enumerate(anns):

        # skipping annotations which are already in RLE format
        if type(ann["segmentation"]) is dict:
            continue

        # find the image for this annotation
        img = [img for img in imgs if img["id"] == ann["image_id"]]
        if len(img) == 0:
            print("Image not found for annotation", ann)
            continue
        img = img[0]

        rle = mask_util.frPyObjects(ann["segmentation"], img["height"], img["width"])[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        anns[idx]["segmentation"] = rle

    return coco


def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error: {e}")

    return wrapper


class ServerState:
    def __init__(
        self,
        seg_model_name: Literal["M2F", "MRCNN"] | None = None,
        cls_model_name: Literal["AMM", "GMM", "MLP"] | None = None,
        pp_model_name: Literal["L2"] | None = None,
        seg_model_dir: str = "./models/seg_models",
        cls_model_dir: str = "./models/cls_models",
        pp_model_dir: str = "./models/pp_models",
        score_threshold: float = 0.0,
        min_class_occupancy: float = 0.0,
        size_threshold: int = 300,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        (
            score_threshold,
            min_class_occupancy,
            size_threshold,
        ) = self._validate_inputs(
            score_threshold,
            min_class_occupancy,
            size_threshold,
        )

        self.seg_model_name = seg_model_name
        self.cls_model_name = cls_model_name
        self.pp_model_name = pp_model_name
        self.seg_model_dir = seg_model_dir
        self.cls_model_dir = cls_model_dir
        self.pp_model_dir = pp_model_dir
        self.device = device
        self.score_threshold = score_threshold
        self.min_class_occupancy = min_class_occupancy
        self.size_threshold = size_threshold

    @handle_exceptions
    def _validate_inputs(
        self,
        score_threshold: float,
        min_class_occupancy: float,
        size_threshold: int,
    ):

        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")

        if not 0 <= min_class_occupancy <= 1:
            raise ValueError("min_class_occupancy must be between 0 and 1")

        if size_threshold < 0:
            raise ValueError("size_threshold must be a positive integer")

        return score_threshold, min_class_occupancy, size_threshold

    def __eq__(self, other):
        if not isinstance(other, ServerState):
            return False

        return all(
            [
                self.seg_model_name == other.seg_model_name,
                self.cls_model_name == other.cls_model_name,
                self.pp_model_name == other.pp_model_name,
                self.seg_model_dir == other.seg_model_dir,
                self.cls_model_dir == other.cls_model_dir,
                self.pp_model_dir == other.pp_model_dir,
                self.score_threshold == other.score_threshold,
                self.min_class_occupancy == other.min_class_occupancy,
                self.size_threshold == other.size_threshold,
                self.device == other.device,
            ]
        )

    def __repr__(self):
        return f"ServerState(segmentation_model={self.seg_model_name}, classification_model={self.cls_model_name}, postprocessing_model={self.pp_model_name}, score_threshold={self.score_threshold}, min_class_occupancy={self.min_class_occupancy}, size_threshold={self.size_threshold})"

    def __str__(self):
        return f"ServerState(segmentation_model={self.seg_model_name}, classification_model={self.cls_model_name}, postprocessing_model={self.pp_model_name}, score_threshold={self.score_threshold}, min_class_occupancy={self.min_class_occupancy}, size_threshold={self.size_threshold})"

    def __dict__(self):
        return {
            "segmentation_model": self.seg_model_name,
            "classification_model": self.cls_model_name,
            "postprocessing_model": self.pp_model_name,
            "score_threshold": self.score_threshold,
            "min_class_occupancy": self.min_class_occupancy,
            "size_threshold": self.size_threshold,
            "device": self.device,
        }

    def to_dict(self):
        return self.__dict__()


@handle_exceptions
def read_image(file: UploadFile):
    try:
        img = cv2.imdecode(
            np.frombuffer(
                file.file.read(),
                np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
    except Exception:
        raise ValueError(f"Error reading image file: {sys.exc_info()[0]}")
    finally:
        file.file.close()

    return img


@handle_exceptions
def load_models(
    cls_model_name: str | None,
    cls_model_path: str | None,
    seg_model_name: str | None,
    seg_model_path: str | None,
    pp_model_name: str | None,
    pp_model_path: str | None,
    device: torch.device,
) -> tuple:

    seg_model = None
    cls_model = None
    pp_model = None

    if cls_model_name is not None:
        cls_model_type = cls_model_name.split("-")[0]

        if cls_model_type == "AMM":
            cls_model = AMM_head
        elif cls_model_type == "GMM":
            cls_model = GMM_head
        elif cls_model_type == "MLP":
            cls_model = MLP_head
        else:
            raise ValueError(
                f"Unknown cls model type: {cls_model_type}, Supported models are: AMM, GMM, MLP"
            )

        cls_model = cls_model.from_pretrained(cls_model_path, device=device)

    if seg_model_name is not None:
        seg_model_type = seg_model_name.split("-")[0]

        if seg_model_type == "M2F":
            seg_model = M2F_model
        elif seg_model_type == "MRCNN":
            seg_model = MRCNN_model
        else:
            raise ValueError(
                f"Unknown seg model type: {seg_model_type}, Supported models are: M2F, MaskRCNN"
            )

        seg_model = seg_model.from_pretrained(seg_model_path, device=device)

    if pp_model_name is not None:
        pp_model_type = pp_model_name.split("-")[0]

        if pp_model_type == "L2":
            pp_model = L2_model
        else:
            raise ValueError(
                f"Unknown pp model type: {pp_model_type}, Supported models are: L2"
            )

        pp_model = pp_model.from_pretrained(pp_model_path, device=device)

    return seg_model, cls_model, pp_model


def check_available_models(model_dir: str) -> list[str]:
    all_models = {}

    if os.path.exists(model_dir) is False:
        return all_models

    model_types = os.listdir(model_dir)
    for model_type in model_types:
        all_models[model_type] = []
        model_type_dir = os.path.join(model_dir, model_type)
        model_names = os.listdir(model_type_dir)
        for model_name in model_names:
            model_path = os.path.join(model_type_dir, model_name)
            if os.path.isdir(model_path):
                all_models[model_type].append(model_name)

    return all_models


def update_server_state_and_predictor(server_state: ServerState):
    def construct_model_paths(server_state: ServerState):
        seg_model_name = server_state.seg_model_name
        cls_model_name = server_state.cls_model_name
        pp_model_name = server_state.pp_model_name
        seg_model_dir = server_state.seg_model_dir
        cls_model_dir = server_state.cls_model_dir
        pp_model_dir = server_state.pp_model_dir

        seg_model_path = None
        cls_model_path = None
        pp_model_path = None
        if seg_model_name is not None:
            seg_model_type = seg_model_name.split("-")[0]
            seg_model_name = "-".join(seg_model_name.split("-")[1:])
            seg_model_path = os.path.join(seg_model_dir, seg_model_type, seg_model_name)
        if cls_model_name is not None:
            cls_model_type = cls_model_name.split("-")[0]
            cls_model_name = "-".join(cls_model_name.split("-")[1:])
            cls_model_path = os.path.join(cls_model_dir, cls_model_type, cls_model_name)
        if pp_model_name is not None:
            pp_model_type = pp_model_name.split("-")[0]
            pp_model_name = "-".join(pp_model_name.split("-")[1:])
            pp_model_path = os.path.join(pp_model_dir, pp_model_type, pp_model_name)
        return seg_model_path, cls_model_path, pp_model_path

    # make the code more readable
    # I could just have them as long arguments in the function call
    # but I like this better
    seg_model_name = server_state.seg_model_name
    cls_model_name = server_state.cls_model_name
    pp_model_name = server_state.pp_model_name
    device = server_state.device

    # We dont need to reload the models if the params change
    score_threshold = server_state.score_threshold
    min_class_occupancy = server_state.min_class_occupancy
    size_threshold = server_state.size_threshold

    (
        seg_model_path,
        cls_model_path,
        pp_model_path,
    ) = construct_model_paths(server_state)

    (
        seg_model,
        cls_model,
        pp_model,
    ) = load_models(
        cls_model_name,
        cls_model_path,
        seg_model_name,
        seg_model_path,
        pp_model_name,
        pp_model_path,
        device,
    )

    predictor = MaskTerial(
        segmentation_model=seg_model,
        classification_model=cls_model,
        postprocessing_model=pp_model,
        score_threshold=score_threshold,
        min_class_occupancy=min_class_occupancy,
        size_threshold=size_threshold,
        device=device,
    )

    return server_state, predictor
