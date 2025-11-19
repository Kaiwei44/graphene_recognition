# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
from typing import List, Literal, Union

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, PolygonMasks, polygons_to_bitmask

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["MaskTerialDatasetMapper"]


class MaskTerialDatasetMapper:

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        dataset_name: str,
        image_format: Literal["L", "BGR", "RGB"],
        size_divisibility: int = -1,
        augmentations: List[Union[T.Augmentation, T.Transform]] | None = None,
        instance_mask_format: Literal["polygon", "bitmask"] = "bitmask",
    ):
        """
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
            size_divisibility: pad image size to be divisible by this value
        """
        if augmentations is None:
            augmentations = []

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.instance_mask_format = instance_mask_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.dataset_name = dataset_name

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "dataset_name": cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0],
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }

        return ret

    def _load_image(self, image_name):
        if self.image_format == "L":
            image = cv2.imread(image_name).mean(axis=-1, keepdims=True).astype(np.uint8)
        elif self.image_format == "BGR":
            image = cv2.imread(image_name)
        elif self.image_format == "RGB":
            image = cv2.imread(image_name)[:, :, ::-1]
        else:
            raise ValueError(f"Unknown image format '{self.image_format}'")
        return {"image": image, "image_name": image_name}

    def _transform_annotations(
        self,
        dataset_dict,
        transforms,
        image_shape,
    ):

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(
            annos,
            image_shape,
            mask_format=self.instance_mask_format,
        )

        # update bounding boxes after augmentation
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def _transform_annotations_with_padding(
        self,
        dataset_dict: dict,
        transforms: T.TransformList,
        image_shape: tuple,
    ):
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        target = Instances(image_shape)

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        masks = []
        if len(annos) and "segmentation" in annos[0]:
            segms = [obj["segmentation"] for obj in annos]
            if self.instance_mask_format == "polygon":
                try:
                    masks = PolygonMasks(segms)
                except ValueError as e:
                    raise ValueError(
                        "Failed to use mask_format=='polygon' from the given annotations!"
                    ) from e
            else:
                assert self.instance_mask_format == "bitmask", self.instance_mask_format
                for segm in segms:
                    if isinstance(segm, list):
                        # polygon
                        masks.append(polygons_to_bitmask(segm, *image_shape))
                    elif isinstance(segm, dict):
                        # COCO RLE
                        masks.append(mask_util.decode(segm))
                    elif isinstance(segm, np.ndarray):
                        # mask array
                        assert (
                            segm.ndim == 2
                        ), f"Expect segmentation of 2 dimensions, got {segm.ndim}."
                        masks.append(segm)
                    else:
                        raise ValueError(
                            f"Cannot convert segmentation of type '{type(segm)}' to BitMasks!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict, or a binary segmentation mask "
                            " in a 2D numpy array of shape HxW."
                        )

        masks = [torch.from_numpy(np.ascontiguousarray(mask)) for mask in masks]

        if len(masks) > 0 and self.size_divisibility > 0:
            padding_size = [
                0,
                self.size_divisibility - image_shape[1] % self.size_divisibility,
                0,
                self.size_divisibility - image_shape[0] % self.size_divisibility,
            ]

            # update image shape
            image_shape = (
                image_shape[0] + padding_size[3],
                image_shape[1] + padding_size[1],
            )
            masks = [F.pad(mask, padding_size, value=0).contiguous() for mask in masks]

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            target.gt_masks = BitMasks(torch.zeros((0, image_shape[0], image_shape[1])))
        else:
            target.gt_masks = BitMasks(torch.stack(masks))

        target.gt_boxes = target.gt_masks.get_bounding_boxes()

        target = utils.filter_empty_instances(target)

        dataset_dict["instances"] = target

    def maybe_add_padding(self, image: np.ndarray) -> torch.Tensor:
        image_shape = image.shape[:2]
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.size_divisibility > 0:
            padding_size = [
                0,
                self.size_divisibility - image_shape[1] % self.size_divisibility,
                0,
                self.size_divisibility - image_shape[0] % self.size_divisibility,
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
        return image

    def __call__(self, dataset_dict: dict) -> dict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)

        # Remove the annotations in training mode
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)

        image = self._load_image(dataset_dict["file_name"])["image"]
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        if "annotations" in dataset_dict:
            self._transform_annotations_with_padding(
                dataset_dict,
                transforms,
                image.shape[:2],
            )

        dataset_dict["image"] = self.maybe_add_padding(image)

        return dataset_dict
