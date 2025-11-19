import cv2
import numpy as np
from detectron2.data import transforms as T
from detectron2.data.transforms import Augmentation
from detectron2.structures import Boxes, pairwise_iou
from fvcore.transforms.transform import CropTransform, NoOpTransform
from numpy import random


class RandomResize(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, lower=0.5, upper=3, apply_prob=1):
        self._init(locals())

    def get_transform(self, image):
        if random.random() > self.apply_prob:
            return NoOpTransform()
        im_shape = image.shape
        scale = random.uniform(self.lower, self.upper)
        new_size = (int(im_shape[0] * scale), int(im_shape[1] * scale))

        return T.ResizeTransform(im_shape[0], im_shape[1], new_size[0], new_size[1])


class ToGray(T.Transform):
    def __init__(self, apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob:
            return img
        return img.mean(axis=2).astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomToGray(T.Transform):
    def __init__(self, apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob:
            return img
        weights = np.random.rand(3)
        weights /= weights.sum()
        gray_img = np.dot(img, weights).astype(np.uint8)
        return gray_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class GaussianNoise(T.Transform):
    def __init__(self, noise_std_range=[0, 20], apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob:
            return img
        noise_std = random.uniform(*self.noise_std_range)
        img = img.copy().astype(np.float32)
        noise = np.random.normal(0, noise_std, img.shape)
        img += noise

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class SaltAndPepperNoise(T.Transform):
    def __init__(self, prob_range=[0.0, 0.1], apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob:
            return img

        prob = random.uniform(*self.prob_range)
        img = img.copy().astype(np.float32)
        noise = np.random.rand(*img.shape[:2])
        img[noise < prob] = 0
        img[noise > 1 - prob] = 255

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomWhiteBalance(T.Transform):
    def __init__(self, apply_prob=1, lower=0.8, upper=1.2):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob or self.lower == self.upper:
            return img
        img = img.copy().astype(np.float32)

        if len(img.shape) == 2:
            img *= random.uniform(self.lower, self.upper)
        else:
            for i in range(img.shape[2]):
                img[:, :, i] *= random.uniform(self.lower, self.upper)

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomChannelDrop(T.Transform):
    def __init__(self, apply_prob=1, drop_prob=0.1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if (
            self.drop_prob == 0
            or len(img.shape) == 2
            or img.shape[2] == 1
            or random.random() > self.apply_prob
        ):
            return img

        img = img.copy()
        drop = np.random.rand(img.shape[2]) < self.drop_prob

        # drop max N-1 channels where N is the number of channels
        if sum(drop) == img.shape[2]:
            drop[random.randint(0, 2)] = False

        img[:, :, drop] = random.randint(0, 1) * 255
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class RandomChannelShuffle(T.Transform):
    def __init__(self, apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if (
            len(img.shape) == 2
            or img.shape[2] == 1
            or random.random() > self.apply_prob
        ):
            return img

        # Expects the image to be in HWC format
        img = img.copy()
        img = img.transpose(2, 0, 1)
        np.random.shuffle(img)
        return img.transpose(1, 2, 0)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class Blurring(T.Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, k_size_range=(0, 3), apply_prob=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.apply_prob:
            return img

        kernel_size_x = random.randint(*self.k_size_range) * 2 + 1
        kernel_size_y = random.randint(*self.k_size_range) * 2 + 1

        return cv2.GaussianBlur(img, (kernel_size_x, kernel_size_y), 0).astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class MinIoUAbsoluteCrop(Augmentation):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size)
        mode_trials: number of trials for sampling min_ious threshold
        crop_trials: number of trials for sampling crop_size after cropping
    """

    def __init__(
        self,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        crop_size_x=512,
        crop_size_y=512,
        mode_trials=1000,
        crop_trials=50,
    ):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.mode_trials = mode_trials
        self.crop_trials = crop_trials

    def get_transform(self, image, boxes):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            boxes: ground truth boxes in (x1, y1, x2, y2) format
        """

        if boxes is None:
            return NoOpTransform()
        h, w, c = image.shape
        for _ in range(self.mode_trials):
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return NoOpTransform()

            min_iou = mode
            for _ in range(self.crop_trials):
                left = random.uniform(w - self.crop_size_x)
                top = random.uniform(h - self.crop_size_y)

                patch = np.array(
                    (
                        int(left),
                        int(top),
                        int(left + self.crop_size_x),
                        int(top + self.crop_size_y),
                    )
                )
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = pairwise_iou(
                    Boxes(patch.reshape(-1, 4)), Boxes(boxes.reshape(-1, 4))
                ).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = (
                            (center[:, 0] > patch[0])
                            * (center[:, 1] > patch[1])
                            * (center[:, 0] < patch[2])
                            * (center[:, 1] < patch[3])
                        )
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                return CropTransform(
                    int(left), int(top), int(self.crop_size_x), int(self.crop_size_y)
                )
