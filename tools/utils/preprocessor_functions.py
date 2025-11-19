import os
from typing import List, Tuple

import cv2
import numpy as np


def add_classes_to_instance_masks(
    mask_directory: str,
    mask_save_directory: str,
    instance_classifiers: List[Tuple[str, int]],
    instance_classes: List[int],
) -> None:
    # extract all unique mask names
    mask_names = set([mask_name for mask_name, _ in instance_classifiers])

    for idx, mask_name in enumerate(mask_names):
        print(f"{idx + 1}/{len(mask_names)} Processed", end="\r")

        mask_path = os.path.join(mask_directory, mask_name)
        mask = cv2.imread(mask_path, 0)

        assert mask is not None, f"Could not load mask {mask_path}"

        semantic_mask = np.zeros_like(mask)
        for instance_class, (classifier_name, classifier_id) in zip(
            instance_classes, instance_classifiers
        ):
            if classifier_name == mask_name:
                semantic_mask[mask == classifier_id] = instance_class

        cv2.imwrite(os.path.join(mask_save_directory, mask_name), semantic_mask)

    print(
        f"\nAdded {len([instance_class for instance_class in instance_classes if instance_class != 0])} instances to the instance masks"
    )


def get_instance_contrasts_from_dir(
    image_directory,
    mask_directory,
    flatfield_path=None,
    use_flatfield=False,
    min_instance_size=200,
) -> List[np.ndarray]:
    # returns a list of all instances
    # each instance is a N x 3 Array with N being the number of pixels in the instance and 3 being the BGR values
    instance_contrasts = []
    instance_classifiers = []

    if use_flatfield and flatfield_path is not None:
        flatfield = cv2.imread(flatfield_path)
        assert (
            flatfield is not None
        ), f"Could not load flatfield at '{flatfield_path}', have you selected the correct path?"

    mask_names = os.listdir(mask_directory)

    for idx, mask_name in enumerate(mask_names):
        print(f"{idx + 1}/{len(mask_names)} read", end="\r")

        image_path = os.path.join(image_directory, mask_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(
                image_directory, mask_name.replace(".png", ".jpg")
            )

        if not os.path.exists(image_path):
            print(f"Could not find image corresponding to mask '{mask_name}', skipping")
            continue

        mask_path = os.path.join(mask_directory, mask_name)

        mask = cv2.imread(mask_path, 0)
        image = cv2.imread(image_path)

        assert mask is not None, f"Could not load mask {mask_path}"
        assert image is not None, f"Could not load image {image_path}"

        if use_flatfield and flatfield_path is not None:
            image = remove_vignette(image, flatfield)

        background_color = calculate_background_color(image)

        if np.any(background_color == 0):
            print(f"Error with image {mask_name}; Invalid Background, skipping")
            continue

        contrast_image = image / background_color - 1

        for instance_id in np.unique(mask):
            # skipping the background
            if instance_id == 0:
                continue

            instance_mask = np.zeros_like(mask)
            instance_mask[mask == instance_id] = 1

            if cv2.countNonZero(instance_mask) < min_instance_size:
                continue

            instance_contrasts.append(contrast_image[instance_mask == 1])
            instance_classifiers.append((mask_name, instance_id))

    return instance_contrasts, instance_classifiers


def remove_vignette(
    image,
    flatfield,
    max_background_value: int = 241,
) -> np.ndarray:
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flatfield (NxMx3 Array): the Flat Field in RGB
        max_background_value (int): the maximum value of the background

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """
    image_no_vigentte = image / flatfield * cv2.mean(flatfield)[:-1]
    image_no_vigentte[image_no_vigentte > max_background_value] = max_background_value
    return np.asarray(image_no_vigentte, dtype=np.uint8)


def calculate_background_color(
    img,
    radius=10,
) -> np.ndarray:
    masks = []

    for i in range(3):
        img_channel = img[:, :, i]
        mask = cv2.inRange(img_channel, 20, 230)
        hist = cv2.calcHist([img_channel], [0], mask, [256], [0, 256])
        hist_mode = np.argmax(hist)
        thresholded_image = cv2.inRange(
            img_channel, int(hist_mode - radius), int(hist_mode + radius)
        )
        background_mask_channel = cv2.erode(
            thresholded_image, np.ones((3, 3)), iterations=3
        )
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])

    return np.array(cv2.mean(img, mask=final_mask)[:3])
