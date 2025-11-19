import cv2
import numpy as np
import torch
from detectron2.structures import Boxes, Instances

from .structures.FlakeClass import Flake
from .modeling.classification_models import BaseClassificationHead
from .modeling.postprocessing_models import BasePostprocessingModel
from .modeling.segmentation_models import BaseSegmentationModel


class MaskTerial:
    def __init__(
        self,
        segmentation_model: BaseSegmentationModel = None,
        classification_model: BaseClassificationHead = None,
        postprocessing_model: BasePostprocessingModel = None,
        score_threshold: float = 0.3,
        min_class_occupancy: float = 0.5,
        size_threshold: int = 200,
        device: torch.device | str = torch.device("cpu"),
    ):
        """Initialize the MaskTerial model with segmentation, classification, and postprocessing components.

        Args:
            segmentation_model (BaseSegmentationModel, optional): The segmentation model for detecting
                flake instances. Defaults to None.
            classification_model (BaseClassificationHead, optional): The classification model for
                determining flake thickness/type. Defaults to None.
            postprocessing_model (BasePostprocessingModel, optional): The postprocessing model for
                filtering false positives. Defaults to None.
            score_threshold (float, optional): Minimum confidence score for predictions to be kept.
                Defaults to 0.3.
            min_class_occupancy (float, optional): Minimum occupancy ratio required for a class
                prediction to be accepted. Defaults to 0.5.
            size_threshold (int, optional): Minimum size (in pixels) for detected flakes to be kept.
                Defaults to 200.
            device (torch.device | str, optional): Device to run inference on. Defaults to CPU.

        Raises:
            AssertionError: If both segmentation_model and classification_model are None.
        """
        self.segmentation_model = segmentation_model
        self.classification_model = classification_model
        self.postprocessing_model = postprocessing_model

        assert not (
            self.segmentation_model is None and self.classification_model is None
        ), "At least one of the segmentation or classification model should be provided"

        self.device = device
        self.size_threshold = size_threshold
        self.score_threshold = score_threshold
        self.min_class_occupancy = min_class_occupancy

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
    ) -> list[Flake]:
        """Predict flakes in the given image using the configured models.

        This method performs end-to-end inference on a single image, detecting flakes
        and optionally classifying their thickness and applying postprocessing filters.

        Args:
            image (torch.Tensor | np.ndarray): Input image to analyze. Can be either a
                PyTorch tensor or numpy array. If numpy array, it will be converted
                to a torch tensor with dtype uint8.

        Returns:
            list[Flake]: List of detected flakes with their properties including mask,
                thickness, size, center coordinates, and confidence scores.

        Raises:
            AssertionError: If the input image is not a tensor or numpy array.

        Note:
            Currently only supports single image input. Batched input support is
            planned for future versions.
        """
        # TODO: Maybe add support for batched input?, currently only supports single image input

        assert type(image) in [
            torch.Tensor,
            np.ndarray,
        ], "Input image must be a tensor or numpy array"

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.uint8)

        image = image.to(self.device)

        if self.segmentation_model is None:
            flakes = self._inference_with_classification_model(image)
        else:
            flakes = self._inference_with_segmentation_model(image)

        if self.postprocessing_model is not None:
            flakes = self._apply_postprocessing_model(image, flakes)

        return flakes

    def _extract_flake_contours_from_semantic_segmentation(
        self,
        semantic_segmentation: np.ndarray | torch.Tensor,
    ) -> tuple[list[np.ndarray], list[int]]:

        def _extract_valid_labels(stats: np.ndarray) -> tuple[list[int], list[int]]:
            """Applies a size threshold to all the detected Flakes

            Args:
                stats (np.ndarray): The Stats of the connected components, has the shape (K x 5) with K being the number of detected components

            Returns:
                Tuple[List[int], List[int]]: A Tuple of two lists, the first list contains the valid labels, the second list contains the sizes of the valid labels
            """
            num_labels = stats.shape[0]
            valid_labels = [
                i
                for i in range(1, num_labels)
                if stats[i, cv2.CC_STAT_AREA] >= self.size_threshold
            ]
            flake_sizes = stats[valid_labels, cv2.CC_STAT_AREA]

            return valid_labels, flake_sizes

        if isinstance(semantic_segmentation, torch.Tensor):
            semantic_segmentation = semantic_segmentation.cpu().numpy()

        all_contours = []
        all_thicknesses = []
        for layer_index in np.unique(semantic_segmentation):
            if layer_index == 0:
                continue

            layer_mask = (semantic_segmentation == layer_index).astype(np.uint8)

            # Remove small outliers
            layer_mask = cv2.morphologyEx(
                layer_mask,
                cv2.MORPH_OPEN,
                kernel=np.ones((3, 3)),
                iterations=2,
            )
            # Skip the layer if there are not enough pixels
            if cv2.countNonZero(layer_mask) < self.size_threshold:
                continue

            # label each connected 'blob' on the mask with an individual number
            # each of these blobs is a flake candidate
            _, labeled_mask, stats, _ = cv2.connectedComponentsWithStats(
                layer_mask, connectivity=4
            )

            # The candidates are filterd by size
            # Only keep the candidates with a size above the threshold
            # and filter the 0 candidate, this is the background
            valid_labels, _ = _extract_valid_labels(stats)
            if len(valid_labels) == 0:
                continue

            ### only keep the instances that are of a certain size
            culled_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
            for label_id in valid_labels:
                culled_mask[labeled_mask == label_id] = 255

            ### get all the contours of the culled mask
            contours, _ = cv2.findContours(
                image=culled_mask,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE,
            )

            all_contours.extend(contours)
            all_thicknesses.extend([layer_index] * len(contours))

        return all_contours, all_thicknesses

    def _apply_postprocessing_model(
        self,
        image: torch.Tensor,
        flakes: list[Flake],
    ) -> list[Flake]:

        flakes: list[Flake] = self.postprocessing_model(image, flakes)

        flakes = [
            flake
            for flake in flakes
            if 1 - flake.false_positive_probability > self.score_threshold
        ]

        return flakes

    def _inference_with_classification_model(
        self,
        image: torch.Tensor,
    ) -> list[Flake]:

        contrast_image = self.calculate_contrast_image(image)
        semantic_segmentation = self.classification_model(contrast_image)
        contours, thicknesses = self._extract_flake_contours_from_semantic_segmentation(
            semantic_segmentation
        )

        flakes: list[Flake] = []
        for contour, thickness in zip(contours, thicknesses):
            flake_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(flake_mask, [contour], -1, 1, -1)

            size = np.sum(flake_mask)

            if size < self.size_threshold:
                continue

            # get the center of the flake, via the moments
            moments = cv2.moments(contour)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            rect = cv2.minAreaRect(contour)
            max_sidelength = max(rect[1])
            min_sidelength = min(rect[1])

            flakes.append(
                Flake(
                    mask=flake_mask,
                    thickness=thickness,
                    size=size,
                    mean_contrast=np.array([0, 0, 0]),
                    center=np.array([center_x, center_y]),
                    min_sidelength=min_sidelength,
                    max_sidelength=max_sidelength,
                    false_positive_probability=0,
                    entropy=0,
                )
            )

        return flakes

    def _inference_with_segmentation_model(
        self,
        image: torch.Tensor,
    ) -> list[Flake]:
        instances: Instances = self.segmentation_model(image)
        instances = self.filter_instances(instances)

        if len(instances) == 0:
            return []

        if self.classification_model is not None:
            flakes = self.infer_classes(image, instances)
        else:
            flakes = self.convert_instances_to_flakes(instances)

        return flakes

    @torch.inference_mode()
    def __call__(self, input: list[dict]) -> list[dict]:
        # TODO: Add support for batched input, currently only supports single image input
        # Should improve evaluation speed

        image = input[0]["image"].permute(1, 2, 0).to(self.device)

        if self.segmentation_model is None:
            flakes = self._inference_with_classification_model(image)
        else:
            flakes = self._inference_with_segmentation_model(image)

        if self.postprocessing_model is not None:
            flakes = self._apply_postprocessing_model(image, flakes)

        return [{"instances": self.convert_flakes_to_instances(image, flakes)}]

    def convert_flakes_to_instances(
        self,
        image: torch.Tensor,
        flakes: list[Flake],
    ) -> Instances:

        # We need to subtract 1 from the thickness to get the correct class, as coco expects the classes to start from 0
        classes = np.array(
            [int(flake.thickness) - 1 for flake in flakes],
        )
        scores = np.array(
            [1 - flake.false_positive_probability for flake in flakes],
        )
        pred_masks = np.array(
            [torch.tensor(flake.mask, dtype=torch.uint8) for flake in flakes],
        )
        pred_boxes = Boxes(
            torch.zeros(
                (len(classes), 4),
                dtype=torch.float32,
                device=self.device,
            )
        )

        return Instances(
            image_size=(image.shape[0], image.shape[1]),
            pred_masks=pred_masks,
            scores=scores,
            pred_classes=classes,
            pred_boxes=pred_boxes,
        )

    def convert_instances_to_flakes(
        self,
        instances: Instances,
    ) -> list[Flake]:

        flakes = []
        for instance_index in range(len(instances)):
            mask = instances[instance_index].pred_masks.squeeze()
            score = instances[instance_index].scores.squeeze().item()

            # we need to increase the class_id by 1, as coco expects the classes to start from 0
            class_id = instances[instance_index].pred_classes.squeeze().item() + 1

            flake_properties, mask = self.extract_properties(mask)

            if flake_properties is None:
                continue

            flakes.append(
                Flake(
                    mask=mask,
                    false_positive_probability=1 - score,
                    thickness=str(class_id),
                    mean_contrast=np.array([0, 0, 0]),
                    **flake_properties,
                )
            )
        return flakes

    def filter_instances(
        self,
        instances: Instances,
    ) -> list[tuple[torch.Tensor, float]]:
        instances = instances[instances.scores > self.score_threshold]
        instances = instances[
            torch.count_nonzero(instances.pred_masks, dim=(1, 2)) > self.size_threshold
        ]
        return instances

    def extract_properties(
        self, mask: torch.Tensor | np.ndarray
    ) -> tuple[dict, np.ndarray] | tuple[None, None]:

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(np.uint8)

        # Extract contours from the mask
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(contours) == 0:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        updated_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(updated_mask, [largest_contour], -1, 1, -1)

        size = cv2.contourArea(largest_contour)

        moments = cv2.moments(largest_contour)
        center_x = int(moments["m10"] / (moments["m00"] + 0.0001))
        center_y = int(moments["m01"] / (moments["m00"] + 0.0001))

        rect = cv2.minAreaRect(largest_contour)
        max_sidelength = float(max(rect[1]))
        min_sidelength = float(min(rect[1]))

        return {
            "size": size,
            "center": (center_x, center_y),
            "max_sidelength": max_sidelength,
            "min_sidelength": min_sidelength,
        }, updated_mask

    @torch.inference_mode()
    def infer_classes(
        self,
        image: torch.Tensor,
        instances: Instances,
    ) -> list[Flake]:
        contrast_image = self.calculate_contrast_image(image)

        # TODO: Improve this by first creating a batched input from the instances and then feeding it to the cls model
        sem_seg = self.classification_model(contrast_image)

        flakes = []
        for instance_index in range(len(instances)):

            mask = instances[instance_index].pred_masks.squeeze()
            score = instances[instance_index].scores.squeeze().item()

            # get the predicted classes from the mask
            class_preds = sem_seg[mask]

            # get the counts for each class predicted in the mask
            classes, counts = torch.unique(class_preds, return_counts=True)

            # normalize the counts to get the probability of each class
            normalized_preds = counts / torch.sum(counts)

            # get the class with the highest probability
            max_prob, max_index = torch.max(normalized_preds, dim=0)
            pred_class = classes[max_index]

            # class_id 0 is the background class, so remove it
            if pred_class == 0 or max_prob < self.min_class_occupancy:
                continue

            flake_properties, mask = self.extract_properties(mask)

            # it may happen that we cant find any properties for the flake
            # in this case we skip the flake
            if flake_properties is None:
                continue

            mean_contrast = torch.mean(contrast_image[mask.astype(bool)], dim=0)

            flakes.append(
                Flake(
                    mask=mask,
                    false_positive_probability=1 - score,
                    thickness=str(pred_class.item()),
                    mean_contrast=mean_contrast.cpu().numpy(),
                    **flake_properties,
                )
            )
        return flakes

    def calculate_mean_background_values(
        self,
        image: torch.Tensor,
        radius: int = 5,
        min_value: int = 20,
        max_value: int = 230,
    ) -> torch.Tensor:
        # if the image is of type half, we need to convert it to float32
        # the reason for this is that the histogram function does not support half precision
        if (
            image.dtype == torch.float16
            or image.dtype == torch.half
            or image.dtype == torch.uint8
        ):
            image = image.float()

        mode_values = torch.zeros(3, dtype=torch.uint8, device=image.device)

        for channel_index in range(3):
            count = torch.histc(image[:, :, channel_index], bins=256, min=0, max=255)
            mode_values[channel_index] = (
                torch.argmax(count[min_value : max_value + 1]) + min_value
            )

        lower_bounds = torch.clamp(mode_values - radius, min=min_value, max=max_value)
        upper_bounds = torch.clamp(mode_values + radius, max=max_value, min=min_value)

        # Vectorized computation for all channels
        masks = (image >= lower_bounds.view(1, 1, 3)) & (
            image <= upper_bounds.view(1, 1, 3)
        )
        sum_values = torch.sum(image * masks, dim=(0, 1))
        num_values = torch.sum(masks, dim=(0, 1))

        means = sum_values / torch.where(
            num_values != 0, num_values, torch.tensor([1.0], device=image.device)
        )

        return means.half()

    def calculate_contrast_image(self, image: torch.Tensor) -> torch.Tensor:
        mean_background_values = self.calculate_mean_background_values(image)

        contrast_image = image / mean_background_values - 1
        return contrast_image
