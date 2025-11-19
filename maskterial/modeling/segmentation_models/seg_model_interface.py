import numpy as np
import torch
from detectron2.structures import Instances


class BaseSegmentationModel:
    def __call__(self, image: torch.Tensor | np.ndarray) -> Instances:
        """_summary_

        Args:
            image (torch.Tensor): An input image tensor of shape (H x W x C).

        Returns:
            Instances: The predicted instances.
        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "BaseSegmentationModel":
        """
        Load the model from the given path.

        Args:
            path (str): The path to the model directory.

        """
        raise NotImplementedError
