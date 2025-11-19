import numpy as np
import torch

from ...structures.FlakeClass import Flake


class BasePostprocessingModel:
    def __call__(
        self,
        image: torch.Tensor | np.ndarray,
        Flakes: list[Flake],
    ) -> list[Flake]:
        """_summary_

        Args:
            image (torch.Tensor or np.ndarray): An input image tensor of shape (H x W x C).
            Flakes (list[Flake]): A list of Flake instances.

        Returns:
            list[Flake]: A list of Flake instances with updated attributes.
        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "BasePostprocessingModel":
        """
        Load the model from the given path.

        Args:
            path (str): The path to the model directory.

        """
        raise NotImplementedError
