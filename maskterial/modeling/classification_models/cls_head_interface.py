import torch


class BaseClassificationHead:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, contrast_image: torch.Tensor) -> torch.Tensor:
        """
        Runs the classification head on the input contrast image.

        Args:
            contrast_image (torch.Tensor): The input contrast image, is expected to be of shape (H x W x C).

        Returns:
            torch.Tensor: The output of the classification head, shape (H x W). The values are the class indices.

        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(path: str, **kwargs) -> "BaseClassificationHead":
        """
        Load the model from the given path.

        Args:
            path (str): The path to the model.

        """
        raise NotImplementedError
