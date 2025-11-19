import json
import os

import numpy as np
import torch

from ...common.fcresnet import FCResNet
from ..cls_head_interface import BaseClassificationHead


class MLP_head(BaseClassificationHead):
    def __init__(
        self,
        model: torch.nn.Module,
        train_mean: torch.Tensor,
        train_std: torch.Tensor,
        device=torch.device("cuda:0"),
    ) -> None:
        self.model = model
        self.train_mean = train_mean
        self.train_std = train_std
        self.device = device

    @torch.no_grad()
    def __call__(self, contrast_image):
        image_shape = contrast_image.shape

        contrast_image = (contrast_image - self.train_mean) / self.train_std
        contrast_image = contrast_image.view(-1, 3)

        output = self.model(contrast_image)
        output = torch.argmax(output, dim=-1)
        output = output.view(image_shape[:-1])

        return output

    @staticmethod
    def from_pretrained(
        path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> "MLP_head":

        model_path = os.path.join(path, "model.pth")
        meta_path = os.path.join(path, "meta_data.json")

        assert os.path.exists(model_path), f"model.pth not found at {model_path}"
        assert os.path.exists(meta_path), f"meta_data.json not found at {meta_path}"

        with open(meta_path, "r") as f:
            meta_data = json.load(f)

        train_mean, train_std = (
            torch.from_numpy(np.array(meta_data["train_mean"])).float().to(device),
            torch.from_numpy(np.array(meta_data["train_std"])).float().to(device),
        )

        model = FCResNet(
            **meta_data["train_config"]["model_arch"],
        ).to(device)

        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        model.eval()

        segmentation_head = MLP_head(
            model=model,
            train_mean=train_mean,
            train_std=train_std,
            device=device,
        )

        return segmentation_head
