import json
import os

import numpy as np
import torch

from ...common.distributions import (
    ImprovedMultivariateNormal,
    ImprovedMultivariateNormalHalf,
)
from ...common.fcresnet import FCResNet
from ..cls_head_interface import BaseClassificationHead


class AMM_head(BaseClassificationHead):
    def __init__(
        self,
        model: FCResNet,
        gmm: ImprovedMultivariateNormal | ImprovedMultivariateNormalHalf,
        train_mean: torch.Tensor,
        train_std: torch.Tensor,
        std_threshold: float,
        device=torch.device("cpu"),
    ) -> None:
        self.model = model
        self.gmm = gmm
        self.train_mean = train_mean
        self.train_std = train_std
        self.device = device
        self.std_treshold = std_threshold

    def postprocess_mh_map(
        self,
        distance_map: torch.Tensor,
        distance_threshold: float = 10,
    ) -> torch.Tensor:
        """Postprocesses the Mahalanobis distance map to get the semantic map of the flakes\n
        This generates a semantic map of flakes with no overlap\n

        Args:
            distance_map (torch.Tensor): The Mahalanobis distance map of the image of shape (H x W x K) with K being the number of components and H and W being the height and width of the image
            distance_threshold (float, optional): The Maximum Distance a value can have in Standard deviation. Defaults to 5.

        Returns:
            torch.Tensor: The semantic map of the flakes of shape (H x W) with K being the number of components and H and W being the height and width of the image
        """
        mh_distance, mh_class = distance_map.min(dim=-1)
        mh_class[mh_distance > distance_threshold] = 0
        return mh_class

    @torch.inference_mode()
    def __call__(self, contrast_image: torch.Tensor) -> torch.Tensor:

        if type(self.device) is str:
            device_type = self.device
        else:
            device_type = self.device.type

        image_shape = contrast_image.shape

        with torch.autocast(device_type=device_type, dtype=torch.float16):
            contrast_image = (contrast_image - self.train_mean) / self.train_std
            contrast_image = contrast_image.view(-1, 3)

            embedding = self.model.get_embedding(contrast_image)
            mh_distance_map = self.gmm.mh_distance(embedding[:, None, :])
            semantic_map = self.postprocess_mh_map(mh_distance_map, self.std_treshold)

        return semantic_map.view(image_shape[:-1])

    def get_mh_distance_map(self, contrast_image: torch.Tensor) -> torch.Tensor:
        image_shape = contrast_image.shape

        contrast_image = (contrast_image - self.train_mean) / self.train_std
        contrast_image = contrast_image.view(-1, 3)

        embedding = self.model.get_embedding(contrast_image)
        mh_distance_map = self.gmm.mh_distance(embedding[:, None, :])
        mh_distance_map = mh_distance_map.view(
            image_shape[0], image_shape[1], -1
        ).permute(2, 0, 1)

        return mh_distance_map

    @staticmethod
    def from_pretrained(
        path: str,
        device: torch.device = torch.device("cpu"),
        std_threshold: int = 10,
        eps: float = 1e-5,
        **kwargs,
    ) -> "AMM_head":

        loc_path = os.path.join(path, "loc.npy")
        cov_path = os.path.join(path, "cov.npy")
        model_path = os.path.join(path, "model.pth")
        meta_path = os.path.join(path, "meta_data.json")

        assert os.path.exists(loc_path), f"loc.npy not found at {loc_path}"
        assert os.path.exists(cov_path), f"cov.npy not found at {cov_path}"
        assert os.path.exists(model_path), f"model.pth not found at {model_path}"
        assert os.path.exists(meta_path), f"meta_data.json not found at {meta_path}"

        with open(meta_path, "r") as f:
            meta_data = json.load(f)

        loc = torch.from_numpy(np.load(loc_path)).float().to(device)
        cov = torch.from_numpy(np.load(cov_path)).float().to(device)
        train_mean, train_std = (
            torch.from_numpy(np.array(meta_data["train_mean"])).float().to(device),
            torch.from_numpy(np.array(meta_data["train_std"])).float().to(device),
        )

        # the cpu doesnt support the faster half precision
        # so we have to use the slower full precision when running on cpu
        MultivariateNormal = (
            ImprovedMultivariateNormal
            if device == torch.device("cpu") or device == "cpu"
            else ImprovedMultivariateNormalHalf
        )

        if eps is None:
            for reg in [1e-7, 1e-6, 1e-5, 1e-4]:
                try:
                    # we need to add a small regularization to the covariance matrix so it doesnt become singular
                    reg_cov = cov + reg * torch.eye(cov.shape[-1], device=device)
                    gmm = MultivariateNormal(
                        loc=loc,
                        cov=reg_cov,
                    )
                    print(f"Loaded GMM with reg={reg}")
                    break
                except torch._C._LinAlgError as e:
                    print(f"LinAlgError: {e}")
        else:
            # we need to add a small regularization to the covariance matrix so it doesnt become singular
            reg_cov = cov + eps * torch.eye(cov.shape[-1], device=device)
            gmm = MultivariateNormal(
                loc=loc,
                cov=reg_cov,
            )

        model = FCResNet(
            **meta_data["train_config"]["model_arch"],
        ).to(device)

        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)
        model.compile()
        model.eval()

        segmentation_head = AMM_head(
            model=model,
            gmm=gmm,
            train_mean=train_mean,
            train_std=train_std,
            std_threshold=std_threshold,
            device=device,
        )

        return segmentation_head
