import os
from typing import List

import cv2
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ....structures.FlakeClass import Flake
from ..pp_model_interface import BasePostprocessingModel


class L2_model(BasePostprocessingModel):
    def __init__(
        self,
        model: LogisticRegression,
        **kwargs,
    ):
        self.model = model

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor | np.ndarray,
        flakes: List[Flake],
    ) -> list[Flake]:

        for flake in flakes:
            flake_mask = flake.mask
            if isinstance(flake_mask, torch.Tensor):
                flake_mask = flake_mask.cpu().numpy()

            contour = cv2.findContours(
                image=flake_mask,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE,
            )[0][0]

            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            arclength = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            solidity = float(area) / convex_hull_area
            arcarea = arclength / area**0.5

            flake.false_positive_probability = round(
                self.model.predict_proba([[arcarea, solidity]])[0][0],
                3,
            )

        return flakes

    def from_pretrained(path: str, **kwargs) -> "L2_model":
        model_path = os.path.join(path, "model.joblib")

        assert os.path.exists(
            model_path
        ), f"Model file with name 'model.joblib' not found at {model_path}"

        model = joblib.load(model_path)
        return L2_model(model, **kwargs)
