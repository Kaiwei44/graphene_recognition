import os

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances

from ..seg_model_interface import BaseSegmentationModel
from .maskformer_model import MaskFormer


class M2F_model(BaseSegmentationModel):
    def __init__(
        self,
        model: torch.nn.Module,
        config: CfgNode,
        device: str | torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def _format_input(
        self,
        input: torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray],
    ) -> list[dict]:

        if not isinstance(input, list):
            input = [input]

        formatted_input = []
        for input_elem in input:
            if isinstance(input_elem, np.ndarray):
                inp = torch.from_numpy(input_elem).to(self.device)
            elif isinstance(input_elem, torch.Tensor):
                inp = input_elem.clone().to(self.device)
            else:
                raise ValueError(
                    f"Input should be a list of torch.Tensor or np.ndarray, got {type(input_elem)}"
                )

            formatted_input.append(
                {
                    "image": inp.permute(2, 0, 1),
                    "height": inp.shape[0],
                    "width": inp.shape[1],
                }
            )

        return formatted_input

    @staticmethod
    def from_pretrained(
        path: str,
        device: str | torch.device = torch.device("cpu"),
        **kwargs,
    ) -> "M2F_model":

        cfg_path = os.path.join(path, "config.yaml")
        model_path = os.path.join(path, "model.pth")

        if not os.path.exists(model_path):
            model_path = os.path.join(path, "model_final.pth")

        assert os.path.exists(
            cfg_path
        ), f"Config file not found at {cfg_path}, Please make sure `config.yaml` is present in the model directory."
        assert os.path.exists(
            model_path
        ), f"Model file not found at {model_path}, Please make sure `model.pth` is present in the model directory."

        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = device if type(device) is str else device.type

        segmentation_model = build_model(cfg)
        DetectionCheckpointer(segmentation_model).load(model_path)
        segmentation_model.to(device)
        # Removed Compilation, currently does not improve performance
        # segmentation_model = torch.compile(
        #     segmentation_model,
        #     fullgraph=False,
        #     mode="reduce-overhead",
        # )
        segmentation_model.eval()

        return M2F_model(
            model=segmentation_model,
            config=cfg,
            device=device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        image: torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray],
    ) -> list[Instances] | Instances:
        self.model.eval()

        # if type(self.device) == str:
        #     device_type = self.device
        # else:
        #     device_type = self.device.type

        # Working with lower precision did not improve performance
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        model_output: list = self.model(self._format_input(image))

        instances_list = []
        # cast to bool
        for output in model_output:
            instances = output["instances"]
            instances.pred_masks = instances.pred_masks > 0.5
            instances_list.append(instances)

        if len(model_output) == 1:
            return instances_list[0]
        else:
            return instances_list
