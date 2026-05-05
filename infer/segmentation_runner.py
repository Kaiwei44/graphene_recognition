from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor

from flake_postprocess import GrapheneFlakePostprocessor, PostprocessParams
from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models import M2F_model
from maskterial.modeling.segmentation_models.M2F import maskformer_model  # noqa: F401
from maskterial.structures.FlakeClass import Flake
from maskterial.utils.dataset_functions import setup_config


DEFAULT_SEG_WEIGHTS = (
    "~/Parameter/trained_para/graphene_mid_finetune_remove_small_flakes_obj20/"
    "model_0009999.pth"
)


@dataclass(slots=True)
class SegmentationResult:
    raw_flakes: list[Flake]
    final_flakes: list[Flake]


def expand_path(path: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def resolve_seg_config(weights_path: str | os.PathLike, config_path: str | os.PathLike | None) -> Path:
    if config_path:
        resolved = expand_path(config_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Segmentation config not found: {resolved}")
        return resolved

    weights = expand_path(weights_path)
    candidate = weights.parent / "config.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Segmentation config was not provided and no config.yaml was found next to "
        f"the weights file: {weights}. Pass --seg-config explicitly."
    )


def build_postprocess_params(args) -> PostprocessParams:
    return PostprocessParams(
        overlap_iou_threshold=args.pp_overlap_iou_threshold,
        overlap_containment_threshold=args.pp_overlap_containment_threshold,
        enable_bridge_merge=args.pp_enable_bridge_merge,
        grow_radius_px=args.pp_grow_radius_px,
        max_boundary_distance_px=args.pp_max_boundary_distance_px,
        lab_l_weight=args.pp_lab_l_weight,
        tau_grow=args.pp_tau_grow,
        tau_pair=args.pp_tau_pair,
        grow_sigma=args.pp_grow_sigma,
        min_bridge_area_px=args.pp_min_bridge_area_px,
        min_bridge_ratio=args.pp_min_bridge_ratio,
        final_min_area_um2=args.pp_final_min_area_um2,
        final_min_score=args.pp_final_min_score,
        final_max_shape_complexity=args.pp_final_max_shape_complexity,
        max_bridge_passes=args.pp_max_bridge_passes,
    )


def postprocess_params_dict(params: PostprocessParams) -> dict:
    return asdict(params)


def flake_score(flake: Flake) -> float:
    return float(max(0.0, min(1.0, 1.0 - float(flake.false_positive_probability))))


def flake_area_um2(flake: Flake) -> float:
    if flake.measurements is None:
        return 0.0
    return float(flake.measurements.area_um2)


def flake_area_px(flake: Flake) -> int:
    if flake.measurements is not None:
        return int(flake.measurements.area_px)
    return int(np.count_nonzero(np.asarray(flake.mask) > 0))


def flake_shape_complexity(flake: Flake) -> float:
    mask = (np.asarray(flake.mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_px = sum(cv2.arcLength(contour, True) for contour in contours)
    area_px = max(int(np.count_nonzero(mask)), 1)
    return float((perimeter_px * perimeter_px) / (4.0 * np.pi * area_px))


def flake_summary(flake: Flake) -> dict:
    return {
        "seg_score": flake_score(flake),
        "area_px": flake_area_px(flake),
        "area_um2": flake_area_um2(flake),
        "center_x": int(flake.center[0]),
        "center_y": int(flake.center[1]),
        "max_sidelength_px": float(flake.max_sidelength),
        "min_sidelength_px": float(flake.min_sidelength),
        "shape_complexity": flake_shape_complexity(flake),
    }


class SegmentationRunner:
    def __init__(
        self,
        config_path: str | os.PathLike,
        weights_path: str | os.PathLike,
        device: str = "cuda",
        score_threshold: float = 0.0,
        size_threshold_px: int = 0,
        postprocess_params: PostprocessParams | None = None,
        use_postprocess: bool = True,
    ) -> None:
        self.config_path = expand_path(config_path)
        self.weights_path = expand_path(weights_path)
        self.device = str(device)
        self.use_postprocess = bool(use_postprocess)
        self.postprocess_params = postprocess_params or PostprocessParams()

        cfg = self._build_cfg()
        self._clamp_detection_count(cfg)
        predictor = DefaultPredictor(cfg)
        seg_model = M2F_model(
            model=predictor.model,
            config=cfg,
            device=torch.device(cfg.MODEL.DEVICE),
        )
        self.maskterial = MaskTerial(
            segmentation_model=seg_model,
            score_threshold=float(score_threshold),
            min_class_occupancy=0.0,
            size_threshold=int(size_threshold_px),
            device=torch.device(cfg.MODEL.DEVICE),
        )
        self.postprocessor = GrapheneFlakePostprocessor(self.postprocess_params)

    def _build_cfg(self):
        args = SimpleNamespace(
            config_file=str(self.config_path),
            opts=[
                "MODEL.WEIGHTS",
                str(self.weights_path),
                "MODEL.DEVICE",
                self.device,
            ],
            resume=False,
            eval_only=True,
            num_gpus=1,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
        )
        return setup_config(args)

    @staticmethod
    def _clamp_detection_count(cfg) -> None:
        max_scores = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES * cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        if cfg.TEST.DETECTIONS_PER_IMAGE <= max_scores:
            return
        cfg.defrost()
        cfg.TEST.DETECTIONS_PER_IMAGE = max_scores
        cfg.freeze()
        print(f"[INFO] Clamped TEST.DETECTIONS_PER_IMAGE to {max_scores} available query scores")

    def run_image(
        self,
        image_bgr: np.ndarray,
        image_stem: str,
        debug_vis_dir: str | os.PathLike | None = None,
    ) -> SegmentationResult:
        raw_flakes = self.maskterial.predict(image_bgr)
        if not self.use_postprocess:
            return SegmentationResult(raw_flakes=raw_flakes, final_flakes=raw_flakes)

        result = self.postprocessor.run(
            image_bgr=image_bgr,
            raw_flakes=raw_flakes,
            debug_vis_dir=str(debug_vis_dir) if debug_vis_dir is not None else None,
            image_stem=image_stem,
        )
        return SegmentationResult(raw_flakes=raw_flakes, final_flakes=result.final_flakes)
