import copy
import itertools
import logging
import os
import weakref
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.engine.train_loop import AMPTrainer, HookBase, SimpleTrainer
from detectron2.projects.deeplab.lr_scheduler import WarmupPolyLR
from detectron2.solver import build_lr_scheduler, get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import _log_api_usage, setup_logger
from torch.nn.parallel import DistributedDataParallel

from .dataset_mapper import MaskTerialDatasetMapper
from .evaluator import FlakeCOCOEvaluator
from .transform_functions import (
    Blurring,
    GaussianNoise,
    RandomChannelDrop,
    RandomChannelShuffle,
    RandomResize,
    RandomWhiteBalance,
    SaltAndPepperNoise,
)


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


class MaskTerial_Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(
        self,
        cfg,
        pretraining_augmentations=False,
    ):
        """
        Args:
            cfg (CfgNode):
        """
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.pretraining_augmentations: bool = pretraining_augmentations
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(
            cfg,
            pretraining_augmentations=self.pretraining_augmentations,
        )

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model,
            data_loader,
            optimizer,
            gather_metric_period=20,
            async_write_metrics=True,
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return FlakeCOCOEvaluator(
            dataset_name,
            output_dir=output_folder,
            tasks=("segm",),
        )

    @classmethod
    def build_train_loader(
        cls,
        cfg,
        pretraining_augmentations=False,
    ):
        augmentations = []
        # these are used when pretraining on simulated data
        if pretraining_augmentations:
            augmentations.extend(
                [
                    # Geometric Transformations
                    RandomResize(0.9, 1.1, apply_prob=0.5),
                    T.RandomCrop("absolute", (1024, 1024)),
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                    # Value Transformations
                    RandomChannelDrop(drop_prob=0.1, apply_prob=0.5),
                    RandomChannelShuffle(apply_prob=0.5),
                    RandomWhiteBalance(apply_prob=1, lower=0.7, upper=1.3),
                    Blurring(k_size_range=(0, 3), apply_prob=0.5),
                    GaussianNoise(noise_std_range=(0, 20), apply_prob=0.5),
                    T.RandomContrast(0.8, 1.2),
                    T.RandomBrightness(0.8, 1.2),
                    SaltAndPepperNoise(prob_range=(0, 0.05), apply_prob=0.5),
                ]
            )
        else:
            augmentations.extend(
                [
                    # Geometric Transformations
                    RandomResize(0.9, 1.1, apply_prob=0.5),
                    T.RandomCrop("absolute", (512, 512)),
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                    # Value Transformations
                    Blurring(k_size_range=(0, 2), apply_prob=0.5),
                    GaussianNoise(noise_std_range=(0, 5), apply_prob=0.5),
                    T.RandomContrast(0.8, 1.2),
                    T.RandomBrightness(0.8, 1.2),
                    SaltAndPepperNoise(prob_range=(0, 0.05), apply_prob=0.5),
                ]
            )

        mapper = MaskTerialDatasetMapper(
            cfg,
            is_train=True,
            augmentations=augmentations,
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = MaskTerialDatasetMapper(
            cfg,
            is_train=False,
            augmentations=[],
        )
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        name = cfg.SOLVER.LR_SCHEDULER_NAME
        if name == "WarmupPolyLR":
            return WarmupPolyLR(
                optimizer,
                cfg.SOLVER.MAX_ITER,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
                power=cfg.SOLVER.POLY_LR_POWER,
                constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER

        if optimizer_type == "SGD":
            params = get_default_optimizer_params(
                model,
                base_lr=cfg.SOLVER.BASE_LR,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
            optimizer = torch.optim.SGD(
                params,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                foreach=True,
            )

        elif optimizer_type == "ADAMW":
            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for module_name, module in model.named_modules():
                for module_param_name, value in module.named_parameters(recurse=False):
                    if not value.requires_grad:
                        continue
                    # Avoid duplicating parameters
                    if value in memo:
                        continue
                    memo.add(value)

                    hyperparams = copy.copy(defaults)
                    if "backbone" in module_name:
                        hyperparams["lr"] = (
                            hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                        )
                    if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                    ):
                        hyperparams["weight_decay"] = 0.0
                    if isinstance(module, norm_module_types):
                        hyperparams["weight_decay"] = weight_decay_norm
                    if isinstance(module, torch.nn.Embedding):
                        hyperparams["weight_decay"] = weight_decay_embed
                    params.append({"params": [value], **hyperparams})
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )

        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def build_writers(self):
        writers = super().build_writers()
        use_wandb = os.getenv("WANDB_ACTIVE", "0")
        if use_wandb == "0":
            return writers

        from .wandb_writer import WandbWriter

        writers.append(WandbWriter(config=self.cfg))

        return writers
