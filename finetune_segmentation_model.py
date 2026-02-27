import warnings

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch

from maskterial.modeling.segmentation_models.M2F.maskformer_model import (
    MaskFormer,  # noqa: F401
)
from maskterial.modeling.segmentation_models.M2F.modeling import *  # noqa: F401, F403
from maskterial.utils.argparser import parse_seg_args
from maskterial.utils.dataset_functions import setup_config
from maskterial.utils.model_trainer import MaskTerial_Trainer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def resolve_train_data_paths(args):
    if args.train_mode == "normal":
        if args.train_image_root is None or args.train_annotation_path is None:
            raise ValueError(
                "When --train-mode normal, both --train-image-root and "
                "--train-annotation-path must be provided."
            )
        return args.train_image_root, args.train_annotation_path

    if (
        args.diffusion_train_image_root is None
        or args.diffusion_train_annotation_path is None
    ):
        raise ValueError(
            "When --train-mode diffusion, both --diffusion-train-image-root and "
            "--diffusion-train-annotation-path must be provided."
        )
    return args.diffusion_train_image_root, args.diffusion_train_annotation_path


def main(args: dict):
    cfg = setup_config(args)
    trainer = MaskTerial_Trainer(
        cfg,
        pretraining_augmentations=args.pretraining_augmentations,
    )
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = parse_seg_args()
    train_image_root, train_annotation_path = resolve_train_data_paths(args)
    args.train_image_root = train_image_root
    args.train_annotation_path = train_annotation_path

    register_coco_instances(
        "Maskterial_Dataset",
        {},
        train_annotation_path,
        train_image_root,
    )

    print("Command Line Args:", args)
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
