import os

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config


def register_all_datasets(root: str, name_prefix: str = "", coco_file_suffix="_300"):
    dataset_names = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    for dataset_name in dataset_names:
        dataset_path = os.path.join(root, dataset_name)
        register_coco_instances(
            f"{name_prefix}{dataset_name}_train",
            {},
            os.path.join(
                dataset_path,
                "RLE_annotations",
                f"train_annotations{coco_file_suffix}.json",
            ),
            os.path.join(dataset_path, "train_images"),
        )
        register_coco_instances(
            f"{name_prefix}{dataset_name}_test",
            {},
            os.path.join(
                dataset_path,
                "RLE_annotations",
                f"test_annotations{coco_file_suffix}.json",
            ),
            os.path.join(dataset_path, "test_images"),
        )
        register_coco_instances(
            f"{name_prefix}{dataset_name}_cls_train",
            {},
            os.path.join(
                dataset_path,
                "RLE_annotations",
                f"train_annotations_with_class{coco_file_suffix}.json",
            ),
            os.path.join(dataset_path, "train_images"),
        )
        register_coco_instances(
            f"{name_prefix}{dataset_name}_cls_test",
            {},
            os.path.join(
                dataset_path,
                "RLE_annotations",
                f"test_annotations_with_class{coco_file_suffix}.json",
            ),
            os.path.join(dataset_path, "test_images"),
        )


def register_all_fewshot_datasets(
    root: str,
    shots: list = [1, 2, 3, 5, 10],
    run_ids: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
):
    dataset_names = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    for dataset_name in dataset_names:
        dataset_path = os.path.join(root, dataset_name)
        image_dir = os.path.join(dataset_path, "train_images")
        for shot in shots:
            for run_id in run_ids:
                register_coco_instances(
                    f"{dataset_name}_{shot}_shot_run_{run_id}_train",
                    {},
                    os.path.join(
                        dataset_path,
                        "RLE_annotations",
                        f"{shot}_shot",
                        f"run_{run_id}",
                        f"train_annotations_300.json",
                    ),
                    image_dir,
                )
                register_coco_instances(
                    f"{dataset_name}_{shot}_shot_run_{run_id}_cls_train",
                    {},
                    os.path.join(
                        dataset_path,
                        "RLE_annotations",
                        f"{shot}_shot",
                        f"run_{run_id}",
                        f"train_annotations_with_class_300.json",
                    ),
                    image_dir,
                )


def add_extended_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0

    # optimizer
    cfg.SOLVER.OPTIMIZER = "SGD"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = (
        "MultiScaleMaskedTransformerDecoder"
    )

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def setup_config(args: dict) -> CN:
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_extended_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
