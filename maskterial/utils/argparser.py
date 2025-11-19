import argparse

import numpy as np


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cls-model-type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cls-model-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seg-model-type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seg-model-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pp-model-type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pp-model-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument("--dataset-name", type=str, default="default")
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    return parser.parse_args()


def parse_cls_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to the directory where the model checkpoints will be saved.",
    )
    parser.add_argument(
        "--train-image-dir",
        type=str,
        help="Path to the training image directory.",
    )
    parser.add_argument(
        "--train-annotation-path",
        type=str,
        help="Path to the training annotation file.",
    )
    parser.add_argument(
        "--test-image-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-annotation-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=np.random.randint(0, 2**16 - 1),
    )
    return parser.parse_args()


def parse_seg_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="number of gpus *per machine*",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="total number of machines",
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--pretraining-augmentations",
        action="store_true",
        help="Whether to use the extended augmentation pipeline.",
    )
    parser.add_argument(
        "--train-image-root",
        type=str,
        help="Path to the training image root, depending on the saved COCO file, it may already be included in the annotation file.",
    )

    parser.add_argument(
        "--train-annotation-path",
        type=str,
        help="Path to the training annotation file, make sure it is in COCO format and the annotations are saved in the RLE format not polygons!",
    )
    parser.add_argument(
        "--dist-url",
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command."
        "For Yacs configs, use space-separated 'PATH.KEY VALUE' pairs"
        "For python-based LazyConfig, use 'path.key=value'",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def parse_pretrain_seg_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="number of gpus *per machine*",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="total number of machines",
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--pretraining-augmentations",
        action="store_true",
        help="Whether to use the extended augmentation pipeline.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Path to directory with all the datasets in it.",
    )
    parser.add_argument(
        "--dist-url",
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command."
        "For Yacs configs, use space-separated 'PATH.KEY VALUE' pairs"
        "For python-based LazyConfig, use 'path.key=value'",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def parse_postprocessing_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=np.random.randint(0, 2**16 - 1),
    )

    return parser.parse_args()
