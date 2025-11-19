import json
import os

import numpy as np
import torch

from maskterial.utils.argparser import parse_cls_args
from maskterial.utils.data_loader import ContrastDataloader

if __name__ == "__main__":
    args = parse_cls_args()

    TRAIN_SEED = args.train_seed
    SAVE_DIR = args.save_dir
    CFG_PATH = args.config
    TRAIN_IMAGE_DIR = args.train_image_dir
    TRAIN_ANNOTATION_PATH = args.train_annotation_path
    TEST_IMAGE_DIR = args.test_image_dir
    TEST_ANNOTATION_PATH = args.test_annotation_path

    with open(CFG_PATH, "r") as f:
        CONFIG = json.load(f)
        DATA_PARAMS = CONFIG["data_params"]

    # As we are sampling the data randomly, we need to set the seed to ensure reproducibility
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    dataloader = ContrastDataloader(
        train_image_dir=TRAIN_IMAGE_DIR,
        train_annotation_path=TRAIN_ANNOTATION_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        test_annotation_path=TEST_ANNOTATION_PATH,
        **DATA_PARAMS,
        verbose=True,
    )

    X_train_full = torch.tensor(dataloader.X_train).float()
    y_train_full = dataloader.y_train

    # fit one Gaussian per class
    loc = torch.stack(
        [
            torch.mean(X_train_full[y_train_full == c], dim=0)
            for c in range(dataloader.num_classes)
        ]
    )
    cov = torch.stack(
        [
            torch.cov(X_train_full[y_train_full == c].T)
            for c in range(dataloader.num_classes)
        ]
    )

    meta_data = {
        "train_config": CONFIG,
        "train_image_dir": TRAIN_IMAGE_DIR,
        "train_annotation_path": TRAIN_ANNOTATION_PATH,
        "test_image_dir": TEST_IMAGE_DIR,
        "test_annotation_path": TEST_ANNOTATION_PATH,
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "meta_data.json"), "w") as f:
        json.dump(meta_data, f, indent=4)

    np.save(os.path.join(SAVE_DIR, "loc.npy"), loc)
    np.save(os.path.join(SAVE_DIR, "cov.npy"), cov)

    # save in the old format for compatibility
    output_dict = {}
    keys = ["b", "g", "r"]
    for c in range(dataloader.num_classes):
        class_loc = loc[c].numpy().tolist()
        class_cov = cov[c].numpy().tolist()
        output_dict[str(c)] = {
            "contrast": {k: class_loc[i] for i, k in enumerate(keys)},
            "covariance_matrix": cov[c].numpy().tolist(),
        }

    with open(os.path.join(SAVE_DIR, "contrast_dict.json"), "w") as f:
        json.dump(output_dict, f, indent=4)
