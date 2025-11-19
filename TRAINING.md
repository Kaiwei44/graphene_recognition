# Training the Models

This Guide provides instructions on how to train or finetune the segmentation, classification, and postprocessing models used in the MaskTerial project.

To train the models on your own data, you need to prepare your dataset in the COCO format.
You can check out the [Dataset Guide](DATASETS.md) for more information on how to prepare your dataset.
We also provide some notebooks in the `/tools` directory to help you convert your dataset to the COCO format and annotate your data.

## Segmentation Models

To finetune the segmentation model you can run the `finetune_segmentation_model.py` script with the following arguments:

| Argument                      | Type   | Required | Default | Description                                                                                           |
| ----------------------------- | ------ | -------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `--config-file`               | string | No       | ""      | Path to config file specifying model parameters                                                       |
| `--pretraining-augmentations` | flag   | No       | False   | Whether to use the extended augmentation pipeline                                                     |
| `--train-image-root`          | string | **Yes**  | -       | Path to the training image root directory                                                             |
| `--train-annotation-path`     | string | **Yes**  | -       | Path to the training annotation file in COCO format (annotations must be in RLE format, not polygons) |
| `--resume`                    | flag   | No       | False   | Whether to attempt to resume from the checkpoint directory                                            |
| `--num-gpus`                  | int    | No       | 1       | Number of GPUs per machine                                                                            |
| `--num-machines`              | int    | No       | 1       | Total number of machines for distributed training                                                     |
| `--machine-rank`              | int    | No       | 0       | The rank of this machine (unique per machine)                                                         |
| `--dist-url`                  | string | No       | auto    | Initialization URL for PyTorch distributed backend                                                    |
| `optional final`              | list   | No       | None    | Modify config options at the end of the command (space-separated "PATH.KEY VALUE" pairs)              |

**Notes:**

- The `--resume` flag attempts to resume training from existing checkpoints in the save directory
- For distributed training across multiple GPUs or machines, adjust `--num-gpus`, `--num-machines`, and `--machine-rank` accordingly
- The `--dist-url` uses a deterministic port calculation to avoid conflicts
- Append key value pairs to override specific configuration values in the config file without modifying the config file, i.e. the weights path or output directory
- Use `OUTPUT_DIR` to specify the output directory where the model checkpoints and logs will be saved
- Annotation files must be in COCO format with RLE-encoded masks, not polygon annotations

### Segmentation Model Training Examples

You may need to change the `OUTPUT_DIR` and `MODEL.WEIGHTS` paths to point to your desired output directory and the pretrained model, respectively.
You can download the pretrained model from the link in the [Model Zoo](./MODELZOO.md).

Finetuning on a given dataset (e.g., GrapheneH) with a pretrained model:

```bash
python finetune_segmentation_model.py \
  --config-file configs/M2F/base_config.yaml \
  --train-image-root data/datasets/GrapheneH/train_images \
  --train-annotation-path data/datasets/GrapheneH/RLE_annotations/train_annotations_300.json \
  OUTPUT_DIR data/models/GrapheneH \
  MODEL.WEIGHTS path/to/the/pretained/model_final.pth
  # SOLVER.IMS_PER_BATCH 16 // You may want to change this to a lower value if you have less than 16GB GPU memory
  # SOLVER.BASE_LR 0.00001 // If you change the Batch Size, you may also want to change the learning rate
```

### Pretrain from Scratch

You can also train the segmentation model from scratch using the `pretrain_segmentation_model.py` script, which uses as similar interface as the finetuning script but is designed for pretraining on synthetic datasets.
But beware, the model was trained on 8 V100 32GB GPUs for 3 days, so this will NOT work on a single GPU.
You can also change the `SOLVER.IMS_PER_BATCH` to a lower value, but your final model performance will likely be different.
Check out the config to change the different values.
Make sure you downloaded **all** the synthetic datasets unzipped them and put them in the `--dataset-root` directory.

```bash
python pretrain_segmentation_model.py \
  --config-file configs/M2F/pretraining_config.yaml \
  --dataset-root /path/to/the/synthetic_datasets \
  --pretraining-augmentations \
  OUTPUT_DIR data/models/pretrained_model \
  SOLVER.IMS_PER_BATCH 52 \
  SOLVER.BASE_LR 0.0001
```

## Classification Models

The training of the Classification models follows the same inteface for all (AMM, GMM, MLP) models

| Argument                  | Description                                                                                     | Example Value                          |
| ------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------- |
| `--config`                | The path to the config file specifying the parameters                                           | `configs/WSe2/AMM/default_config.json` |
| `--save-dir`              | The directory where the trained model should be saved to                                        | `output/AMM_WSe2`                      |
| `--train-image-dir`       | The directory of the training images                                                            | `data/WSe2/test_masks`                 |
| `--train-annotation-path` | The directoy or path of the train annotation file in COCO format or the binary masks            | `data/WSe2/train_masks`                |
| `--test-image-dir`        | The directory of the test images (Optional)                                                     | `data/WSe2/test_images`                |
| `--test-annotation-path`  | The directoy or path of the train annotation file in COCO format or the binary masks (Optional) | `data/WSe2/test_masks`                 |
| `--train-seed`            | The seed for the random sampling, for reproducability                                           | `42`                                   |

One can then run the following scripts to train the respective models

```bash
python train_AMM_head.py # Train the AMM
python train_MLP_head.py # Train the MLP (AMM without normalization and Gaussian Predicitons)
python train_GMM_head.py # Train the GMM
```

There is also an interactive notebook `train_AMM_interactive.ipynb` that allows you to train the AMM model interactively, which is useful for debugging and testing purposes.

### Classification Model Training Examples

Training the AMM

```bash
python train_AMM_head.py \
  --config configs/AMM/default_config.json \
  --train-image-dir data/GrapheneH/train_images \
  --train-annotation-path data/GrapheneH/RLE_annotations/train_annotations_with_class_300.json \
  --save-dir models/classification_models/AMM/GrapheneH \
  --train-seed 42
```

Training the MLP on the GrapheneH Dataset with a seed of 42

```bash
python train_MLP_head.py \
  --config configs/MLP/default_config.json \
  --train-image-dir data/GrapheneH/train_images \
  --train-annotation-path data/GrapheneH/RLE_annotations/train_annotations_with_class_300.json \
  --save-dir models/classification_models/MLP/GrapheneH \
  --train-seed 42
```

Training the GMM on the GrapheneH Dataset with a seed of 42

```bash
python train_MLP_head.py \
  --config configs/MLP/default_config.json \
  --train-image-dir data/GrapheneH/train_images \
  --train-annotation-path data/GrapheneH/RLE_annotations/train_annotations_with_class_300.json \
  --save-dir models/classification_models/MLP/GrapheneH \
  --train-seed 42
```

## Postprocessing Models

For this we only use the simple L2 Postprocessing model from the paper [An open-source robust machine learning platform for real-time detection and classification of 2D material flakes](https://iopscience.iop.org/article/10.1088/2632-2153/ad2287).
You need to download the training from their [Repository](https://github.com/Jaluus/2DMatGMM).  
[This is a direct download link for the dataset](https://zenodo.org/records/8042835/files/FP_Detector_Dataset.zip?download=1)

To train the model you can run the `train_L2_postprocessing_model.py` script with the following arguments

| Argument       | Description                                              | Example Value                           |
| -------------- | -------------------------------------------------------- | --------------------------------------- |
| `--config`     | The path to the config file specifying the parameters    | `configs/L2/default_config.json`        |
| `--save-dir`   | The directory where the trained model should be saved to | `models/postprocessing_models/L2_Model` |
| `--data-dir`   | The directory of the training data                       | `data/FP_Dataset`                       |
| `--train-seed` | The seed for the random sampling, for reproducability    | `42`                                    |

### Example Training L2 Model

```bash
python train_L2_postprocessing_model.py \
  --config configs/L2/default_config.json  \
  --data-dir data/FP_Dataset \
  --save-dir models/postprocessing_models/L2 \
  --train-seed 42
```
