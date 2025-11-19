# Evaluation

To evaluate the models, you can use the following command.
The evaluation script will load the model and perform inference on the test images.
The results will be saved in the specified output directory.

## Evaluation Script Parameters

| Parameter           | Type   | Required | Default        | Description                                                          |
| ------------------- | ------ | -------- | -------------- | -------------------------------------------------------------------- |
| `--cls-model-type`  | string | No       | None           | Classification model type to evaluate (options: "AMM", "GMM", "MLP") |
| `--cls-model-root`  | string | No       | None           | Path to the classification model directory                           |
| `--seg-model-type`  | string | No       | None           | Segmentation model type to evaluate (options: "M2F", "MRCNN")        |
| `--seg-model-root`  | string | No       | None           | Path to the segmentation model directory                             |
| `--pp-model-type`   | string | No       | None           | Post-processing model type to evaluate (options: "L2")               |
| `--pp-model-root`   | string | No       | None           | Path to the post-processing model directory                          |
| `--device`          | string | No       | "cuda"         | Device to run evaluation on ("cuda" or "cpu")                        |
| `--dataset-name`    | string | No       | "default"      | Name for the dataset registration                                    |
| `--annotation-path` | string | **Yes**  | -              | Path to the COCO-format annotation file                              |
| `--image-dir`       | string | **Yes**  | -              | Directory containing test images                                     |
| `--output-dir`      | string | No       | "eval_results" | Directory to save evaluation results                                 |

**Notes:**

- You must provide at least one model type (classification, segmentation, or post-processing) with its corresponding model root path.
- Both `--annotation-path` and `--image-dir` are required parameters.
- The default device is "cuda" if available; specify "cpu" if no GPU is available, else the script will error out.
- The `--dataset-name` parameter is optional and can be used to register the dataset with a custom name, which can be useful for logging or tracking purposes.

## Examples

Evaluate AMM model with Mask2Former (M2F) segmentation model on GrapheneH dataset

```shell
python evaluate_model.py \
    --cls-model-type AMM \
    --cls-model-root data/models/classification_models/AMM/GrapheneH \
    --seg-model-type M2F \
    --seg-model-root data/models/segmentation_models/M2F/GrapheneH \
    --device cuda \
    --output-dir evaluation_results/AMM_M2F_GrapheneH \
    --image-dir data/datasets/GrapheneH/test_images \
    --annotation-path data/datasets/GrapheneH/RLE_annotations/test_annotations_with_class_300.json
```

Evalutate only the GMM model on GrapheneH dataset

```shell
python evaluate_model.py \
    --cls-model-type GMM \
    --cls-model-root data/models/classification_models/GMM/GrapheneH \
    --device cpu \
    --output-dir evaluation_results/GMM_GrapheneH \
    --image-dir data/datasets/GrapheneH/test_images \
    --annotation-path data/datasets/GrapheneH/RLE_annotations/test_annotations_with_class_300.json
```

Evaluate the GMM model with the L2 Post-Processing Model on WS2 dataset

```shell
python evaluate_model.py \
    --cls-model-type GMM \
    --cls-model-root data/models/classification_models/GMM/WSe2 \
    --pp-model-type L2 \
    --pp-model-root data/models/postprocessing_models/L2 \
    --device cuda \
    --output-dir evaluation_results/GMM_L2PP_WSe2 \
    --image-dir data/datasets/WSe2/test_images \
    --annotation-path data/datasets/WSe2/RLE_annotations/test_annotations_with_class_300.json
```

Evaluate AMM model with Mask2Former (M2F) segmentation model on WSe2 dataset

```shell
python evaluate_model.py \
    --cls-model-type AMM \
    --cls-model-root data/models/classification_models/AMM/WSe2 \
    --seg-model-type M2F \
    --seg-model-root data/models/segmentation_models/M2F/WSe2 \
    --device cuda \
    --output-dir evaluation_results/AMM_M2F_WSe2 \
    --image-dir data/datasets/WSe2/test_images \
    --annotation-path data/datasets/WSe2/RLE_annotations/test_annotations_with_class_300.json
```
