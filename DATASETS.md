# Datasets

All Datasets are available on the [Zenodo Repository](https://zenodo.org/records/15765514).

## Datasets Overview

### Real Datasets

These datasets contain real images of 2D material flakes, annotated with their respective classes and layer counts.  
The test and train images are form multiple different exfoliation runs, which are used to ensure that no data leakage occurs between the training and testing sets.

| Dataset   | Training Images | Testing Images | Annotated Flakes |                                     Zenodo Link                                     |  
| --------- | --------------: | -------------: | ---------------- | :---------------------------------------------------------------------------------: |
| GrapheneL |             425 |           1362 | 1 to 4 layers    | [Download](https://zenodo.org/records/15765514/files/Real_GrapheneL.zip?download=1) |
| GrapheneM |             325 |            357 | 1 to 4 layers    | [Download](https://zenodo.org/records/15765514/files/Real_GrapheneM.zip?download=1) |
| GrapheneH |             438 |            480 | 1 to 4 layers    | [Download](https://zenodo.org/records/15765514/files/Real_GrapheneH.zip?download=1) |
| WSe2      |              97 |             99 | 1 to 3 layers    |   [Download](https://zenodo.org/records/15765514/files/Real_WSe2.zip?download=1)    |
| WSe2L     |              92 |            420 | 1 to 3 layers    |   [Download](https://zenodo.org/records/15765514/files/Real_WSe2L.zip?download=1)   |  
| MoSe2     |              97 |             63 | 1 to 2 layers    |   [Download](https://zenodo.org/records/15765514/files/Real_MoSe2.zip?download=1)   |  
| WS2       |              53 |             94 | 1 layer          |    [Download](https://zenodo.org/records/15765514/files/Real_WS2.zip?download=1)    |
| hBN_Thin  |              73 |             62 | 1 to 3 layers    | [Download](https://zenodo.org/records/15765514/files/Real_hBN_Thin.zip?download=1)  |

### Synthetic Datasets

These images are synthetically generated using the [MaskTerial Synthetic Data Generator](https://github.com/Jaluus/Maskterial-Synthetic-Data).  
Check out that repository for more information on how to generate your own synthetic data.

| Dataset  | Training Images | Testing Images | Annotated Flakes |                                       Zenodo Link                                       |
| -------- | --------------: | -------------: | ---------------- | :-------------------------------------------------------------------------------------: |
| Graphene |           42274 |            100 | 1 to 10 layers   | [Download](https://zenodo.org/records/15765514/files/Synthetic_Graphene.zip?download=1) |  
| CrI3     |           40338 |            100 | 1 to 10 layers   |   [Download](https://zenodo.org/records/15765514/files/Synthetic_CrI3.zip?download=1)   |
| hBN      |           43511 |            100 | 1 to 10 layers   |   [Download](https://zenodo.org/records/15765514/files/Synthetic_hBN.zip?download=1)    |
| MoSe2    |           41645 |            100 | 1 to 10 layers   |  [Download](https://zenodo.org/records/15765514/files/Synthetic_MoSe2.zip?download=1)   |
| TaS2     |           42850 |            100 | 1 to 10 layers   |   [Download](https://zenodo.org/records/15765514/files/Synthetic_TaS2.zip?download=1)   |
| WS2      |           42451 |            100 | 1 to 10 layers   |   [Download](https://zenodo.org/records/15765514/files/Synthetic_WS2.zip?download=1)    |
| WSe2     |           41146 |            100 | 1 to 10 layer    |   [Download](https://zenodo.org/records/15765514/files/Synthetic_WSe2.zip?download=1)   |

## Structure of the MaskTerial Dataset

A MaskTerial Dataset follows the structure below:

```markdown
GrapheneH
├───meta_data
│ ├───test_set_name_to_uuid.json
│ └───train_set_name_to_uuid.json
├───RLE_annotations
│ ├───1_shot
│ │ ├───run_0
│ │ │ ├───train_annotations_300.json
│ │ │ └───train_annotations_with_class_300.json
│ │ ├───run_1
│ │ │ ├───train_annotations_300.json
│ │ │ └───train_annotations_with_class_300.json
│ │ ├───...
│ │ └───run_9
│ │ ├───train_annotations_300.json
│ │ └───train_annotations_with_class_300.json
│ ├───2_shot
│ │ └───[similar run_0 to run_9 structure]
│ ├───3_shot
│ │ └───[similar run_0 to run_9 structure]
│ ├───5_shot
│ │ └───[similar run_0 to run_9 structure]
│ ├───10_shot
│ │ └───[similar run_0 to run_9 structure]
│ ├───test_annotations_300.json
│ ├───test_annotations_with_class_200.json
│ ├───test_annotations_with_class_300.json
│ ├───test_annotations_with_class_full.json
│ ├───train_annotations_300.json
│ ├───train_annotations_with_class_200.json
│ ├───train_annotations_with_class_300.json
│ └───train_annotations_with_class_full.json
├───test_images
| └───[images for testing]
├───test_semantic_masks
│ └───[semantic masks for testing]
├───train_images
│ └───[images for training]
└───train_semantic_masks
│ └───[semantic masks for training]
```

The RLE_annotations folder contains annotation files in the [COCO format](https://cocodataset.org/#format-data), the suffix `_200` indicates the minimum number of pixels used in that file, the annotation files with the `_full` suffix contain all possible flake instances but is not used during evaluation, the evaluation uses the `_300` file.

The difference between the `train_annotations_with_class_300.json` and `train_annotations_300.json` files is that the former contains the class information for each instance, while the latter only contains information about if the given instance is a flake or not. The `test_annotations_with_class_300.json` and `test_annotations_300.json` files serve the same purpose for the test set.
This is used to ablate the model's performance when given the class information or not.

There are also the `1_shot`, `2_shot`, `3_shot`, `5_shot`, and `10_shot` folders, which contain the annotations for the respective number of shots. Each of these folders contains runs from `run_0` to `run_9`, which are used for training with different random seeds and to check the stability of the model. The `train_annotations_with_class_300.json` file contains the annotations for the training set with classes, while the `train_annotations_300.json` file contains the annotations without classes.

Furthermore the instances described in the COCO annotation file are transcribed as a semantic mask in the semantic mask folder.
The pixel values in the semantic masks correspond to the class of the instance, with `0` representing the background and `1`, `2`, `3`, and so on representing the different classes of flakes. This is why the semantic masks may look black, as the pixel values are very low.

Please note, the provided semantic masks only include instances with an area **larger than 200px**. Given the images are captured with a 20x Objective, this equates to approximately 30μm² in size.

## Annotation Process

All images where annotated using the [labelme](https://github.com/wkentaro/labelme) tool to generate polygon annotations, which were then converted to RLE annotations.
It is also possible to use [labelstudio](https://labelstud.io/) to annotate the images, as it supports the COCO format and can be used to generate RLE annotations.
You just need to make sure that your final annotations are in the COCO format and are using **RLE-encoded masks, not polygon annotations**.
