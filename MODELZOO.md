# Model Zoo

Here is a list of pretrained models and their corresponding datasets. The models are organized by the type of dataset they were trained on, and the specific model architectures used.
All models are available for download from [Zenodo](https://zenodo.org/records/15765516), and the links are provided in the tables below.

## Segmentation Models

These models are used for segmentation tasks, specifically for detection flakes of interest in 2D materials.

### Synthetic Models

This model is the starting point from which we fine-tune the models on the real datasets.

| Model Name  | Dataset   | Notes             |                               Download                               |
| ----------- | --------- | ----------------- | :------------------------------------------------------------------: |
| Mask2Former | Synthetic | ResNet50 Backbone | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_Synthetic_Data.zip?download=1) |

### Finetuned Models

All these models where trained on the synthetic dataset and then fine-tuned on the real datasets.

| Model Name  | Dataset              | Notes |                               Download                               |
| ----------- | -------------------- | ----- | :------------------------------------------------------------------: |
| Mask2Former | Graphene<sub>L</sub> |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_GrapheneL.zip?download=1) |
| Mask2Former | Graphene<sub>M</sub> |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_GrapheneM.zip?download=1) |
| Mask2Former | Graphene<sub>H</sub> |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_GrapheneH.zip?download=1) |
| Mask2Former | WSe2                 |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_WSe2.zip?download=1) |
| Mask2Former | WSe2<sub>L</sub>     |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_WSe2L.zip?download=1) |
| Mask2Former | MoSe2                |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_MoSe2.zip?download=1) |
| Mask2Former | WS2                  |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_WS2.zip?download=1) |
| Mask2Former | hBN<sub>thin</sub>   |       | [Model](https://zenodo.org/records/15765516/files/SEG_M2F_hBN_Thin.zip?download=1) |

## Classification Models

These models are used for the classification tasks, specifically for assigning the classes to the flakes detected by the segmentation models.

| Model Name | Dataset              | Notes                 |                               Download                               |
| ---------- | -------------------- | --------------------- | :------------------------------------------------------------------: |
| AMM        | Graphene<sub>L</sub> |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_GrapheneL.zip?download=1) |
| AMM        | Graphene<sub>M</sub> |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_GrapheneM.zip?download=1) |
| AMM        | Graphene<sub>H</sub> |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_GrapheneH.zip?download=1) |
| AMM        | WSe2                 |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_WSe2.zip?download=1) |
| AMM        | WSe2<sub>L</sub>     |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_WSe2L.zip?download=1) |
| AMM        | MoSe2                |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_MoSe2.zip?download=1) |
| AMM        | WS2                  |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_WS2.zip?download=1) |
| AMM        | hBN<sub>thin</sub>   |                       | [Model](https://zenodo.org/records/15765516/files/CLS_AMM_hBN_Thin.zip?download=1) |
| GMM        | Graphene<sub>L</sub> |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_GrapheneL.zip?download=1) |
| GMM        | Graphene<sub>M</sub> | Currently Unavailable |                                  -                                   |
| GMM        | Graphene<sub>H</sub> |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_GrapheneH.zip?download=1) |
| GMM        | WSe2                 |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_WSe2.zip?download=1) |
| GMM        | WSe2<sub>L</sub>     |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_WSe2L.zip?download=1) |
| GMM        | MoSe2                |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_MoSe2.zip?download=1) |
| GMM        | WS2                  |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_WS2.zip?download=1) |
| GMM        | hBN<sub>thin</sub>   |                       | [Model](https://zenodo.org/records/15765516/files/CLS_GMM_hBN_Thin.zip?download=1) |

## Post Processing Models

These models are used to postprocess the detected flakes and remove the false positives.
If you are using a segmentation model, I would recommend against using it.

| Model Name | Dataset                  | Notes             |                               Download                               |
| ---------- | ------------------------ | ----------------- | :------------------------------------------------------------------: |
| L2         | False Positive Detection | Material Agnostic | [Model](https://zenodo.org/records/15765516/files/PP_L2_material_agnostic.zip?download=1) |
