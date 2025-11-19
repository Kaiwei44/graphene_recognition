# MaskTerial: A Foundation Model for Automated 2D Material Flake Detection

<div align="center">

[![ArXiv](https://img.shields.io/badge/ArXiv-2412.09333-b31b1b.svg)](https://arxiv.org/abs/2412.09333)
[![DataGen Demo Website](https://img.shields.io/badge/DataGen-Demo-blue)](https://datagen.uslu.tech)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14415557.svg)](https://doi.org/10.5281/zenodo.14415557)
[![BibTeX](https://img.shields.io/badge/BibTeX-gray)](#CitingMaskTerial)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wirro3JP7GG0jylQpzMKpNT5hch4FcJa?usp=sharing)

</div>

<div style="max-width: 600px; margin: auto;">
<img src="etc/hook_figure.png" alt="MaskTerial Logo" style="width: 100%; height: auto">
</div>

This repository hosts the code and related resources for the MaskTerial project, a robust Deep-Learning based model for real-time detection and classification of 2D material flakes.

The paper can be found on ArXiv: [MaskTerial: A Foundation Model for Automated 2D Material Flake Detection](https://arxiv.org/abs/2412.09333)  
For citations, please refer to the [Citing MaskTerial](#CitingMaskTerial) section below.

<!-- ArXiv Paper DOI: [10.48550/arXiv.2412.09333](https://doi.org/10.48550/arXiv.2412.09333)  
Code DOI: [10.5281/zenodo.14415557](https://doi.org/10.5281/zenodo.14415557)  
Dataset DOI: [10.5281/zenodo.15765514](https://doi.org/10.5281/zenodo.15765514)  
Pretrained Models DOI: [10.5281/zenodo.15765516](https://doi.org/10.5281/zenodo.15765516) -->

| Resource | Badge | DOI | Link |
|----------|-------|-----|------|
| Paper | [![ArXiv](https://img.shields.io/badge/ArXiv-2412.09333-b31b1b.svg)](https://arxiv.org/abs/2412.09333) | 10.48550/arXiv.2412.09333 | [ArXiv](https://arxiv.org/abs/2412.09333) |
| Code | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14415557.svg)](https://doi.org/10.5281/zenodo.14415557) | 10.5281/zenodo.14415557 | [Zenodo](https://zenodo.org/records/14415557) |
| Dataset | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15765514.svg)](https://doi.org/10.5281/zenodo.15765514) | 10.5281/zenodo.15765514 | [Zenodo](https://zenodo.org/records/15765514) |
| Pretrained Models | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15765516.svg)](https://doi.org/10.5281/zenodo.15765516) | 10.5281/zenodo.15765516 | [Zenodo](https://zenodo.org/records/15765516) |

## Abstract

The detection and classification of exfoliated two-dimensional (2D) material flakes from optical microscope images can be automated using computer vision algorithms.
This has the potential to increase the accuracy and objectivity of classification and the efficiency of sample fabrication, and it allows for large-scale data collection.
Existing algorithms often exhibit challenges in identifying low-contrast materials and typically require large amounts of training data.
Here, we present a deep learning model, called MaskTerial, that uses an instance segmentation network to reliably identify 2D material flakes.
The model is extensively pre-trained using a synthetic data generator, that generates realistic microscopy images from unlabeled data.
This results in a model that can to quickly adapt to new materials with as little as 5 to 10 images.
Furthermore, an uncertainty estimation model is used to finally classify the predictions based on optical contrast.
We evaluate our method on eight different datasets comprising five different 2D materials and demonstrate significant improvements over existing techniques in the detection of low-contrast materials such as hexagonal boron nitride.

## Features

- **Robust Detection:** The model is less sensitive to camera noise, SiO$_2$ variation, and brightness differences.
- **Interpretable and Physics-based:** The model uses well understood features reported on by literature and can be easily interpreted and validated.
- **Backwards Compatible:** If needed, the GMM from the [previous paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad2287) can be used to classify the Flakes.
- **Fully Open-Source:** The code, dataset, and model are all fully open-source and available under an MIT license.
- **Extensible:** The model can be easily extended with other Deep Learning models or other feature extraction methods.
- **Pretrained Model Weights:** Pretrained model weights are available for immediate use, only small finetuning is needed.

## Repository Structure

The paper comprises two repositories each containing a part of the project:

- [**MaskTerial**](https://github.com/Jaluus/MaskTerial): The code for the model training, evaluation, and inference.
- [**Maskterial-Synthetic-Data**](https://github.com/Jaluus/Maskterial-Synthetic-Data): The code for the synthetic data generation.

## 🚀 Quickstart

For a quick demo of the model, you can use the [Colab Notebook](https://colab.research.google.com/drive/1wirro3JP7GG0jylQpzMKpNT5hch4FcJa?usp=sharing).  
We also provide an [example Jupyter Notebook](demo/demo_inference.ipynb) in the `demo` folder that demonstrates how to use the model for inference and evaluation.  
Otherwise we provide an inference server and a frontend to try out the models and see how they work through docker.

## Installation

To install and use MaskTerial, you first need to install the required dependencies.
This installation process has been tested using Python 3.12.

### Setting Up a Python Virtual Environment

For optimal compatibility, we recommend setting up a new Python virtual environment and installing the required packages within that environment.
If you're using `conda` as your package manager, you can create and activate the virtual environment using the following commands:

```shell
conda create --name maskterial python=3.12 -y
conda activate maskterial
```

### Installing Required Packages

To install the required packages, you need to run the following commands in order:

```shell
# We first install pytorch and torchvision with the appropriate CUDA version.
# Make sure you have a GPU with CUDA support.
# You can also use a newer version of torch, but we tested it with 2.5.1 on CUDA 11.8
# If you change the version, make sure to also change the MultiScaleDeformableAttention version accordingly.
# But as of writing the newset supported version is 2.5.1 for pre-built wheels.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# You can also install the CPU version of pytorch if you don't have a GPU.
# But I STRONGLY recommend using a GPU for this project.
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# We then install the pre-built MultiScaleDeformableAttention package.
# For more information visit the issue: https://github.com/facebookresearch/Mask2Former/issues/232
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder MultiScaleDeformableAttention==1.0+9b0651cpt2.5.1cu118

# When using CPU, you can install the CPU version of MultiScaleDeformableAttention.
# pip install --extra-index-url https://miropsota.github.io/torch_packages_builder MultiScaleDeformableAttention==1.0+9b0651cpt2.5.1cpu

# We then install the required packages for MaskTerial.
# Detectron2 requires Pytorch to be installed first.
pip install git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761

# You can also install the pre-built Detectron2 package if you have problems with the above command.
# CPU:
# pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+2a420edpt2.5.1cpu
# GPU:
# pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+2a420edpt2.5.1cu118

# We also need to install panopticapi and cityscapesScripts packages
pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
pip install git+https://github.com/mcordts/cityscapesScripts.git@067792b80e446e809dd5516ee80d39fa92e18269

# Then clone and install the MaskTerial repository.
git clone https://github.com/Jaluus/MaskTerial.git MaskTerial_Repo
pip install -e MaskTerial_Repo/

# If you want to use the Demo Script you may also want to install jupyter
pip install jupyter
```

### Docker Environment

We also provide a Dockerfile to build the environment with all the dependencies.
This is useful if you want to run the code in a containerized environment.

You can build the Docker image using the following command:

```shell
docker build -t maskterial:latest .
```

Once the image is built, you can run it using the following command:

```shell
docker run --gpus=all --rm -it -v /path/to/the/data/dir:/maskterial/data maskterial:latest bash
```

Make sure you set the correct path to the data directory.

### Inference Server

To run the inference server, we provide a Docker Compose file that sets up the server and the frontend.

With CUDA support (GPU):

```shell
docker compose -f docker-compose.cuda.yml up --build
```

Without CUDA support (CPU):

```shell
docker compose -f docker-compose.cpu.yml up --build
```

This will build the Docker image and start the inference server and frontend.
You can then access the frontend at `http://localhost:8080`.
It assumes that all the necessary models lie in the `/data` folder.
To change that, go the the docker compose file and change the `volumes.data_mount.device` parameter.

```dockerfile
... Other Code ...

volumes:
  data_mount:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data # Change this to the path where your data is stored
  build_mount:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./maskterial-train-frontend/dist
```

## Training

For instructions on how to train the model, please refer to the [Training Guide](TRAINING.md).

## Evaluating

For instructions on how to evaluate the model, please refer to the [Evaluation Guide](EVALUATION.md).

## Model Zoo

All Pretrained Models can be found on [Zenodo](https://zenodo.org/records/15765516).  
Also check out the [Model Zoo](MODELZOO.md) for the specific download links.

## Datasets

All Datasets can be found on [Zenodo](https://zenodo.org/records/15765514).  
For specific download links and inforamtion on the datasets check out the [Dataset Guide](DATASETS.md).  
The guide also contains information on how to use the datasets and how to prepare the datasets for training and evaluation.  
For information on how to generate synthetic data, please refer to the [MaskTerial Synthetic Data Repository](https://github.com/Jaluus/Maskterial-Synthetic-Data).

## <a name="CitingMaskTerial"></a>Citing MaskTerial

If you use any of our code, the model, dataset, generated data or for detection or any derivate thereof in your research or find the code helpful, we would appreciate a citation to the original paper.
This helps to support the development of open-source tools and datasets in our field and encourages the publication of more tools.

```bibtex
@article{uslu2024maskterial,
    title={MaskTerial: A Foundation Model for Automated 2D Material Flake Detection},
    author={Jan-Lucas Uslu and Alexey Nekrasov and Alexander Hermans and Bernd Beschoten and Bastian Leibe and Lutz Waldecker and Christoph Stampfer},
    year={2024},
    eprint={2412.09333},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.09333},
}
```

## Contact

If you encounter any issues or have questions about the project, feel free to open an issue on our GitHub repository.
This Repo is currently maintained by [Jan-Lucas Uslu](mailto:janlucas.uslu@gmail.com).
