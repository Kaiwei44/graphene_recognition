FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel AS builder

# Define the Graphics Card to use
# Taken from https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
# 6.0;6.1;6.2 | 10xx, Pxxx
# 7.0;7.2 | Titan V
# 7.5 | 16xx, 20xx, 20xxS
# 8.0;8.6;8.7 | 30xx, Axxx
# 8.9 | 40xx
# 9.0 | H100
# If you are using a different GPU, make sure you add it to the list.
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.5;8.0;8.6;8.7;8.9"
ENV FORCE_CUDA="1"

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt update

# Install some basic utilities needed for OpenCV.
RUN apt install -y wget curl ca-certificates sudo git bzip2 libx11-6 build-essential

# Install OpenCV dependencies.
RUN apt install ffmpeg libsm6 libxext6  -y

# Create a working directory.
RUN mkdir /maskterial
WORKDIR /maskterial

# Install the Deformable Attention library.
# For more information on the Precombiled Wheels go to: https://github.com/facebookresearch/Mask2Former/issues/232
RUN --mount=type=cache,target=/root/.cache/pip \ 
    pip install --extra-index-url https://miropsota.github.io/torch_packages_builder MultiScaleDeformableAttention==1.0+9b0651cpt2.5.1cu118

# We could also compile the ops needed for the Deformable Attention library from source, but it is not necessary for the current version of Maskterial.
# Uncomment the following lines to compile the ops from source.
# COPY maskterial/modeling/segmentation_models/M2F/modeling/pixel_decoder/ops /maskterial/maskterial/modeling/segmentation_models/M2F/modeling/pixel_decoder/ops
# RUN python /maskterial/maskterial/modeling/segmentation_models/M2F/modeling/pixel_decoder/ops/setup.py build install

# Install detectron2 from source.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761

COPY requirements_frozen.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements_frozen.txt,target=requirements_frozen.txt \
    python -m pip install -r requirements_frozen.txt

COPY . .