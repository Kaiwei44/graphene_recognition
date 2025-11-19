import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from maskterial.modeling.classification_models import AMM_head, GMM_head
from maskterial.maskterial import MaskTerial
from maskterial.modeling.segmentation_models import M2F_model
from maskterial.utils.inference_server_utils import (
    ServerState,
    check_available_models,
    convert_coco_polygon_to_rle,
    read_image,
    update_server_state_and_predictor,
)

# Working with lower precision improves performance marginally
torch.set_float32_matmul_precision("medium")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

server_state: ServerState | None = None
predictor: MaskTerial | None = None
currently_training = False
file_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model_dir = os.path.join(file_dir, "data", "models", "classification_models")
seg_model_dir = os.path.join(file_dir, "data", "models", "segmentation_models")
pp_model_dir = os.path.join(file_dir, "data", "models", "postprocessing_models")

pretrained_m2f_path = os.path.join(
    file_dir,
    "data",
    "models",
    "segmentation_models",
    "M2F",
    "Synthetic_Data",
    "model_final.pth",
)

available_cls_models = check_available_models(cls_model_dir)
available_seg_models = check_available_models(seg_model_dir)
available_pp_models = check_available_models(pp_model_dir)


@app.get("/")
async def check_server_state():
    if server_state is not None:
        return server_state.to_dict()

    return "No Models are loaded, try to run inference on the /predict endpoint"


@app.get("/status")
async def check_status():
    if currently_training:
        return "Currently training, try again later"
    return "Ready for inference"


@app.get("/available_models")
async def get_models():
    global available_cls_models, available_seg_models, available_pp_models
    available_cls_models = check_available_models(cls_model_dir)
    available_seg_models = check_available_models(seg_model_dir)
    available_pp_models = check_available_models(pp_model_dir)

    return {
        "available_models": {
            "classification_models": available_cls_models,
            "segmentation_models": available_seg_models,
            "postprocessing_models": available_pp_models,
        },
    }


@app.post("/upload/amm")
async def upload_amm(
    model_name: str = Form(...),
    metadata_file: UploadFile = File(...),
    loc_file: UploadFile = File(...),
    cov_file: UploadFile = File(...),
    weights_file: UploadFile = File(...),
):

    available_models = check_available_models(cls_model_dir)
    if model_name in available_models["AMM"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )

    new_model_path = os.path.join(cls_model_dir, "AMM", model_name)

    try:
        os.makedirs(new_model_path)

        loc_path = os.path.join(new_model_path, "loc.npy")
        cov_path = os.path.join(new_model_path, "cov.npy")
        model_path = os.path.join(new_model_path, "model.pth")
        meta_path = os.path.join(new_model_path, "meta_data.json")

        with open(meta_path, "wb") as f:
            f.write(metadata_file.file.read())

        with open(loc_path, "wb") as f:
            f.write(loc_file.file.read())

        with open(cov_path, "wb") as f:
            f.write(cov_file.file.read())

        with open(model_path, "wb") as f:
            f.write(weights_file.file.read())

        # Create a new model instance to check if the model is valid
        AMM_head.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/upload/gmm")
async def upload_gmm(
    model_name: str = Form(...),
    contrast_file: UploadFile = File(...),
):

    available_models = check_available_models(cls_model_dir)
    if model_name in available_models["GMM"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )

    new_model_path = os.path.join(cls_model_dir, "GMM", model_name)

    try:
        os.makedirs(new_model_path)
        contrast_dict_path = os.path.join(new_model_path, "contrast_dict.json")
        with open(contrast_dict_path, "wb") as f:
            f.write(contrast_file.file.read())

        GMM_head.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/upload/m2f")
async def upload_M2F(
    model_name: str = Form(...),
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(None),
):

    available_models = check_available_models(seg_model_dir)
    if model_name in available_models["M2F"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )
    new_model_path = os.path.join(seg_model_dir, "M2F", model_name)

    try:
        os.makedirs(new_model_path)

        config_path = os.path.join(new_model_path, "config.yaml")
        with open(config_path, "wb") as f:
            f.write(config_file.file.read())

        model_path = os.path.join(new_model_path, "model.pth")
        with open(model_path, "wb") as f:
            f.write(model_file.file.read())

        M2F_model.from_pretrained(new_model_path)

    except Exception as e:
        shutil.rmtree(new_model_path)
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")

    return f"Model {model_name} uploaded successfully"


@app.post("/delete_model")
async def delete_model(
    model_class: Literal["segmentation", "classification"] = Form(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
):
    model_dir = None
    if model_class == "segmentation":
        model_dir = seg_model_dir
    elif model_class == "classification":
        model_dir = cls_model_dir

    available_models = check_available_models(model_dir)

    # check if the model exists in the available models [model_type][model_name]
    if model_type not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type {model_type} not found for {model_class}",
        )
    if model_name not in available_models[model_type]:
        raise HTTPException(
            status_code=404,
            detail=f"Model name {model_name} not found in {model_type} for {model_class}",
        )

    model_path = os.path.join(model_dir, model_type, model_name)

    try:
        shutil.rmtree(model_path)
        return f"Model {model_name} deleted successfully"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.get("/download_model")
async def download_model(
    model_class: Literal["segmentation", "classification", "postprocessing"],
    model_type: str,
    model_name: str,
):
    model_dir = None
    if model_class == "segmentation":
        model_dir = seg_model_dir
    elif model_class == "classification":
        model_dir = cls_model_dir
    elif model_class == "postprocessing":
        model_dir = pp_model_dir

    available_models = check_available_models(model_dir)

    # check if the model exists in the available models [model_type][model_name]
    if model_type not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model type {model_type} not found for {model_class}",
        )
    if model_name not in available_models[model_type]:
        raise HTTPException(
            status_code=404,
            detail=f"Model name {model_name} not found in {model_type} for {model_class}",
        )

    model_path = os.path.join(model_dir, model_type, model_name)

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_path)
                    zip_file.write(file_path, arcname)

        # Ensure buffer's pointer is at the beginning
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={model_class}-{model_type}-{model_name}.zip"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create zip file: {str(e)}"
        )


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    segmentation_model: str | None = Form(None),
    classification_model: str | None = Form(None),
    postprocessing_model: str | None = Form(None),
    score_threshold: float = Form(0.0),
    min_class_occupancy: float = Form(0.0),
    size_threshold: int = Form(300),
    return_bbox: bool = Form(False),
):
    global server_state, predictor

    if currently_training:
        return "Currently training, try again later"

    new_server_state = ServerState(
        seg_model_name=segmentation_model,
        cls_model_name=classification_model,
        pp_model_name=postprocessing_model,
        score_threshold=score_threshold,
        min_class_occupancy=min_class_occupancy,
        size_threshold=size_threshold,
        cls_model_dir=cls_model_dir,
        seg_model_dir=seg_model_dir,
        pp_model_dir=pp_model_dir,
        device=device,
    )

    if server_state != new_server_state:
        server_state, predictor = update_server_state_and_predictor(new_server_state)

    start = time.time()
    img = read_image(files[0])
    print("Read Time:", round(time.time() - start, 3), "s")

    start = time.time()
    result = predictor.predict(img)
    print("Pred Time:", round(time.time() - start, 3), "s")

    result_dict = [r.to_dict(return_bbox=return_bbox) for r in result]
    return result_dict


@app.post("/train/m2f")
async def train(
    model_name: str = Form(...),
    dataset_file: UploadFile = File(...),
    config_file: UploadFile = File(None),
):
    global currently_training, server_state

    available_models = check_available_models(seg_model_dir)
    if model_name in available_models["M2F"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} already exists, please choose a different name",
        )
    new_model_path = os.path.join(seg_model_dir, "M2F", model_name)
    data_dir = os.path.join(new_model_path, "data")
    image_dir = os.path.join(data_dir, "images")
    ann_path = os.path.join(data_dir, "result.json")
    RLE_ann_path = os.path.join(data_dir, "result_RLE.json")

    if config_file is None:
        config_path = os.path.join("configs", "M2F", "base_config.yaml")
    else:
        config_path = os.path.join(new_model_path, "train_config.yaml")

    try:
        currently_training = True
        os.makedirs(new_model_path)
        os.makedirs(data_dir)

        with zipfile.ZipFile(dataset_file.file, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # we now need to convert the polygon annotations to RLE format
        with open(ann_path, "r") as f:
            coco_ann = convert_coco_polygon_to_rle(json.load(f))
            # we also need to strip the first images\/ from the image path
            for ann in coco_ann["images"]:
                ann["file_name"] = ann["file_name"][7:]

        with open(RLE_ann_path, "w") as f:
            json.dump(coco_ann, f)

        if config_file is not None:
            with open(config_path, "wb") as f:
                f.write(config_file.file.read())

        os.environ["WANDB_ACTIVE"] = "0"

        torch.cuda.empty_cache()
        server_state = None

        cmd = [
            sys.executable,
            "-u",
            "train_seg_model_terminal.py",
            "--config-file",
            config_path,
            "--train-image-root",
            image_dir,
            "--train-annotation-path",
            RLE_ann_path,
            "--dist-url",
            "auto",
            "OUTPUT_DIR",
            new_model_path,
            "MODEL.WEIGHTS",
            pretrained_m2f_path,
        ]

        def stream_process():
            global currently_training
            yield "Training started...\r"
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stdout and stderr
                    bufsize=1,  # Line-buffered
                    text=True,
                )
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    yield line
            except Exception as e:
                raise Exception(str(e))
            finally:
                process.stdout.close()
                process.wait()
                currently_training = False
                torch.cuda.empty_cache()

        return StreamingResponse(stream_process(), media_type="text/event-stream")

    except Exception as e:
        shutil.rmtree(new_model_path)
        currently_training = False
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
