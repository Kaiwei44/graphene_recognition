import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from pycocotools.mask import decode

URL = "http://localhost:8000/api/predict"
IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "demo",
    "images",
    "GrapheneH",
    "1fb9a0c6-da4a-4727-b9c0-473ddd7d0fd9.jpg",
)

# The Model name is "ModelType-MaterialName" and typically saved in the "ModelType/MaterialName" format.
PRED_PARAMS = {
    "segmentation_model": "M2F-GrapheneH",
    "classification_model": "AMM-GrapheneH",
    "score_threshold": 0.3,
    "min_class_occupancy": 0,
    "size_threshold": 300,
}

image = cv2.imread(IMAGE_PATH)

_, img_encoded = cv2.imencode(
    ".jpg",
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 90],
)

resp = requests.post(
    url=URL,
    files={("files", img_encoded.tobytes())},
    data=PRED_PARAMS,
)

if resp.status_code == 200:
    predicted_mask = np.zeros((1200, 1920))
    for res in resp.json():
        print(res)
        decoded_mask = decode(res["mask"])
        predicted_mask += decoded_mask

    predicted_mask_gradient = cv2.Laplacian(predicted_mask, cv2.CV_64F)
    predicted_mask_gradient = cv2.dilate(predicted_mask_gradient, None, iterations=1)
    image[predicted_mask_gradient > 0] = [0, 0, 255]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Predicted Instances")
    plt.show()
else:
    print(f"Error: {resp.status_code}")
    print(resp.text)
