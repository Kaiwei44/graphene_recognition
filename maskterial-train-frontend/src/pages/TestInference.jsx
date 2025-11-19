import styles from "./TestInference.module.css";
import { useState, useEffect } from "react";
import { ImageDropzone } from "../components/ImageDropzone";
import { CanvasImage } from "../components/CanvasImage";
import { Paper, Select, Button } from "@mantine/core";
import { notifications } from "@mantine/notifications";

const formatModelData = (data) => {
  return Object.keys(data).reduce((acc, model) => {
    const materials = data[model];
    return [...acc, ...materials.map((material) => `${model}-${material}`)];
  }, []);
};

export function TestInference() {
  const [isLoading, setIsLoading] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentImageURL, setCurrentImageURL] = useState(null);
  const [availableSegModels, setAvailableSegModels] = useState([]);
  const [availableClsModels, setAvailableClsModels] = useState([]);
  const [availablePPModels, setAvailablePPModels] = useState([]);
  const [selectedSegModel, setSelectedSegModel] = useState(null);
  const [selectedClsModel, setSelectedClsModel] = useState(null);
  const [selectedPPModel, setSelectedPPModel] = useState(null);
  const [inferenceResults, setInferenceResults] = useState([]);

  const handleUserImageInput = (files) => {
    setCurrentImageURL(URL.createObjectURL(files[0]));
    setCurrentImage(files[0]);
  };

  const runInference = () => {
    if (isLoading) {
      return;
    }

    if (!currentImage) {
      return;
    }

    if (!selectedSegModel && !selectedClsModel) {
      return;
    }

    setIsLoading(true);

    // send a POST request to the server with the selected models and the image
    let formData = new FormData();
    formData.append("files", currentImage, currentImage.name);
    if (selectedSegModel) {
      formData.append("segmentation_model", selectedSegModel);
    }
    if (selectedClsModel) {
      formData.append("classification_model", selectedClsModel);
    }
    if (selectedPPModel) {
      formData.append("postprocessing_model", selectedPPModel);
    }
    formData.append("score_threshold", 0.3);
    formData.append("return_bbox", true);

    fetch(import.meta.env.VITE_INFERENCE_URL, {
      method: "POST",
      body: formData,
    })
      .then(async (response) => {
        const status = response.status;
        const data = await response.json();
        return { status, data };
      })
      .then(({ status, data }) => {
        console.log(data);
        if (status === 200) {
          notifications.show({
            title: "Success",
            message: "Results received successfully",
            color: "blue",
            autoClose: false,
          });
          setInferenceResults(data);
        } else {
          notifications.show({
            title: "Error",
            message: `Hmm, something went wrong: ${data.detail}`,
            color: "red",
            autoClose: false,
          });
        }
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  useEffect(() => {
    fetch(import.meta.env.VITE_AVAILABLE_MODELS_URL)
      .then((response) => response.json())
      .then((data) => {
        setAvailableSegModels(
          formatModelData(data.available_models.segmentation_models)
        );
        setAvailableClsModels(
          formatModelData(data.available_models.classification_models)
        );
        setAvailablePPModels(
          formatModelData(data.available_models.postprocessing_models)
        );
      });
  }, []);

  const ImageSection = (
    <>
      {!currentImage && (
        <div className={styles.dropzoneContainer}>
          <ImageDropzone
            handleImageUpload={handleUserImageInput}
            className={styles.dropzone}
          />
        </div>
      )}
      {currentImageURL && (
        <CanvasImage src={currentImageURL} flakes={inferenceResults} />
      )}
    </>
  );

  const controlSection = (
    <Paper p="md" shadow="xs" withBorder>
      <Select
        data={availableSegModels}
        label="Segmentation Model"
        placeholder="None"
        value={selectedSegModel}
        onChange={setSelectedSegModel}
        clearable
      />
      <Select
        data={availableClsModels}
        label="Classification Model"
        placeholder="None"
        value={selectedClsModel}
        onChange={setSelectedClsModel}
        clearable
      />
      <Select
        data={availablePPModels}
        label="Postprocessing Model"
        placeholder="None"
        value={selectedPPModel}
        onChange={setSelectedPPModel}
        clearable
      />
      <Button
        color="blue"
        className={styles.inferenceButton}
        onClick={runInference}
        disabled={
          !currentImage || (!selectedSegModel && !selectedClsModel) || isLoading
        }
        loading={isLoading}
      >
        Run Inference
      </Button>
    </Paper>
  );

  return (
    <div className={styles.gridContainer}>
      <div className={styles.imageSection}>{ImageSection}</div>
      <div className={styles.controlSection}>{controlSection}</div>
    </div>
  );
}
