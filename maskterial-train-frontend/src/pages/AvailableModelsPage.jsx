import { ModelTable } from "../components/ModelTable";
import styles from "./AvailableModelsPage.module.css";
import { Divider } from "@mantine/core";
import { useState, useEffect } from "react";
import { notifications } from "@mantine/notifications";

const fetchModels = (setter) => {
  fetch(import.meta.env.VITE_AVAILABLE_MODELS_URL)
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      setter(data.available_models);
    });
};

export function AvailableModelsPage() {
  const [availableModels, setAvailableModels] = useState(null);

  useEffect(() => {
    fetchModels(setAvailableModels);
  }, []);

  const clickedButton = (model_class, model_type, model_name, action) => {
    if (action === "download") {
      let url = new URL(import.meta.env.VITE_DOWNLOAD_MODEL_URL);
      url.searchParams.append("model_class", model_class);
      url.searchParams.append("model_type", model_type);
      url.searchParams.append("model_name", model_name);

      window.open(url, "_blank");
    } else if (action === "delete") {
      if (model_class === "postprocessing") {
        notifications.show({
          title: "Error",
          message: "Cannot delete postprocessing models, currently disabled",
          color: "red",
          autoClose: false,
        });
        return;
      }

      // ask for confirmation
      if (window.confirm(`Are you sure you want to delete ${model_name}?`)) {
        let form = new FormData();
        form.append("model_class", model_class);
        form.append("model_type", model_type);
        form.append("model_name", model_name);

        fetch(import.meta.env.VITE_DELETE_MODEL_URL, {
          method: "POST",
          body: form,
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
                message: "Model deleted successfully",
                color: "blue",
                autoClose: false,
              });
              fetchModels(setAvailableModels);
            } else {
              notifications.show({
                title: "Error",
                message: `Model deletion failed: ${data.detail}`,
                color: "red",
                autoClose: false,
              });
            }
          });
      }
    }
  };

  return (
    <div className="MainSection">
      {availableModels && (
        <>
          <div className={styles.ModelSection}>
            <Divider
              label="Classification Models"
              labelPosition="center"
              my="sm"
            />
            <ModelTable
              models={availableModels.classification_models}
              clickedButton={clickedButton.bind(this, "classification")}
            />
          </div>
          <div className={styles.ModelSection}>
            <Divider
              label="Segmentation Models"
              labelPosition="center"
              my="sm"
            />
            <ModelTable
              models={availableModels.segmentation_models}
              clickedButton={clickedButton.bind(this, "segmentation")}
            />
          </div>
          <div className={styles.ModelSection}>
            <Divider
              label="Postprocessing Models"
              labelPosition="center"
              my="sm"
            />
            <ModelTable
              models={availableModels.postprocessing_models}
              clickedButton={clickedButton.bind(this, "postprocessing")}
            />
          </div>
        </>
      )}
    </div>
  );
}
