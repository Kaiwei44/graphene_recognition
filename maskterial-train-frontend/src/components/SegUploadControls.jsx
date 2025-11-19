import { TextInput, Select, FileInput, Button } from "@mantine/core";
import styles from "./SegUploadControls.module.css";
import { useState } from "react";
import { notifications } from "@mantine/notifications";

export function SegUploadControls() {
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState("Mask2Former");
  const [modelName, setModelName] = useState("");
  const [modelFile, setModelFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);

  const [ModelNameError, setModelNameError] = useState(false);
  const [ModelFileError, setModelFileError] = useState(false);
  const [ConfigFileError, setConfigFileError] = useState(false);

  const uploadModel = () => {
    setModelNameError(!modelName);
    setModelFileError(!modelFile);
    setConfigFileError(!configFile);
    if (!modelName || !modelFile || !configFile) {
      return;
    }
    setLoading(true);
    let uploadUrl = import.meta.env.VITE_M2F_UPLOAD_URL;

    let form = new FormData();
    form.append("model_name", modelName);
    form.append("config_file", configFile, configFile.name);
    form.append("model_file", modelFile, modelFile.name);

    fetch(uploadUrl, {
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
            message: "Model uploaded successfully",
            color: "blue",
            autoClose: false,
          });
        } else {
          notifications.show({
            title: "Error",
            message: `Model upload failed: ${data.detail}`,
            color: "red",
            autoClose: false,
          });
        }
        setLoading(false);
      });
  };

  return (
    <div className={styles.outerDiv}>
      <div className={styles.controlDiv}>
        <div className={styles.inputDiv}>
          <Select
            data={["Mask2Former"]}
            label="Model Type"
            placeholder="Pick value"
            value={modelType}
            onChange={setModelType}
          />
        </div>
        <div className={styles.inputDiv}>
          <TextInput
            withAsterisk
            label="Model Name"
            description="What you want the Model to show up as later. Name should be descriptive, i.e. 'WSe2_1_to_8_layers'"
            placeholder="My Model Name"
            value={modelName}
            onChange={(event) => setModelName(event.currentTarget.value)}
            error={ModelNameError ? "Model Name is required" : false}
          />
        </div>
        <div className={styles.inputDiv}>
          <FileInput
            withAsterisk
            label="Model Weights File"
            description="The model.pth file of the model"
            placeholder=".pth File"
            accept=".pth"
            value={modelFile}
            onChange={setModelFile}
            error={ModelFileError ? "Model File is required" : false}
          />
        </div>
        <div className={styles.inputDiv}>
          <FileInput
            withAsterisk
            label="Model Config File"
            description="The config.yaml file of the model"
            placeholder=".yaml File"
            accept=".yaml"
            value={configFile}
            onChange={setConfigFile}
            error={ConfigFileError ? "Config File is required" : false}
          />
        </div>
        <div className={styles.inputDiv}>
          <Button
            color="blue"
            fullWidth
            onClick={uploadModel}
            loading={loading}
          >
            Upload Model
          </Button>
        </div>
      </div>
    </div>
  );
}
