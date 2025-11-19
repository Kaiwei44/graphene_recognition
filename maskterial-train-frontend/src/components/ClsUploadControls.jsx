import { TextInput, Select, FileInput, Button } from "@mantine/core";
import styles from "./ClsUploadControls.module.css";
import { useState } from "react";
import { notifications } from "@mantine/notifications";

export function ClsUploadControls() {
  const [loading, setLoading] = useState(false);
  const [modelType, setModelType] = useState("GMM");
  const [modelName, setModelName] = useState("");

  const [gmmContrastFile, setGmmContrastFile] = useState(null);

  const [ModelNameError, setModelNameError] = useState(false);

  const [ammLocFile, setAmmLocFile] = useState(null);
  const [ammCovFile, setAmmCovFile] = useState(null);
  const [ammMetaFile, setAmmMetaFile] = useState(null);
  const [ammModelFile, setAmmModelFile] = useState(null);

  const [GmmContrastFileError, setGmmContrastFileError] = useState(false);
  const [AmmLocFileError, setAmmLocFileError] = useState(false);
  const [AmmCovFileError, setAmmCovFileError] = useState(false);
  const [AmmMetaFileError, setAmmMetaFileError] = useState(false);
  const [AmmModelFileError, setAmmModelFileError] = useState(false);

  const uploadModel = () => {
    setModelNameError(!modelName);
    if (!modelName) {
      return;
    }

    setLoading(true);
    let form = new FormData();
    let uploadUrl = "";

    if (modelType === "GMM") {
      setGmmContrastFileError(!gmmContrastFile);
      if (!gmmContrastFile) {
        setLoading(false);
        return;
      }
      form.append("model_name", modelName);
      form.append("contrast_file", gmmContrastFile, gmmContrastFile.name);
      uploadUrl = import.meta.env.VITE_GMM_UPLOAD_URL;
    } else {
      setAmmLocFileError(!ammLocFile);
      setAmmCovFileError(!ammCovFile);
      setAmmMetaFileError(!ammMetaFile);
      setAmmModelFileError(!ammModelFile);
      if (!ammLocFile || !ammCovFile || !ammMetaFile || !ammModelFile) {
        setLoading(false);
        return;
      }
      form.append("model_name", modelName);
      form.append("loc_file", ammLocFile, ammLocFile.name);
      form.append("cov_file", ammCovFile, ammCovFile.name);
      form.append("metadata_file", ammMetaFile, ammMetaFile.name);
      form.append("weights_file", ammModelFile, ammModelFile.name);
      uploadUrl = import.meta.env.VITE_AMM_UPLOAD_URL;
    }

    // upload the model
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
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const AMM_UPLOAD_FRAGMENT = (
    <>
      <div className={styles.inputDiv}>
        <FileInput
          withAsterisk
          label="Model Metadata File"
          description="The meta_data.json file of the model"
          placeholder=".json File"
          accept=".json"
          value={ammMetaFile}
          onChange={setAmmMetaFile}
          error={AmmMetaFileError ? "Meta File is required" : false}
        />
      </div>
      <div className={styles.inputDiv}>
        <FileInput
          withAsterisk
          label="Model Location File"
          description="The loc.npy file of the model"
          placeholder=".npy File"
          accept=".npy"
          value={ammLocFile}
          onChange={setAmmLocFile}
          error={AmmLocFileError ? "Loc File is required" : false}
        />
      </div>
      <div className={styles.inputDiv}>
        <FileInput
          withAsterisk
          label="Model Covariance File"
          description="The cov.npy file of the model"
          placeholder=".npy File"
          accept=".npy"
          value={ammCovFile}
          onChange={setAmmCovFile}
          error={AmmCovFileError ? "Cov File is required" : false}
        />
      </div>
      <div className={styles.inputDiv}>
        <FileInput
          withAsterisk
          label="Model Weights File"
          description="The model.pth file of the model"
          placeholder=".pth File"
          accept=".pth"
          value={ammModelFile}
          onChange={setAmmModelFile}
          error={AmmModelFileError ? "Model File is required" : false}
        />
      </div>
    </>
  );

  const GMM_UPLOAD_FRAGMENT = (
    <>
      <div className={styles.inputDiv}>
        <FileInput
          withAsterisk
          label="Model Contrast Definition File"
          description="The contrast_dict.json file of the model"
          placeholder=".json File"
          accept=".json"
          value={gmmContrastFile}
          onChange={setGmmContrastFile}
          error={GmmContrastFileError ? "Contrast File is required" : false}
        />
      </div>
    </>
  );

  return (
    <div className={styles.outerDiv}>
      <div className={styles.controlDiv}>
        <div className={styles.inputDiv}>
          <Select
            data={["GMM", "AMM"]}
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
        {modelType === "GMM" ? GMM_UPLOAD_FRAGMENT : AMM_UPLOAD_FRAGMENT}
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
