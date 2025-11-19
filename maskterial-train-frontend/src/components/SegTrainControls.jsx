import {
  TextInput,
  Select,
  FileInput,
  Button,
  Paper,
  Text,
} from "@mantine/core";
import styles from "./SegTrainControls.module.css";
import { useState } from "react";

export function SegTrainControls() {
  const [consoleOutput, setConsoleOutput] = useState("Standing By...");
  const [loading, setLoading] = useState(false);

  const [modelType, setModelType] = useState("Mask2Former");
  const [modelName, setModelName] = useState("");
  const [datasetFile, setDatasetFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);

  const [ModelNameError, setModelNameError] = useState(false);
  const [DatasetFileError, setDatasetFileError] = useState(false);

  const startTraining = () => {
    setModelNameError(!modelName);
    setDatasetFileError(!datasetFile);
    if (!modelName || !datasetFile) {
      return;
    }
    setConsoleOutput("Starting Training...");
    setLoading(true);

    console.log("Starting Training...");
    console.log("Model Type:", modelType);
    console.log("Model Name:", modelName);
    console.log("Dataset File:", datasetFile);
    console.log("Config File:", configFile);

    if (modelType === "Mask2Former") {
      console.log("Training Mask2Former Model...");

      let form = new FormData();
      form.append("model_name", modelName);
      form.append("dataset_file", datasetFile);
      if (configFile) {
        form.append("config_file", configFile);
      }

      fetch(import.meta.env.VITE_M2F_TRAIN_URL, {
        method: "POST",
        body: form,
      })
        .then((response) => {
          const reader = response.body.getReader(); // Create a reader to read the stream
          const decoder = new TextDecoder("utf-8"); // Decode the stream chunks as UTF-8

          const processStream = async () => {
            let textBuffer = ""; // Buffer to hold partial data

            while (true) {
              const { done, value } = await reader.read();
              if (done) break; // Exit the loop when the stream is complete

              const chunk = decoder.decode(value, { stream: true }); // Decode the chunk
              textBuffer += chunk; // Append chunk to the buffer

              // Process and display complete responses if any
              const lines = textBuffer.split("\n"); // Assuming responses are newline-separated
              textBuffer = lines.pop(); // Keep the last incomplete line in the buffer

              for (const line of lines) {
                if (line.trim()) {
                  console.log("Received:", line); // Log each line
                  setConsoleOutput((prev) => prev + "\n" + line);
                }
              }
            }

            // Log any leftover data
            if (textBuffer.trim()) {
              console.log("Final data:", textBuffer);
              setConsoleOutput((prev) => prev + "\n" + textBuffer);
            }
          };

          return processStream();
        })
        .catch((error) => {
          console.error("Error:", error);
          setConsoleOutput("Error occurred while streaming data.");
        })
        .finally(() => {
          setLoading(false);
        });
    }
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
            label="Dataset File"
            description="The ZIP File downloaded from LabelStudio"
            placeholder=".zip File"
            accept=".zip"
            value={datasetFile}
            onChange={setDatasetFile}
            error={DatasetFileError ? "Dataset File is required" : false}
          />
        </div>
        <Paper
          shadow="xs"
          radius="md"
          withBorder
          p="xs"
          className={styles.dangerzone}
        >
          <Text fw={700}>Danger Zone</Text>
          <div className={styles.inputDiv}>
            <FileInput
              label="Config File"
              description="The Custom Train Config File, ONLY USE THIS IF YOU REALLY KNOW WHAT YOU'RE DOING."
              placeholder=".yaml File / .yml File"
              accept=".yaml,.yml"
              value={configFile}
              onChange={setConfigFile}
            />
          </div>
        </Paper>
        <div className={styles.inputDiv}>
          <Button
            color="blue"
            fullWidth
            onClick={startTraining}
            loading={loading}
          >
            Train Model
          </Button>
        </div>
      </div>
      <div className={styles.consoleOutput}>
        <pre>{consoleOutput}</pre>
      </div>
    </div>
  );
}
