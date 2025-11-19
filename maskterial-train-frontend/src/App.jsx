import "./App.css";
import { Tabs } from "@mantine/core";
import { TrainModelPage } from "./pages/TrainModelPage";
import { AvailableModelsPage } from "./pages/AvailableModelsPage";
import { UploadModelPage } from "./pages/UploadModelPage";
import { TestInference } from "./pages/TestInference";

function App() {
  return (
    <>
      <Tabs
        defaultValue="train"
        styles={{
          list: { height: "5vh" },
          panel: { height: "95vh", overflow: "auto" },
          root: { height: "100vh" },
        }}
      >
        <Tabs.List grow>
          <Tabs.Tab value="train">Train Model</Tabs.Tab>
          <Tabs.Tab value="upload">Upload Trained Model</Tabs.Tab>
          <Tabs.Tab value="check">Check Available Models</Tabs.Tab>
          <Tabs.Tab value="test">Test Inference</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="train">
          <TrainModelPage />
        </Tabs.Panel>
        <Tabs.Panel value="upload">
          <UploadModelPage />
        </Tabs.Panel>
        <Tabs.Panel value="check">
          <AvailableModelsPage />
        </Tabs.Panel>
        <Tabs.Panel value="test">
          <TestInference />
        </Tabs.Panel>
      </Tabs>
    </>
  );
}

export default App;
