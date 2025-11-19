import { Paper, Tabs } from "@mantine/core";
import styles from "./TrainModelPage.module.css";
import { SegTrainControls } from "../components/SegTrainControls";

export function TrainModelPage() {
  return (
    <div className="MainSection-Fix">
      <Paper padding="md" shadow="xs" withBorder>
        <Tabs defaultValue="seg">
          <Tabs.List grow>
            <Tabs.Tab value="seg">Segmentation Model</Tabs.Tab>
            <Tabs.Tab value="cls">Classification Model</Tabs.Tab>
            {/* <Tabs.Tab value="pp">Postprocessing Model</Tabs.Tab> */}
          </Tabs.List>

          <Tabs.Panel value="seg">
            <SegTrainControls />
          </Tabs.Panel>
          <Tabs.Panel value="cls"></Tabs.Panel>
          {/* <Tabs.Panel value="pp"></Tabs.Panel> */}
        </Tabs>
      </Paper>
    </div>
  );
}
