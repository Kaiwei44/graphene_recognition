import styles from "./UploadModelPage.module.css";
import { Paper, Tabs } from "@mantine/core";
import { SegUploadControls } from "../components/SegUploadControls";
import { ClsUploadControls } from "../components/ClsUploadControls";

export function UploadModelPage() {
  return (
    <div className="MainSection-Fix">
      <Paper
        padding="md"
        shadow="xs"
        withBorder
        className={styles.ControlPaper}
      >
        <Tabs defaultValue="seg">
          <Tabs.List grow>
            <Tabs.Tab value="seg">Segmentation Model</Tabs.Tab>
            <Tabs.Tab value="cls">Classification Model</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="seg">
            <SegUploadControls />
          </Tabs.Panel>
          <Tabs.Panel value="cls">
            <ClsUploadControls />
          </Tabs.Panel>
        </Tabs>
      </Paper>
    </div>
  );
}
