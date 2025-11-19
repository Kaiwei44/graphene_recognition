import { createRoot } from "react-dom/client";
import { MantineProvider } from "@mantine/core";
import { Notifications } from "@mantine/notifications";
import "./index.css";
import App from "./App.jsx";
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "@mantine/notifications/styles.css";

createRoot(document.getElementById("root")).render(
  <MantineProvider>
    <Notifications />
    <App />
  </MantineProvider>
);
