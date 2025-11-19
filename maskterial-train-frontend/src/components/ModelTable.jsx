import { Button, Paper, Group } from "@mantine/core";
import styles from "./ModelTable.module.css";

export function ModelTable(props) {
  let models = props.models;

  const clsRows = Object.entries(models)
    .map(([model, materials]) => {
      return materials.map((material) => {
        return (
          <tr key={material + model}>
            <td>{model}</td>
            <td>{material}</td>
            <td>
              <Group grow>
                <Button
                  size="xs"
                  variant="light"
                  color="blue"
                  onClick={() =>
                    props.clickedButton(model, material, "download")
                  }
                >
                  Download
                </Button>
                <Button
                  size="xs"
                  variant="light"
                  color="red"
                  onClick={() => props.clickedButton(model, material, "delete")}
                >
                  Delete
                </Button>
              </Group>
            </td>
          </tr>
        );
      });
    })
    .flat();
  return (
    <div className={styles.ModelTable}>
      <Paper padding="md" shadow="xs" withBorder>
        <table>
          <thead>
            <tr>
              <th>Model Type</th>
              <th>Model Name</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>{clsRows}</tbody>
        </table>
      </Paper>
    </div>
  );
}
