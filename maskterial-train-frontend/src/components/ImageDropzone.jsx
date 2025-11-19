import { Group, Text } from "@mantine/core";
import { Dropzone, IMAGE_MIME_TYPE } from "@mantine/dropzone";
import styles from "./ImageDropzone.module.css";

export function ImageDropzone({ handleImageUpload, ...props }) {
  return (
    <Dropzone
      onDrop={(files) => handleImageUpload(files)}
      onReject={(files) => console.log("rejected files", files)}
      maxSize={5 * 1024 ** 2}
      accept={IMAGE_MIME_TYPE}
      {...props}
    >
      <Group
        justify="center"
        gap="xl"
        style={{ pointerEvents: "none" }}
        className={styles.dropzoneGroup}
      >
        <Dropzone.Accept></Dropzone.Accept>
        <Dropzone.Reject></Dropzone.Reject>
        <Dropzone.Idle></Dropzone.Idle>

        <div>
          <Text size="xl" inline>
            Drag a flake image here or click to select files
          </Text>
          <Text size="sm" c="dimmed" inline mt={7}>
            The image should be a .jpg, .jpeg, or .png file
          </Text>
        </div>
      </Group>
    </Dropzone>
  );
}
