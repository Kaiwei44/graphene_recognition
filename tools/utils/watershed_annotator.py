import os

import cv2
import numpy as np


class Watershed_Annotator:
    def __init__(
        self,
        image_directory,
        mask_directory,
        start_idx=0,
        image_names=None,
    ):
        self.current_class = 0
        self.current_index = start_idx

        # tab10 colors from 0 to 255
        self.class_colors = [
            [0, 0, 0],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
            [23, 190, 207],
        ]
        # bgr to rgb
        self.class_colors = [color[::-1] for color in self.class_colors]

        self.disable_mask = True
        self.overexpose = False

        self.marks_updated = False
        self.marks = []

        self.l_mouse_down = False
        self.r_mouse_down = False
        self.mm_mouse_down = False
        self.l_mouse_pressed = False
        self.r_mouse_pressed = False
        self.mm_mouse_pressed = False

        self.image_directory = image_directory
        self.mask_directory = mask_directory

        if image_names is None:
            self.image_names = [
                name
                for name in os.listdir(self.image_directory)
                if (name.endswith(".png") or name.endswith(".jpg"))
            ]
        else:
            self.image_names = image_names

        self.image_paths = [
            os.path.join(self.image_directory, name) for name in self.image_names
        ]

        assert os.path.exists(
            self.image_directory
        ), "Image directory does not exist or the path is incorrect."
        assert os.path.exists(
            self.mask_directory
        ), "Mask directory does not exist or the path is incorrect."

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.l_mouse_down:
                self.l_mouse_pressed = True
            self.l_mouse_down = True

        if event == cv2.EVENT_LBUTTONUP:
            if self.l_mouse_down:
                self.l_mouse_pressed = False
            self.l_mouse_down = False

        if event == cv2.EVENT_MBUTTONDOWN:
            if not self.mm_mouse_down:
                self.mm_mouse_pressed = True
            self.mm_mouse_down = True

        if event == cv2.EVENT_MBUTTONUP:
            if self.mm_mouse_down:
                self.mm_mouse_pressed = False
            self.mm_mouse_down = False

        if event == cv2.EVENT_RBUTTONDOWN:
            if not self.r_mouse_down:
                self.r_mouse_pressed = True
            self.r_mouse_down = True

        if event == cv2.EVENT_RBUTTONUP:
            if self.r_mouse_down:
                self.r_mouse_pressed = False
            self.r_mouse_down = False

        if event == cv2.EVENT_MOUSEMOVE:
            self.l_mouse_pressed = False
            self.r_mouse_pressed = False
            self.mm_mouse_pressed = False

        if self.mm_mouse_pressed:
            if self.current_class == 0:
                self.current_class = 1
            self.marks.append([(x, y), self.current_class])
            self.marks_updated = True

        if self.l_mouse_pressed:
            self.current_class += 1
            self.marks.append([(x, y), self.current_class])
            self.marks_updated = True

        if self.r_mouse_pressed:
            self.marks.append([(x, y), 0])
            self.marks_updated = True

    def clear_marks(self):
        self.current_class = 0
        self.marks = []
        self.marks_updated = True

    def update_current_image(self):
        self.current_image = cv2.imread(self.image_paths[self.current_index])
        self.current_image = cv2.medianBlur(self.current_image, 3)

        # increase the contrast of the image
        if self.overexpose:
            self.current_image = cv2.convertScaleAbs(
                self.current_image, alpha=3, beta=0
            )

        self.current_image_display = self.current_image.copy()
        self.current_class = 0
        self.disable_mask = True
        self.clear_marks()

    def save_mask(self):
        new_mask_name = self.image_names[self.current_index].split(".")[0] + ".png"

        if len(self.marks) == 0:
            blank_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            cv2.imwrite(os.path.join(self.mask_directory, new_mask_name), blank_mask)
            return

        watershed_mask = np.zeros(self.current_image.shape[:2], dtype=np.int32)
        # idx = 1 means background
        # everything else is foreground
        for mark in self.marks:
            watershed_mask[mark[0][1], mark[0][0]] = mark[1] + 1
        # run the watershed algorithm with the chosen markers
        cv2.watershed(self.current_image, watershed_mask)
        watershed_mask[watershed_mask == -1] = 1
        watershed_mask = watershed_mask - 1
        watershed_mask = watershed_mask.astype(np.uint8)

        cv2.imwrite(
            os.path.join(self.mask_directory, new_mask_name),
            watershed_mask,
        )

    def undo_last_mark(self):
        if len(self.marks) > 0:
            self.marks = self.marks[:-1]
            self.marks_updated = True
            if len(self.marks) > 0:
                # the marks array looks somthing like
                # [1,2,3,0,0,3,4,5,5,0,0,0,6]
                # we the delete the last element and get
                # [1,2,3,0,0,3,4,5,5,0,0,0]
                # our current class should then be the last non-zero element
                # i.e. 5, and not 0
                # using self.current_class = self.marks[-1][1] will get us 0 instead of 5
                # so we need to find the last non-zero element
                self.mark_classes = [mark[1] for mark in self.marks]
                self.mark_classes.reverse()
                self.current_class = next((x for x in self.mark_classes if x), 0)
            else:
                self.current_class = 0

    def run(self):
        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", self.mouse_callback)
        cv2.setWindowTitle("Annotator", self.image_paths[self.current_index])

        self.update_current_image()

        while True:
            # Check if a mask already exists for the current image

            mask_name = self.image_names[self.current_index].split(".")[0] + ".png"
            mask_exists = os.path.exists(os.path.join(self.mask_directory, mask_name))
            window_title = f"{self.image_names[self.current_index]} | {self.current_index + 1 }/{len(self.image_paths)} | Class: {self.current_class}"

            if mask_exists:
                window_title += " | Mask exists"

            cv2.imshow("Annotator", self.current_image_display)
            cv2.setWindowTitle(
                "Annotator",
                window_title,
            )

            key = cv2.waitKey(10)

            if key == 27:
                break

            if key == ord("c"):
                self.clear_marks()

            if key == ord("s"):
                self.save_mask()

            if key == ord("o"):
                self.overexpose = not self.overexpose
                self.update_current_image()

            if key == ord("r"):
                self.undo_last_mark()

            if key == ord("x"):
                self.disable_mask = not self.disable_mask
                self.marks_updated = True

            if key == ord("d"):
                if self.current_index < len(self.image_paths) - 1:
                    self.current_index += 1
                    self.update_current_image()

            if key == ord("a"):
                if self.current_index > 0:
                    self.current_index -= 1
                    self.update_current_image()

            # If we clicked somewhere, call the watershed algorithm on our chosen markers
            if self.marks_updated:
                self.current_image_display = self.current_image.copy()

                watershed_mask = np.zeros(self.current_image.shape[:2], dtype=np.int32)
                for mark in self.marks:
                    cv2.circle(
                        watershed_mask,
                        mark[0],
                        2,
                        mark[1] + 1,
                        -1,
                    )
                    # watershed_mask[mark[0][1], mark[0][0]] = mark[1] + 1

                # run the watershed algorithm with the chosen markers
                cv2.watershed(self.current_image, watershed_mask)

                if not self.disable_mask:
                    for i in range(2, np.max(watershed_mask) + 1):
                        self.current_image_display[watershed_mask == i] = (
                            self.class_colors[(i - 1) % len(self.class_colors)]
                        )

                    self.current_image_display = cv2.addWeighted(
                        self.current_image, 0.8, self.current_image_display, 0.2, 0
                    )

                boundary_mask = np.where(watershed_mask == -1, 1, 0).astype(np.uint8)
                cv2.dilate(boundary_mask, np.ones((3, 3)), boundary_mask, iterations=1)

                self.current_image_display[boundary_mask == 1] = [0, 0, 0]

                for mark in self.marks:
                    mark_class_id = mark[1]

                    cv2.circle(
                        self.current_image_display,
                        mark[0],
                        2,
                        self.class_colors[mark_class_id % len(self.class_colors)],
                        -1,
                    )

                self.marks_updated = False

        cv2.destroyAllWindows()
