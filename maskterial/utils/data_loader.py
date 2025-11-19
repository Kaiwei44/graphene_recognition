import os
from typing import Literal, Tuple, Union

import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class ContrastDataloader:
    def __init__(
        self,
        train_image_dir: str,
        train_annotation_path: str,
        test_image_dir: str = None,
        test_annotation_path: str = None,
        max_samples_per_class: int = 30_000,
        loaded_test_samples: int = 50_000,
        uniform_class_sampling: bool = True,
        use_normalization: bool = True,
        use_DBSCAN: bool = True,
        DBSCAN_eps: float = 0.1,
        use_Nearest_Neighbors: bool = True,
        neighbors: int = 25,
        verbose: bool = False,
    ):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train_image_dir = train_image_dir
        self.train_annotation_path = train_annotation_path

        self.test_image_dir = test_image_dir
        self.test_annotation_path = test_annotation_path

        self.uniform_class_sampling = uniform_class_sampling
        self.max_samples_per_class = max_samples_per_class

        self.use_DBSCAN = use_DBSCAN
        self.DBSCAN_eps = DBSCAN_eps

        self.use_Nearest_Neighbors = use_Nearest_Neighbors
        self.neighbors = neighbors

        self.use_normalization = use_normalization

        if os.path.isdir(self.train_annotation_path):
            self.X_train, self.y_train = self.load_contrasts_from_directory(
                self.train_image_dir, self.train_annotation_path
            )
        elif os.path.isfile(
            self.train_annotation_path
        ) and self.train_annotation_path.endswith(".json"):
            self.X_train, self.y_train = self.load_contrasts_from_COCO(
                self.train_image_dir, self.train_annotation_path
            )
        else:
            raise ValueError(
                f"Invalid annotation path {self.train_annotation_path}. Expected a directory or a COCO json file"
            )

        self.class_ids = np.unique(self.y_train)
        self.num_classes = len(np.unique(self.y_train))
        # <--- 第 1 块代码添加在这里
        print("--- 初始数据状态 ---")
        print(f"所有类别: {np.unique(self.y_train)}")
        print(f"每个类别的像素点数量:\n{np.bincount(self.y_train)}")
        # <--- 添加结束

        if self.test_image_dir is not None and self.test_annotation_path is not None:
            self.X_test, self.y_test = self.load_contrasts_from_COCO(
                self.test_image_dir, self.test_annotation_path
            )

            # randomly sample a fixed amount of samples from the test set
            if self.X_test.shape[0] > loaded_test_samples:
                indices = np.random.choice(
                    self.X_test.shape[0], loaded_test_samples, replace=False
                )
                self.X_test = self.X_test[indices]
                self.y_test = self.y_test[indices]

        if self.max_samples_per_class != -1:
            self._apply_class_sampling()

        if self.use_Nearest_Neighbors:
            if verbose:
                print(
                    "Applying Nearest Neighbors (Denoising Classes)",
                    flush=True,
                )
            self._apply_Nearest_Neighbors()
            # <--- 第 2 块代码添加在这里
            print("--- Nearest Neighbors 处理完毕后 ---")
            print(f"剩余类别: {np.unique(self.y_train)}")
            print(f"每个类别的像素点数量:\n{np.bincount(self.y_train)}")
            # <--- 添加结束

        if self.use_DBSCAN:
            if verbose:
                print(
                    "Applying DBSCAN (Removing Outliers)",
                    flush=True,
                )
            self._apply_DBSCAN()
            # <--- 第 3 块代码添加在这里
            print("--- DBSCAN 处理完毕后 ---")
            if self.y_train.size > 0:
                print(f"剩余类别: {np.unique(self.y_train)}")
                print(f"每个类别的像素点数量:\n{np.bincount(self.y_train)}")
            else:
                print("数据已被清空!")
            # <--- 添加结束


        if self.use_normalization:
            if verbose:
                print(
                    "Applying Normalization (Projecting to Normal Distribution)",
                    flush=True,
                )
            self._apply_normalization()

        if self.uniform_class_sampling:
            if verbose:
                print(
                    "Applying class partitioning (Uniform Class Sampling)",
                    flush=True,
                )
            self._apply_class_partition()
        

    def load_contrasts_from_COCO(
        self,
        image_dir: str,
        annotation_path: str,
    ):
        coco = COCO(annotation_path)

        # we expect the category ids to be in the range [1, num_classes]
        # e.g. [1,2,3,4] for mono-, bi-, tri-, and quad-layer flakes
        # we add the 0 class as the background class
        category_ids = coco.getCatIds()
        if 0 not in category_ids:
            category_ids = [0] + category_ids

        image_dicts = coco.loadImgs(coco.getImgIds())

        X, y = [], []

        for idx, image_dict in enumerate(image_dicts):

            # display every 5% of the images
            try:
                if idx % (len(image_dicts) // 20) == 0:
                    print(
                        f"Processing Image {idx + 1:5}/{len(image_dicts):5} ({(idx + 1) / len(image_dicts):.1%})",
                        flush=True,
                    )
            except ZeroDivisionError:
                print(
                    f"Processing Image {idx + 1:5}/{len(image_dicts):5} ({(idx + 1) / len(image_dicts):.1%})",
                    flush=True,
                )

            image_path = os.path.join(image_dir, image_dict["file_name"])
            image = cv2.imread(image_path)

            background_color = self._calculate_background_color(image)

            # skip the image if the background color is not valid
            # i.e. the background of any of the channels is black
            if np.any(background_color < 1):
                continue

            contrast_image = (image / background_color) - 1

            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_dict["id"]]))

            # we keep track of the image mask to later sample from the background
            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for annotation in annotations:
                class_id = annotation["category_id"]
                mask = coco.annToMask(annotation)
                full_mask[mask != 0] = 1

                # erode the mask to remove the edges
                # these tend to be noisy and not representative of the class
                mask = cv2.erode(
                    mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=3,
                )

                # remove small masks which came up due to eroding
                if np.count_nonzero(mask) < 200:
                    continue

                X.extend(contrast_image[mask != 0])
                y.extend([class_id] * np.count_nonzero(mask))

            # add the background class by sampling 10000 pixels from the background
            background_pixels = contrast_image[full_mask == 0]
            sampled_indices = np.random.choice(
                background_pixels.shape[0], 10000, replace=True
            )
            X.extend(background_pixels[sampled_indices])
            y.extend([0] * 10000)

        print(f"\nLoaded {len(image_dicts)} images from COCO dataset")

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def load_contrasts_from_directory(
        self,
        image_dir: str,
        mask_dir: str,
    ):
        mask_names = os.listdir(mask_dir)

        X, y = [], []
        for idx, mask_name in enumerate(mask_names):
            print(
                f"Processing Image {idx + 1:5}/{len(mask_names):5} ({(idx + 1) / len(mask_names):.1%})",
                end="\r",
            )

            mask_path = os.path.join(mask_dir, mask_name)
            image_path = os.path.join(image_dir, mask_name)

            # check if the image exists, if not try to find the image with the same name but different extension
            # if that also does not exist, skip the image
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, mask_name.replace(".png", ".jpg"))
            if not os.path.exists(image_path):
                print(f"\nImage {image_path} not found, skipping...")
                continue

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, 0)

            background_color = self._calculate_background_color(image)
            # skip the image if the background color is not valid
            # i.e. the background of any of the channels is black
            if np.any(background_color < 1):
                print(f"\nInvalid background color for image {image_path}, skipping...")
                continue

            contrast_image = (image / background_color) - 1

            for class_id in np.unique(mask):
                if class_id == 0:
                    continue

                class_mask = np.array(np.where(mask == class_id, 1, 0), dtype=np.uint8)
                class_mask = cv2.erode(
                    class_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=3,
                )

                X.extend(contrast_image[class_mask != 0])
                y.extend([class_id] * np.count_nonzero(class_mask))

            # add the background class by sampling 10000 pixels from the background
            background_pixels = contrast_image[mask == 0]
            sampled_indices = np.random.choice(
                background_pixels.shape[0], 10000, replace=True
            )
            X.extend(background_pixels[sampled_indices])
            y.extend([0] * 10000)

        print(f"\nLoaded {len(mask_names)} images from masks")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def _calculate_background_color(self, image, radius=10):
        masks = []

        for i in range(3):
            image_channel = image[:, :, i]
            mask = cv2.inRange(image_channel, 20, 230)
            hist = cv2.calcHist([image_channel], [0], mask, [256], [0, 256])
            hist_mode = np.argmax(hist)
            thresholded_image = cv2.inRange(
                image_channel, int(hist_mode - radius), int(hist_mode + radius)
            )
            background_mask_channel = cv2.erode(
                thresholded_image, np.ones((3, 3)), iterations=3
            )
            masks.append(background_mask_channel)

        final_mask = cv2.bitwise_and(masks[0], masks[1])
        final_mask = cv2.bitwise_and(final_mask, masks[2])

        return np.array(cv2.mean(image, mask=final_mask)[:3])

    def _apply_class_sampling(self):
        class_ids = np.unique(self.y_train)

        X_sampled = []
        y_sampled = []

        for class_id in class_ids:
            class_data = self.X_train[self.y_train == class_id]
            replace = class_data.shape[0] < self.max_samples_per_class
            sampled_indices = np.random.choice(
                class_data.shape[0], self.max_samples_per_class, replace=replace
            )
            X_sampled.extend(class_data[sampled_indices])
            y_sampled.extend([class_id] * self.max_samples_per_class)

        self.X_train = np.array(X_sampled, dtype=np.float32)
        self.y_train = np.array(y_sampled, dtype=np.int32)

    def _apply_normalization(self):
        # apply weighted normalization
        # get the smallest class count
        class_counts = np.min(np.bincount(self.y_train))

        full_sampled_data = []
        # sample from each class the same amount of samples
        # and calculate the mean and std of the samples
        for class_id in np.unique(self.y_train):
            class_data = self.X_train[self.y_train == class_id]
            np.random.shuffle(class_data)
            sampled_data = class_data[:class_counts]
            full_sampled_data.extend(sampled_data)

        full_sampled_data = np.array(full_sampled_data, dtype=np.float32)

        self.X_train_mean = np.mean(full_sampled_data, axis=0)
        self.X_train_std = np.std(full_sampled_data, axis=0)

        self.X_train = (self.X_train - self.X_train_mean) / self.X_train_std

        if self.X_test is not None:
            self.X_test = (self.X_test - self.X_train_mean) / self.X_train_std

    def _apply_DBSCAN(self):
        class_ids = np.unique(self.y_train)

        X_clean = []
        y_clean = []

        for class_id in class_ids:
            class_data = self.X_train[self.y_train == class_id]

            dbscan = DBSCAN(
                eps=self.DBSCAN_eps, min_samples=class_data.shape[0] // 10, n_jobs=-1
            )
            dbscan.fit(class_data)

            core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True
            class_data_core = class_data[core_samples_mask]

            X_clean.extend(class_data_core)
            y_clean.extend([class_id] * class_data_core.shape[0])

        self.X_train = np.array(X_clean, dtype=np.float32)
        self.y_train = np.array(y_clean, dtype=np.int32)

    def _apply_Nearest_Neighbors(self):
        # denoise the classes
        NN = NearestNeighbors(n_neighbors=self.neighbors)
        NN.fit(self.X_train)

        # Find the indices of the 9 nearest neighbors for each point in X
        _, indices = NN.kneighbors(self.X_train)

        # Look up the labels of the nearest neighbors using the indices
        neighbor_labels = self.y_train[indices]

        # Determine the new label for each point based on the majority label of its neighbors
        self.y_train = np.array(
            [np.argmax(np.bincount(labels)) for labels in neighbor_labels]
        )

    def _apply_class_partition(self):
        self.X_train_dict = {
            class_id: self.X_train[self.y_train == class_id]
            for class_id in np.unique(self.y_train)
        }

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.uniform_class_sampling:
            samples_per_class = batch_size // self.num_classes

            # sample samples_per_class samples from each class
            X_batch, y_batch = [], []
            for class_id in self.class_ids:
                class_data = self.X_train_dict[class_id]
                sample_indices = np.random.choice(
                    class_data.shape[0], samples_per_class, replace=True
                )
                X_batch.extend(class_data[sample_indices])
                y_batch.extend([class_id] * samples_per_class)
            X_batch = np.array(X_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.int32)
            shuffel_index = np.random.permutation(X_batch.shape[0])
            X_batch = X_batch[shuffel_index]
            y_batch = y_batch[shuffel_index]

        else:
            sample_indices = np.random.choice(
                self.X_train.shape[0], batch_size, replace=True
            )
            X_batch = np.array(self.X_train[sample_indices], dtype=np.float32)
            y_batch = np.array(self.y_train[sample_indices], dtype=np.int32)

        return X_batch, y_batch

    def get_test_data(self, equal_class_sampling: bool = True):
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test dataset provided")

        if equal_class_sampling:
            min_class_counts = np.min(np.bincount(self.y_test))

            X_test = []
            y_test = []
            for class_id in self.class_ids:
                class_data = self.X_test[self.y_test == class_id]
                sample_indices = np.random.choice(
                    class_data.shape[0], min_class_counts, replace=False
                )
                X_test.extend(class_data[sample_indices])
                y_test.extend([class_id] * min_class_counts)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.int32)
            return X_test, y_test

        else:
            return self.X_test, self.y_test


def load_contrast_from_directory(
    dataset_root: str,
    split: Union[Literal["train", "test"], None] = None,
):
    if split is None:
        data_dir = dataset_root
    else:
        data_dir = os.path.join(dataset_root, split)

    file_names = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    file_classes = [int(f.split("_")[-1].split(".")[0]) for f in file_names]

    X, y = [], []
    for file_name, file_class in zip(file_names, file_classes):
        file_path = os.path.join(data_dir, file_name)
        contrast: list = np.load(file_path, allow_pickle=True)

        concat_contrasts = []
        for c in contrast:
            concat_contrasts.extend(c)

        X.extend(concat_contrasts)
        y.extend([file_class] * len(concat_contrasts))

    return np.array(X, dtype=np.float32), np.array(y)
