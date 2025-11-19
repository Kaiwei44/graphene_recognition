import json
import os

import numpy as np

BASE_DATASET_PATH = "./data/datasets"
MATERIALS = [
    "GrapheneH",
    "GrapheneL",
    "GrapheneM",
    "hBN_Thin",
    "MoSe2",
    "WS2",
    "WSe2",
    "WSe2L",
]

NUM_RUNS = 10
NUM_IMAGES_PER_CLASS = [1, 2, 3, 5, 10]

np.random.seed(42)
for material in MATERIALS:
    material_path = os.path.join(BASE_DATASET_PATH, material)
    meta_path = os.path.join(
        material_path,
        "RLE_annotations",
        "train_annotations_with_class_300.json",
    )
    cls_meta_path = os.path.join(
        material_path,
        "RLE_annotations",
        "train_annotations_300.json",
    )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(cls_meta_path, "r") as f:
        cls_meta = json.load(f)

    cls_categories = cls_meta["categories"].copy()
    cls_instances = cls_meta["annotations"].copy()

    categories = meta["categories"].copy()
    instances = meta["annotations"].copy()
    images = meta["images"].copy()

    print(
        f"Material: {material} has {len(categories)} categories and {len(images)} images"
    )

    for run_id in range(NUM_RUNS):
        for num_flakes in NUM_IMAGES_PER_CLASS:
            np.random.shuffle(instances)

            new_dataset_path = os.path.join(
                material_path,
                "RLE_annotations",
                f"{num_flakes}_shot",
                f"run_{run_id}",
            )

            os.makedirs(new_dataset_path, exist_ok=True)

            selected_instances_ids = set()
            for category in categories:
                num_selected_flakes = 0
                for instance in instances:
                    if instance["category_id"] == category["id"]:
                        image_id = instance["image_id"]
                        # now the all the instances that are on the image
                        same_instances = [
                            instance
                            for instance in instances
                            if instance["image_id"] == image_id
                        ]
                        for same_instance in same_instances:
                            selected_instances_ids.add(same_instance["id"])
                        num_selected_flakes += 1

                    if num_selected_flakes >= num_flakes:
                        break

            num_instances_per_class = {category["id"]: 0 for category in categories}
            selected_image_ids = set()
            for instance_id in selected_instances_ids:
                instace_info = [
                    instance for instance in instances if instance["id"] == instance_id
                ][0]
                num_instances_per_class[instace_info["category_id"]] += 1
                selected_image_ids.add(instace_info["image_id"])

            new_instances = [
                instance
                for instance in instances
                if instance["id"] in selected_instances_ids
            ]
            new_cls_instance = [
                cls_instance
                for cls_instance in cls_instances
                if cls_instance["id"] in selected_instances_ids
            ]
            new_images = [
                image for image in images if image["id"] in selected_image_ids
            ]

            new_coco = meta.copy()
            new_coco["images"] = new_images
            new_coco["annotations"] = new_instances

            new_class_agnostic_coco = cls_meta.copy()
            new_class_agnostic_coco["images"] = new_images
            new_class_agnostic_coco["annotations"] = new_cls_instance

            with open(
                os.path.join(new_dataset_path, "train_annotations_with_class_300.json"),
                "w",
            ) as f:
                json.dump(new_coco, f, indent=4)

            with open(
                os.path.join(new_dataset_path, "train_annotations_300.json"), "w"
            ) as f:
                json.dump(new_class_agnostic_coco, f, indent=4)
