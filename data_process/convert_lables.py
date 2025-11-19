#!/usr/bin/env python3
# merge_to_single_class.py
"""
把 COCO 标注合并为单类 "sample"（id=1）：
- 保留并合并：thin / thick / intermediate  → category_id = 1
- 丢弃：tapeResidue / flakes → 当作背景
- 默认只保留“仍有正例”的图片；如需保留空图，把 KEEP_ONLY_POSITIVE_IMAGES 设为 False

用法示例：
python merge_to_single_class.py \
  --src /abs/path/_annotations_rle.coco.json \
  --dst /abs/path/_annotations_rle_binary.coco.json
"""

import json
import argparse
from collections import Counter

KEEP_NAMES = {"thin", "thick", "intermediate"}
DROP_NAMES = {"tapeResidue", "flakes"}
SINGLE_CLASS = {"id": 1, "name": "sample"}
KEEP_ONLY_POSITIVE_IMAGES = True  # 没有正例的图片是否剔除

def main(src, dst):
    with open(src, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    # id -> name
    id2name = {c["id"]: c.get("name", "") for c in cats}

    # 统计
    by_class_before = Counter()
    kept_by_old_class = Counter()
    dropped_by_old_class = Counter()

    new_anns = []
    used_img_ids = set()

    for ann in anns:
        cname = id2name.get(ann.get("category_id"), "")
        by_class_before[cname] += 1
        if cname in KEEP_NAMES:
            ann2 = dict(ann)
            ann2["category_id"] = SINGLE_CLASS["id"]
            new_anns.append(ann2)
            used_img_ids.add(ann2["image_id"])
            kept_by_old_class[cname] += 1
        else:
            dropped_by_old_class[cname] += 1

    if KEEP_ONLY_POSITIVE_IMAGES:
        new_images = [im for im in images if im.get("id") in used_img_ids]
    else:
        new_images = images

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_anns,
        "categories": [SINGLE_CLASS],
    }

    with open(dst, "w") as f:
        json.dump(out, f)

    # 摘要
    print("=== Merge Summary ===")
    print("source:", src)
    print("dest  :", dst)
    print(f"images: {len(images)} -> {len(new_images)}")
    print(f"anns  : {len(anns)} -> {len(new_anns)}")
    print("before by class:", dict(by_class_before))
    print("kept    by class:", dict(kept_by_old_class))
    print("dropped by class:", dict(dropped_by_old_class))

    # 基础一致性检查
    img_ids = {im["id"] for im in new_images}
    dangling = [a["id"] for a in new_anns if a["image_id"] not in img_ids]
    if dangling:
        print(f"[WARN] found {len(dangling)} annotations with unknown image_id")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="输入 COCO JSON 路径")
    ap.add_argument("--dst", required=True, help="输出合并后的 COCO JSON 路径")
    args = ap.parse_args()
    main(args.src, args.dst)