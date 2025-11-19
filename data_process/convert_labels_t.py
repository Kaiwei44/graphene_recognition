# merge_all_flakes_to_single.py
# 用法:
#   python merge_all_flakes_to_single.py \
#     --src /abs/path/_annotations_rle.coco.json \
#     --dst /abs/path/_annotations_rle_flakes1class.coco.json \
#     [--keep-empty]   # 可选: 保留没有正例的空图片

import json, argparse
from collections import Counter

TARGET = {"id": 1, "name": "flakes"}  # 合并后的单一类别
# 如果数据里没有 supercategory 信息，兜底用名字集合来判断
NAME_WHITELIST = {"flakes", "thin", "thick", "intermediate", "tapeResidue"}

def main(src, dst, keep_empty=False):
    with open(src, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns   = coco.get("annotations", [])
    cats   = coco.get("categories", [])

    id2name = {c["id"]: c.get("name","") for c in cats}
    id2super = {c["id"]: c.get("supercategory","") for c in cats}

    # 需要保留并合并的 category_id 集合：supercategory == 'flakes' 或 名字在白名单里
    keep_cat_ids = set()
    for c in cats:
        name = c.get("name","")
        sup  = c.get("supercategory","")
        if sup == "flakes" or name in NAME_WHITELIST:
            keep_cat_ids.add(c["id"])

    before_cnt = Counter()
    kept_old   = Counter()
    dropped    = Counter()

    new_anns = []
    used_img_ids = set()

    for a in anns:
        cid = a.get("category_id")
        cname = id2name.get(cid, "")
        before_cnt[cname] += 1

        if cid in keep_cat_ids:
            na = dict(a)
            na["category_id"] = TARGET["id"]
            new_anns.append(na)
            used_img_ids.add(na["image_id"])
            kept_old[cname] += 1
        else:
            dropped[cname] += 1

    if keep_empty:
        new_images = images
    else:
        keep_set = used_img_ids
        new_images = [im for im in images if im.get("id") in keep_set]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_anns,
        "categories": [TARGET],  # 只保留一个类别
    }

    with open(dst, "w") as f:
        json.dump(out, f)

    print("=== Merge to single 'flakes' class ===")
    print("src :", src)
    print("dst :", dst)
    print(f"images: {len(images)} -> {len(new_images)}")
    print(f"anns  : {len(anns)} -> {len(new_anns)}")
    print("by class BEFORE :", dict(before_cnt))
    print("kept by old name:", dict(kept_old))
    print("dropped by name :", dict(dropped))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep-empty", action="store_true", help="保留没有正例的空图片")
    args = ap.parse_args()
    main(args.src, args.dst, keep_empty=args.keep_empty)
