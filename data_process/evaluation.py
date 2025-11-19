import os
import cv2
import random
import argparse
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# === 这里导入你仓库里定义 setup_config 的真实路径（按你找到的文件改） ===
# 例：from maskterial.configs.setup import setup_config
from maskterial.utils.dataset_functions import setup_config   # <<< 如果导入失败，请把路径改成你项目里的实际模块路径
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True, help="./configs/M2F/base_config.yaml")
    ap.add_argument("--weights", required=True, help="~/Parameter/trained_para/model_final.pth")
    ap.add_argument("--image-root", required=True, help="~/Data/hBN/test")
    ap.add_argument("--ann", required=True, help="~/Data/hBN/test/_annotations_1class.coco.json")
    ap.add_argument("--outdir", default="./vis_pred", help="./data_process/eval_vis")
    ap.add_argument("--num-samples", type=int, default=20, help="")
    ap.add_argument("--scale", type=float, default=0.8, help="可视化缩放")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draw-gt", action="store_true", help="同时导出 GT 可视化（用于对照）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)

    # 1) 用 setup_config 按“训练同款流程”构建 cfg
    #    只覆盖一项：MODEL.WEIGHTS 指向你微调后的权重；其他都用 base_config 默认
    class Args:
        config_file = args.config_file
        opts = ["MODEL.WEIGHTS", args.weights]
        # default_setup 需要的最少字段：
        resume = False
        eval_only = True
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"

    cfg = setup_config(Args())  # 内部会 get_cfg -> add_deeplab_config -> add_extended_config -> merge_from_file -> merge_from_list

    # 2) 注册测试集（你已把测试集合并为单类）
    TEST_NAME = "graphene_test_bin"
    register_coco_instances(TEST_NAME, {}, args.ann, args.image_root)
    MetadataCatalog.get(TEST_NAME).set(thing_classes=["sample"])  # 仅影响显示名
    meta = MetadataCatalog.get(TEST_NAME)

    # 3) 预测与可视化
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(TEST_NAME)
    samples = dataset_dicts if args.num_samples == -1 else random.sample(
        dataset_dicts, min(args.num_samples, len(dataset_dicts))
    )

    for d in samples:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=meta, scale=args.scale, instance_mode=ColorMode.IMAGE)
        pred_img = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

        base = os.path.splitext(os.path.basename(d["file_name"]))[0]
        pred_path = os.path.join(args.outdir, f"{base}_pred.jpg")
        cv2.imwrite(pred_path, pred_img)

        if args.draw_gt:
            v2 = Visualizer(img[:, :, ::-1], metadata=meta, scale=args.scale, instance_mode=ColorMode.IMAGE)
            gt_img = v2.draw_dataset_dict(d).get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(args.outdir, f"{base}_gt.jpg"), gt_img)

        print("saved:", pred_path)
    print("All done. ->", args.outdir)

if __name__ == "__main__":
    main()