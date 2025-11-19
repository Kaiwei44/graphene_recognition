import os, argparse, types, numpy as np
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from maskterial.utils.dataset_functions import setup_config 

def build_cfg_with_setup(config_file, weights, extra_opts=None):
    args = types.SimpleNamespace(
        config_file=config_file,
        opts=["MODEL.WEIGHTS", weights] + (extra_opts or []),
        resume=False, eval_only=True, num_gpus=1, num_machines=1, machine_rank=0, dist_url="auto"
    )
    return setup_config(args)

def compute_recall_fp_per_image(coco_eval, iou_thr=0.5):
    """
    依据 pycocotools 的匹配结果计算：
      - Recall@IoU=iou_thr = TP / GT
      - FP/image@IoU=iou_thr
    仅考虑 area=all、maxDet=最后一个（通常=100），适合单类任务（多类会做平均）。
    """
    if coco_eval is None or coco_eval.evalImgs is None:
        return 0.0, 0.0, dict(tp=0, gt=0, fp=0, images=0)

    # 选中 IoU=0.5 所在索引
    ious = coco_eval.params.iouThrs
    ti = int(np.where(np.isclose(ious, iou_thr))[0][0]) if np.any(np.isclose(ious, iou_thr)) else 0

    gt_total = 0
    tp_total = 0
    fp_total = 0
    img_ids = set()

    for ev in coco_eval.evalImgs:
        if ev is None:  # 某些 (img, cat) 组合没有标注
            continue
        img_ids.add(ev["image_id"])

        # ev 中的数据结构：dtMatches[T,D], dtIgnore[T,D], gtIgnore[G]
        dtMatches_t = np.array(ev["dtMatches"][ti])  # >0 表示该预测在该 IoU 阈值被匹配上
        dtIgnore_t  = np.array(ev["dtIgnore"][ti]).astype(np.uint8)
        gtIgnore    = np.array(ev["gtIgnore"]).astype(np.uint8)

        # 本图有效 GT 数（不忽略的）
        gt_total += int(np.sum(gtIgnore == 0))

        # TP：未忽略 且 被匹配
        tp = int(np.sum((dtIgnore_t == 0) & (dtMatches_t > 0)))
        tp_total += tp

        # FP：未忽略 且 未匹配
        fp = int(np.sum((dtIgnore_t == 0) & (dtMatches_t == 0)))
        fp_total += fp

    n_img = len(img_ids) if img_ids else 1
    recall = tp_total / gt_total if gt_total > 0 else 0.0
    fp_per_img = fp_total / n_img if n_img > 0 else 0.0
    return recall, fp_per_img, dict(tp=tp_total, gt=gt_total, fp=fp_total, images=n_img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--ann", required=True)   # 单类合并后的 COCO JSON
    ap.add_argument("--dataset-name", default="graphene_test_bin")
    ap.add_argument("--outdir", default="./eval_out")
    args = ap.parse_args()

    # 注册测试集（单类）
    register_coco_instances(args.dataset_name, {}, args.ann, args.image_root)
    MetadataCatalog.get(args.dataset_name).set(thing_classes=["flakes"])

    # cfg：完全复用训练时的 setup_config，只覆盖权重
    cfg = build_cfg_with_setup(args.config_file, args.weights)
    predictor = DefaultPredictor(cfg)

    os.makedirs(args.outdir, exist_ok=True)
    evaluator = COCOEvaluator(args.dataset_name, distributed=False, output_dir=args.outdir)
    val_loader = build_detection_test_loader(cfg, args.dataset_name)

    # 运行评测（会打印标准 COCO 表）
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    # 取出 segm 的 COCOeval 对象
    coco_eval = evaluator._coco_eval.get("segm", None)

    # ------- 1) 读取 COCO 标准指标 -------
    if coco_eval is not None and coco_eval.stats is not None:
        AP     = float(coco_eval.stats[0])  # AP@[.50:.95]
        AP50   = float(coco_eval.stats[1])  # AP@0.50
        AR100  = float(coco_eval.stats[8])  # AR@100（参考）
    else:
        AP = AP50 = AR100 = float("nan")

    # ------- 2) 计算 Recall@0.5 / FP per image -------
    recall05, fp_per_img05, dbg = compute_recall_fp_per_image(coco_eval, iou_thr=0.5)

    print("\n==== Summary (segm) ====")
    print(f"AP@[.50:.95]: {AP:.4f}")
    print(f"AP@0.50     : {AP50:.4f}")
    print(f"AR@100      : {AR100:.4f}")
    print(f"Recall@0.50 : {recall05:.4f}   (TP={dbg['tp']},  GT={dbg['gt']})")
    print(f"FP/image@0.50: {fp_per_img05:.4f}   (FP={dbg['fp']}, Images={dbg['images']})")
    print(f"Raw results written to: {args.outdir}")

if __name__ == "__main__":
    main()