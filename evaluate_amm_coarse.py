#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from maskterial.modeling.common.fcresnet import FCResNet


LABEL_NAMES = {
    0: "background",
    1: "low_1_5",
    2: "mid_6_8",
    3: "thick_9_plus",
}


def calculate_background_color(image: np.ndarray, radius: int = 10) -> np.ndarray:
    masks = []
    for channel_idx in range(3):
        image_channel = image[:, :, channel_idx]
        mask = cv2.inRange(image_channel, 20, 230)
        hist = cv2.calcHist([image_channel], [0], mask, [256], [0, 256])
        hist_mode = int(np.argmax(hist))
        thresholded = cv2.inRange(
            image_channel,
            int(hist_mode - radius),
            int(hist_mode + radius),
        )
        background_mask_channel = cv2.erode(
            thresholded,
            np.ones((3, 3), dtype=np.uint8),
            iterations=3,
        )
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])
    return np.array(cv2.mean(image, mask=final_mask)[:3], dtype=np.float32)


def load_model(model_dir: Path, device: torch.device, eps: float) -> dict:
    with (model_dir / "meta_data.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model = FCResNet(**meta["train_config"]["model_arch"]).to(device)
    state = torch.load(model_dir / "model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    loc = torch.from_numpy(np.load(model_dir / "loc.npy")).float().to(device)
    cov = torch.from_numpy(np.load(model_dir / "cov.npy")).float().to(device)
    eye = torch.eye(cov.shape[-1], device=device).unsqueeze(0)
    inv_cov = torch.linalg.pinv(cov + eps * eye)

    train_mean = torch.tensor(meta["train_mean"], dtype=torch.float32, device=device)
    train_std = torch.tensor(meta["train_std"], dtype=torch.float32, device=device)

    return {
        "model": model,
        "loc": loc,
        "inv_cov": inv_cov,
        "train_mean": train_mean,
        "train_std": train_std,
        "meta": meta,
    }


@torch.inference_mode()
def predict_pixels(
    contrast_pixels: np.ndarray,
    bundle: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = bundle["model"]
    loc = bundle["loc"]
    inv_cov = bundle["inv_cov"]
    train_mean = bundle["train_mean"]
    train_std = bundle["train_std"]

    logit_preds = []
    amm_preds = []

    for start in range(0, contrast_pixels.shape[0], batch_size):
        chunk = contrast_pixels[start : start + batch_size]
        x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        x = (x - train_mean) / train_std

        logits = model(x)
        logit_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        emb = model.get_embedding(x)
        diff = emb[:, None, :] - loc[None, :, :]
        distances = torch.einsum("bkd,kde,bke->bk", diff, inv_cov, diff)
        amm_preds.append(torch.argmin(distances, dim=1).cpu().numpy())

    return np.concatenate(logit_preds), np.concatenate(amm_preds)


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    return cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=iterations,
    ).astype(bool)


def majority_vote(labels: np.ndarray, ignore_background: bool) -> int:
    labels = labels.astype(np.int64)
    if ignore_background:
        foreground = labels[labels != 0]
        if foreground.size:
            labels = foreground
    if labels.size == 0:
        return 0
    counts = np.bincount(labels, minlength=4)
    return int(np.argmax(counts))


def confusion_matrix(y_true: list[int], y_pred: list[int], labels: list[int]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            mat[label_to_idx[true], label_to_idx[pred]] += 1
    return mat


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("nan")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class_accuracy(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[int, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    out = {}
    for label in labels:
        mask = y_true_arr == label
        out[label] = float(np.mean(y_pred_arr[mask] == label)) if np.any(mask) else float("nan")
    return out


def macro_f1(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    scores = []
    for label in labels:
        tp = np.sum((y_true_arr == label) & (y_pred_arr == label))
        fp = np.sum((y_true_arr != label) & (y_pred_arr == label))
        fn = np.sum((y_true_arr == label) & (y_pred_arr != label))
        denom = 2 * tp + fp + fn
        scores.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def binary_metrics(y_true: list[int], y_pred: list[int], positive_label: int) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    tp = np.sum((y_true_arr == positive_label) & (y_pred_arr == positive_label))
    fp = np.sum((y_true_arr != positive_label) & (y_pred_arr == positive_label))
    fn = np.sum((y_true_arr == positive_label) & (y_pred_arr != positive_label))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_confusion(path: Path, mat: np.ndarray, labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [LABEL_NAMES.get(label, str(label)) for label in labels])
        for label, row in zip(labels, mat):
            writer.writerow([LABEL_NAMES.get(label, str(label))] + row.tolist())


def print_confusion(title: str, mat: np.ndarray, labels: list[int]) -> None:
    names = [LABEL_NAMES.get(label, str(label)) for label in labels]
    widths = [max(len(name), 8) for name in names]
    first_width = max(len(title), 12)
    print(f"\n[{title}]")
    print(" " * first_width + " " + " ".join(f"{name:>{width}}" for name, width in zip(names, widths)))
    for name, width, row in zip(names, widths, mat):
        values = " ".join(f"{int(value):>{w}}" for value, w in zip(row, widths))
        print(f"{name:>{first_width}} {values}")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bundle = load_model(args.model_dir, device=device, eps=args.cov_eps)

    coco = COCO(str(args.annotation_path))
    category_names = {
        int(cat["id"]): cat.get("name", LABEL_NAMES.get(int(cat["id"]), str(cat["id"])))
        for cat in coco.loadCats(coco.getCatIds())
    }
    for category_id, name in category_names.items():
        LABEL_NAMES[category_id] = name

    out_dir = args.out_dir or (args.model_dir / "eval_coarse")
    out_dir.mkdir(parents=True, exist_ok=True)

    flake_true: list[int] = []
    flake_pred_logit: list[int] = []
    flake_pred_amm: list[int] = []

    pixel_true_counter: Counter[int] = Counter()
    pixel_pred_logit_counter: Counter[tuple[int, int]] = Counter()
    pixel_pred_amm_counter: Counter[tuple[int, int]] = Counter()

    skipped_images = 0
    skipped_annotations = 0

    image_ids = coco.getImgIds()
    for idx, image_id in enumerate(image_ids, start=1):
        if idx == 1 or idx == len(image_ids) or idx % max(1, len(image_ids) // 20) == 0:
            print(f"Processing image {idx}/{len(image_ids)}", flush=True)

        image_info = coco.loadImgs([image_id])[0]
        image_path = args.image_dir / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_images += 1
            continue

        background_color = calculate_background_color(image)
        if np.any(background_color < 1):
            skipped_images += 1
            continue
        contrast_image = (image.astype(np.float32) / background_color) - 1.0

        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
            true_label = int(annotation["category_id"])
            if true_label not in (1, 2, 3):
                continue

            mask = coco.annToMask(annotation)
            mask = erode_mask(mask, iterations=args.erode_iterations)
            if int(mask.sum()) < args.min_mask_pixels:
                skipped_annotations += 1
                continue

            pixels = contrast_image[mask]
            logit_labels, amm_labels = predict_pixels(
                pixels,
                bundle=bundle,
                device=device,
                batch_size=args.batch_size,
            )

            flake_true.append(true_label)
            flake_pred_logit.append(
                majority_vote(logit_labels, ignore_background=args.ignore_background_in_vote)
            )
            flake_pred_amm.append(
                majority_vote(amm_labels, ignore_background=args.ignore_background_in_vote)
            )

            pixel_true_counter[true_label] += int(pixels.shape[0])
            for label, count in zip(*np.unique(logit_labels, return_counts=True)):
                pixel_pred_logit_counter[(true_label, int(label))] += int(count)
            for label, count in zip(*np.unique(amm_labels, return_counts=True)):
                pixel_pred_amm_counter[(true_label, int(label))] += int(count)

    labels = [0, 1, 2, 3]
    foreground_labels = [1, 2, 3]

    logit_cm = confusion_matrix(flake_true, flake_pred_logit, labels)
    amm_cm = confusion_matrix(flake_true, flake_pred_amm, labels)
    write_confusion(out_dir / "flake_confusion_logit.csv", logit_cm, labels)
    write_confusion(out_dir / "flake_confusion_amm.csv", amm_cm, labels)

    pixel_logit_cm = np.zeros((4, 4), dtype=np.int64)
    pixel_amm_cm = np.zeros((4, 4), dtype=np.int64)
    for (true_label, pred_label), count in pixel_pred_logit_counter.items():
        if 0 <= true_label <= 3 and 0 <= pred_label <= 3:
            pixel_logit_cm[true_label, pred_label] += count
    for (true_label, pred_label), count in pixel_pred_amm_counter.items():
        if 0 <= true_label <= 3 and 0 <= pred_label <= 3:
            pixel_amm_cm[true_label, pred_label] += count
    write_confusion(out_dir / "pixel_confusion_logit.csv", pixel_logit_cm, labels)
    write_confusion(out_dir / "pixel_confusion_amm.csv", pixel_amm_cm, labels)

    metrics = {
        "num_images": len(image_ids),
        "num_flakes": len(flake_true),
        "skipped_images": skipped_images,
        "skipped_annotations": skipped_annotations,
        "flake_logit_accuracy": accuracy(flake_true, flake_pred_logit),
        "flake_amm_accuracy": accuracy(flake_true, flake_pred_amm),
        "flake_logit_macro_f1": macro_f1(flake_true, flake_pred_logit, foreground_labels),
        "flake_amm_macro_f1": macro_f1(flake_true, flake_pred_amm, foreground_labels),
        "flake_logit_per_class_accuracy": {
            LABEL_NAMES[label]: value
            for label, value in per_class_accuracy(flake_true, flake_pred_logit, foreground_labels).items()
        },
        "flake_amm_per_class_accuracy": {
            LABEL_NAMES[label]: value
            for label, value in per_class_accuracy(flake_true, flake_pred_amm, foreground_labels).items()
        },
        "low_vs_rest_logit": binary_metrics(flake_true, flake_pred_logit, positive_label=1),
        "low_vs_rest_amm": binary_metrics(flake_true, flake_pred_amm, positive_label=1),
        "pixel_true_counts": {
            LABEL_NAMES[label]: int(pixel_true_counter[label]) for label in foreground_labels
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n[Flake-level majority vote]")
    print(f"  logit accuracy: {metrics['flake_logit_accuracy']:.4f}")
    print(f"  AMM accuracy:   {metrics['flake_amm_accuracy']:.4f}")
    print(f"  logit macro-F1: {metrics['flake_logit_macro_f1']:.4f}")
    print(f"  AMM macro-F1:   {metrics['flake_amm_macro_f1']:.4f}")

    print("\n[Logit per-class accuracy]")
    for name, value in metrics["flake_logit_per_class_accuracy"].items():
        print(f"  {name}: {value:.4f}")
    print("\n[AMM per-class accuracy]")
    for name, value in metrics["flake_amm_per_class_accuracy"].items():
        print(f"  {name}: {value:.4f}")

    print("\n[Low-vs-rest gate]")
    logit_low = metrics["low_vs_rest_logit"]
    amm_low = metrics["low_vs_rest_amm"]
    print(
        "  logit "
        f"precision={logit_low['precision']:.4f} "
        f"recall={logit_low['recall']:.4f} "
        f"f1={logit_low['f1']:.4f}"
    )
    print(
        "  AMM   "
        f"precision={amm_low['precision']:.4f} "
        f"recall={amm_low['recall']:.4f} "
        f"f1={amm_low['f1']:.4f}"
    )

    print_confusion("Flake confusion, logit", logit_cm, labels)
    print_confusion("Flake confusion, AMM", amm_cm, labels)
    print(f"\nSaved evaluation files to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--cov-eps", type=float, default=1e-5)
    parser.add_argument("--erode-iterations", type=int, default=3)
    parser.add_argument("--min-mask-pixels", type=int, default=200)
    parser.add_argument(
        "--include-background-in-vote",
        dest="ignore_background_in_vote",
        action="store_false",
        help="Allow background to win flake-level majority vote.",
    )
    parser.set_defaults(ignore_background_in_vote=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
