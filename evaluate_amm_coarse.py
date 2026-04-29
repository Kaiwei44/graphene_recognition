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

    loc = None
    inv_cov = None
    if (model_dir / "loc.npy").exists() and (model_dir / "cov.npy").exists():
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
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    model = bundle["model"]
    loc = bundle["loc"]
    inv_cov = bundle["inv_cov"]
    train_mean = bundle["train_mean"]
    train_std = bundle["train_std"]

    logit_preds = []
    amm_preds = []
    prob_sum = torch.zeros(model.last.out_features, dtype=torch.float64, device=device)
    num_pixels = 0

    for start in range(0, contrast_pixels.shape[0], batch_size):
        chunk = contrast_pixels[start : start + batch_size]
        x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
        x = (x - train_mean) / train_std

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        prob_sum += probs.double().sum(dim=0)
        num_pixels += int(probs.shape[0])
        logit_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        if loc is not None and inv_cov is not None:
            emb = model.get_embedding(x)
            diff = emb[:, None, :] - loc[None, :, :]
            distances = torch.einsum("bkd,kde,bke->bk", diff, inv_cov, diff)
            amm_preds.append(torch.argmin(distances, dim=1).cpu().numpy())

    mean_probs = (prob_sum / max(num_pixels, 1)).float().cpu().numpy()
    amm_array = np.concatenate(amm_preds) if amm_preds else None
    return np.concatenate(logit_preds), amm_array, mean_probs


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


def foreground_softmax_prediction(mean_probs: np.ndarray) -> tuple[int, float, float]:
    foreground = mean_probs[1:4].astype(np.float64)
    foreground_sum = float(foreground.sum())
    if foreground_sum <= 0:
        return 0, 0.0, 0.0
    foreground_probs = foreground / foreground_sum
    pred = int(np.argmax(foreground_probs) + 1)
    return pred, float(foreground_probs[0]), float(np.max(foreground_probs))


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


def score_threshold_metrics(
    y_true: list[int],
    scores: list[float],
    thresholds: list[float],
    positive_label: int,
) -> list[dict[str, float]]:
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    out = []
    for threshold in thresholds:
        pred_positive = scores_arr >= threshold
        true_positive = y_true_arr == positive_label
        tp = int(np.sum(true_positive & pred_positive))
        fp = int(np.sum((~true_positive) & pred_positive))
        fn = int(np.sum(true_positive & (~pred_positive)))
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out.append(
            {
                "threshold": float(threshold),
                "selected": int(np.sum(pred_positive)),
                "selected_rate": float(np.mean(pred_positive)) if len(pred_positive) else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return out


def write_confusion(path: Path, mat: np.ndarray, labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [LABEL_NAMES.get(label, str(label)) for label in labels])
        for label, row in zip(labels, mat):
            writer.writerow([LABEL_NAMES.get(label, str(label))] + row.tolist())


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    flake_pred_softmax_fg: list[int] = []
    flake_low_prob_raw: list[float] = []
    flake_low_prob_fg: list[float] = []
    flake_fg_confidence: list[float] = []
    prediction_rows: list[dict] = []

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
            logit_labels, amm_labels, mean_probs = predict_pixels(
                pixels,
                bundle=bundle,
                device=device,
                batch_size=args.batch_size,
            )
            softmax_fg_pred, low_prob_fg, fg_confidence = foreground_softmax_prediction(mean_probs)

            flake_true.append(true_label)
            logit_majority_pred = majority_vote(
                logit_labels, ignore_background=args.ignore_background_in_vote
            )
            flake_pred_logit.append(logit_majority_pred)
            flake_pred_softmax_fg.append(softmax_fg_pred)
            flake_low_prob_raw.append(float(mean_probs[1]) if mean_probs.shape[0] > 1 else 0.0)
            flake_low_prob_fg.append(low_prob_fg)
            flake_fg_confidence.append(fg_confidence)
            amm_majority_pred = None
            if amm_labels is not None:
                amm_majority_pred = majority_vote(
                    amm_labels, ignore_background=args.ignore_background_in_vote
                )
                flake_pred_amm.append(amm_majority_pred)

            pixel_true_counter[true_label] += int(pixels.shape[0])
            for label, count in zip(*np.unique(logit_labels, return_counts=True)):
                pixel_pred_logit_counter[(true_label, int(label))] += int(count)
            if amm_labels is not None:
                for label, count in zip(*np.unique(amm_labels, return_counts=True)):
                    pixel_pred_amm_counter[(true_label, int(label))] += int(count)

            prediction_rows.append(
                {
                    "filename": image_info["file_name"],
                    "image_id": image_id,
                    "ann_id": annotation.get("id", ""),
                    "true_label": true_label,
                    "true_name": LABEL_NAMES.get(true_label, str(true_label)),
                    "pred_logit_majority": logit_majority_pred,
                    "pred_softmax_fg": softmax_fg_pred,
                    "pred_amm_majority": "" if amm_majority_pred is None else amm_majority_pred,
                    "p_background": float(mean_probs[0]) if mean_probs.shape[0] > 0 else 0.0,
                    "p_low_raw": float(mean_probs[1]) if mean_probs.shape[0] > 1 else 0.0,
                    "p_mid_raw": float(mean_probs[2]) if mean_probs.shape[0] > 2 else 0.0,
                    "p_thick_raw": float(mean_probs[3]) if mean_probs.shape[0] > 3 else 0.0,
                    "p_low_fg": low_prob_fg,
                    "fg_confidence": fg_confidence,
                    "mask_pixels": int(pixels.shape[0]),
                }
            )

    labels = [0, 1, 2, 3]
    foreground_labels = [1, 2, 3]

    logit_cm = confusion_matrix(flake_true, flake_pred_logit, labels)
    softmax_fg_cm = confusion_matrix(flake_true, flake_pred_softmax_fg, labels)
    amm_cm = confusion_matrix(flake_true, flake_pred_amm, labels) if flake_pred_amm else None
    write_confusion(out_dir / "flake_confusion_logit.csv", logit_cm, labels)
    write_confusion(out_dir / "flake_confusion_softmax_fg.csv", softmax_fg_cm, labels)
    if amm_cm is not None:
        write_confusion(out_dir / "flake_confusion_amm.csv", amm_cm, labels)

    pixel_logit_cm = np.zeros((4, 4), dtype=np.int64)
    pixel_amm_cm = np.zeros((4, 4), dtype=np.int64)
    for (true_label, pred_label), count in pixel_pred_logit_counter.items():
        if 0 <= true_label <= 3 and 0 <= pred_label <= 3:
            pixel_logit_cm[true_label, pred_label] += count
    if pixel_pred_amm_counter:
        for (true_label, pred_label), count in pixel_pred_amm_counter.items():
            if 0 <= true_label <= 3 and 0 <= pred_label <= 3:
                pixel_amm_cm[true_label, pred_label] += count
    write_confusion(out_dir / "pixel_confusion_logit.csv", pixel_logit_cm, labels)
    if pixel_pred_amm_counter:
        write_confusion(out_dir / "pixel_confusion_amm.csv", pixel_amm_cm, labels)

    thresholds = [float(x) for x in args.low_thresholds.split(",") if x.strip()]
    low_sweep_fg = score_threshold_metrics(
        flake_true, flake_low_prob_fg, thresholds=thresholds, positive_label=1
    )
    low_sweep_raw = score_threshold_metrics(
        flake_true, flake_low_prob_raw, thresholds=thresholds, positive_label=1
    )
    write_dict_rows(out_dir / "predictions.csv", prediction_rows)
    write_dict_rows(out_dir / "low_gate_sweep_fg_prob.csv", low_sweep_fg)
    write_dict_rows(out_dir / "low_gate_sweep_raw_prob.csv", low_sweep_raw)

    metrics = {
        "num_images": len(image_ids),
        "num_flakes": len(flake_true),
        "skipped_images": skipped_images,
        "skipped_annotations": skipped_annotations,
        "flake_logit_accuracy": accuracy(flake_true, flake_pred_logit),
        "flake_softmax_fg_accuracy": accuracy(flake_true, flake_pred_softmax_fg),
        "flake_amm_accuracy": accuracy(flake_true, flake_pred_amm) if flake_pred_amm else None,
        "flake_logit_macro_f1": macro_f1(flake_true, flake_pred_logit, foreground_labels),
        "flake_softmax_fg_macro_f1": macro_f1(
            flake_true, flake_pred_softmax_fg, foreground_labels
        ),
        "flake_amm_macro_f1": macro_f1(flake_true, flake_pred_amm, foreground_labels)
        if flake_pred_amm
        else None,
        "flake_logit_per_class_accuracy": {
            LABEL_NAMES[label]: value
            for label, value in per_class_accuracy(flake_true, flake_pred_logit, foreground_labels).items()
        },
        "flake_softmax_fg_per_class_accuracy": {
            LABEL_NAMES[label]: value
            for label, value in per_class_accuracy(
                flake_true, flake_pred_softmax_fg, foreground_labels
            ).items()
        },
        "flake_amm_per_class_accuracy": {
            LABEL_NAMES[label]: value
            for label, value in per_class_accuracy(flake_true, flake_pred_amm, foreground_labels).items()
        }
        if flake_pred_amm
        else None,
        "low_vs_rest_logit": binary_metrics(flake_true, flake_pred_logit, positive_label=1),
        "low_vs_rest_softmax_fg_hard": binary_metrics(
            flake_true, flake_pred_softmax_fg, positive_label=1
        ),
        "low_vs_rest_amm": binary_metrics(flake_true, flake_pred_amm, positive_label=1)
        if flake_pred_amm
        else None,
        "low_gate_sweep_fg_prob": low_sweep_fg,
        "low_gate_sweep_raw_prob": low_sweep_raw,
        "pixel_true_counts": {
            LABEL_NAMES[label]: int(pixel_true_counter[label]) for label in foreground_labels
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n[Flake-level majority vote]")
    print(f"  logit accuracy: {metrics['flake_logit_accuracy']:.4f}")
    print(f"  softmax-fg accuracy: {metrics['flake_softmax_fg_accuracy']:.4f}")
    if metrics["flake_amm_accuracy"] is not None:
        print(f"  AMM accuracy:   {metrics['flake_amm_accuracy']:.4f}")
    print(f"  logit macro-F1: {metrics['flake_logit_macro_f1']:.4f}")
    print(f"  softmax-fg macro-F1: {metrics['flake_softmax_fg_macro_f1']:.4f}")
    if metrics["flake_amm_macro_f1"] is not None:
        print(f"  AMM macro-F1:   {metrics['flake_amm_macro_f1']:.4f}")

    print("\n[Logit per-class accuracy]")
    for name, value in metrics["flake_logit_per_class_accuracy"].items():
        print(f"  {name}: {value:.4f}")
    print("\n[Softmax foreground per-class accuracy]")
    for name, value in metrics["flake_softmax_fg_per_class_accuracy"].items():
        print(f"  {name}: {value:.4f}")
    if metrics["flake_amm_per_class_accuracy"] is not None:
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
    softmax_low = metrics["low_vs_rest_softmax_fg_hard"]
    print(
        "  softmax-fg "
        f"precision={softmax_low['precision']:.4f} "
        f"recall={softmax_low['recall']:.4f} "
        f"f1={softmax_low['f1']:.4f}"
    )
    if metrics["low_vs_rest_amm"] is not None:
        amm_low = metrics["low_vs_rest_amm"]
        print(
            "  AMM   "
            f"precision={amm_low['precision']:.4f} "
            f"recall={amm_low['recall']:.4f} "
            f"f1={amm_low['f1']:.4f}"
        )

    print("\n[Low gate threshold sweep, foreground-normalized probability]")
    for row in low_sweep_fg:
        print(
            f"  t={row['threshold']:.2f} "
            f"selected={row['selected']:3d} "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} "
            f"f1={row['f1']:.4f}"
        )

    print_confusion("Flake confusion, logit", logit_cm, labels)
    print_confusion("Flake confusion, softmax-fg", softmax_fg_cm, labels)
    if amm_cm is not None:
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
        "--low-thresholds",
        default="0.40,0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated thresholds for p(low | foreground) gate evaluation.",
    )
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
