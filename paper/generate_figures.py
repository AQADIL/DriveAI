from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import django
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.patches import Polygon
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = PAPER_ROOT / "assets"
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "indrive_car_check.settings")
django.setup()

from ml_model.training import _extract_embeddings, collect_samples  # noqa: E402
from ml_model.views import load_vision_runtime, process_image  # noqa: E402


COLORS = {
    "dent": "#f59e0b",
    "scratch": "#22d3ee",
    "tear": "#f97316",
    "missing_part": "#ef4444",
    "puncture": "#a855f7",
    "lamp_broken": "#eab308",
    "glass_shatter": "#3b82f6",
    "dirt": "#84cc16",
}


def _project_path(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _save_dataset_montage() -> None:
    rows = (
        (
            "Damaged vehicles",
            [
                "fresh_data/data1a/validation/00-damage/0001.JPEG",
                "fresh_data/data1a/validation/00-damage/0004.JPEG",
                "fresh_data/data1a/validation/00-damage/0005.JPEG",
            ],
        ),
        (
            "Intact vehicles",
            [
                "fresh_data/data1a/validation/01-whole/0001.jpg",
                "fresh_data/data1a/validation/01-whole/0064.jpg",
                "fresh_data/data1a/validation/01-whole/0199.jpg",
            ],
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.2), constrained_layout=True)
    for row_index, (row_label, paths) in enumerate(rows):
        for column_index, relative_path in enumerate(paths):
            axis = axes[row_index, column_index]
            with Image.open(_project_path(relative_path)) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#334155")
                spine.set_linewidth(1.2)
            if column_index == 0:
                axis.set_ylabel(row_label, fontsize=11, fontweight="bold")
    fig.savefig(ASSET_ROOT / "dataset_samples.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _annotated_axis(axis, path: Path, panel_label: str, minimum_confidence: float = 0.0) -> None:
    with path.open("rb") as stream:
        result = process_image(stream)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        axis.imshow(image)

    damage_regions = [
        region
        for region in result["regions"]
        if region["type"] != "dirt" and region["confidence"] >= minimum_confidence
    ]
    for region_index, region in enumerate(damage_regions):
        points = np.asarray(
            [
                [point["x"] * width, point["y"] * height]
                for point in region["geometry"]["polygon"]
            ]
        )
        color = COLORS.get(region["type"], "#ef4444")
        axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, alpha=0.25, linewidth=2.2))
        anchor = points[np.argmin(points[:, 1])]
        label = f"{region['type'].replace('_', ' ')}  {region['confidence']:.1f}%"
        axis.text(
            anchor[0],
            max(8, anchor[1] - 5 - 24 * region_index),
            label,
            color="white",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none", "alpha": 0.92},
        )
    axis.text(
        0.015,
        0.975,
        panel_label,
        transform=axis.transAxes,
        va="top",
        color="white",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.88},
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)


def _save_detection_montage() -> None:
    examples = [
        ("(a) missing bumper and hood deformation", "paper/assets/source_user_damage.jpg"),
        ("(b) front-fender dent", "fresh_data/data1a/validation/00-damage/0015.JPEG"),
        ("(c) side-panel scratch", "fresh_data/data1a/validation/00-damage/0018.JPEG"),
        ("(d) missing front-corner components", "fresh_data/data1a/validation/00-damage/0039.JPEG"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    for axis, (panel_label, relative_path) in zip(axes.flat, examples, strict=True):
        _annotated_axis(axis, _project_path(relative_path), panel_label, minimum_confidence=92.0)
    fig.savefig(ASSET_ROOT / "detection_samples.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_integrity_evaluation() -> dict[str, float | int | list[list[int]]]:
    cache_path = ASSET_ROOT / "integrity_embeddings_cache.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        matrix = cached["matrix"]
        train_mask = cached["train_mask"]
        validation_mask = cached["validation_mask"]
        labels = cached["labels"]
    else:
        samples, _ = collect_samples()
        runtime = load_vision_runtime()
        batch_size = int(os.environ["VISION_BATCH_SIZE"])
        matrix, valid_samples, unreadable = _extract_embeddings(runtime, samples, batch_size)
        if unreadable:
            raise RuntimeError(f"Unreadable evaluation images: {unreadable}")
        train_mask = np.asarray([sample.split == "train" for sample in valid_samples])
        validation_mask = np.asarray([sample.split == "validation" for sample in valid_samples])
        labels = np.asarray([sample.intact for sample in valid_samples], dtype=np.int8)
        np.savez_compressed(
            cache_path,
            matrix=matrix,
            train_mask=train_mask,
            validation_mask=validation_mask,
            labels=labels,
        )
    model = LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(matrix[train_mask], labels[train_mask])
    validation_labels = labels[validation_mask]
    probabilities = model.predict_proba(matrix[validation_mask])[:, 1]
    predictions = probabilities >= 0.5
    matrix_counts = confusion_matrix(validation_labels, predictions, labels=[0, 1])

    metrics = {
        "validation_images": int(validation_mask.sum()),
        "accuracy": round(float(accuracy_score(validation_labels, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(validation_labels, predictions)), 4),
        "f1": round(float(f1_score(validation_labels, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(validation_labels, probabilities)), 4),
        "threshold": 0.5,
        "confusion_matrix": matrix_counts.tolist(),
    }

    stored = json.loads((PROJECT_ROOT / "ml_model/car_condition.metrics.json").read_text(encoding="utf-8"))
    expected = stored["integrity_validation"]
    for metric_name in ("accuracy", "balanced_accuracy", "f1", "threshold"):
        if metrics[metric_name] != expected[metric_name]:
            raise RuntimeError(
                f"Reproduced {metric_name}={metrics[metric_name]} does not match stored {expected[metric_name]}"
            )
    roc_auc_delta = round(float(metrics["roc_auc"] - expected["roc_auc"]), 4)
    if abs(roc_auc_delta) > 0.001:
        raise RuntimeError(
            f"Reproduced roc_auc={metrics['roc_auc']} differs materially from stored {expected['roc_auc']}"
        )
    metrics["archived_roc_auc"] = expected["roc_auc"]
    metrics["roc_auc_delta"] = roc_auc_delta

    fig, axis = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    display = ConfusionMatrixDisplay(matrix_counts, display_labels=["Damaged", "Intact"])
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Held-out integrity classification (n = 915)", fontweight="bold")
    fig.savefig(ASSET_ROOT / "confusion_matrix.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.3, 5.0), constrained_layout=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(validation_labels, probabilities)
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#2563eb",
        linewidth=2.3,
        label=f"CLIP + logistic head (AUC = {metrics['roc_auc']:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2, label="Chance")
    axis.set_title("Integrity receiver-operating characteristic", fontweight="bold")
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right")
    fig.savefig(ASSET_ROOT / "integrity_roc.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        ASSET_ROOT / "integrity_validation_predictions.npz",
        labels=validation_labels,
        probabilities=probabilities,
        predictions=predictions,
    )
    (ASSET_ROOT / "integrity_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    _save_dataset_montage()
    _save_detection_montage()
    metrics = _save_integrity_evaluation()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
