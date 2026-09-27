from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from ml_model.vision import VisionRuntime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
ARTIFACT_VERSION = "clip-integrity-v1"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _configured_path(name: str) -> Path:
    path = Path(_required_env(name))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DATA_ROOT = _configured_path("CAR_DATA_ROOT")
DEFAULT_MODEL_PATH = _configured_path("CAR_MODEL_PATH")


@dataclass(frozen=True)
class Sample:
    path: Path
    intact: int
    split: str
    source: str
    digest: str


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _csv_samples(split: str) -> tuple[list[Sample], int]:
    directory = DATA_ROOT / split
    split_name = "validation" if split == "val" else split
    samples: list[Sample] = []
    missing = 0
    with (directory / "labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = directory / row["image"]
            if not image_path.is_file():
                missing += 1
                continue
            samples.append(Sample(image_path, int(row["intact"]), split_name, "csv", _digest(image_path)))
    return samples, missing


def _folder_samples(split: str) -> list[Sample]:
    directory = DATA_ROOT / "data1a" / split
    split_name = "train" if split == "training" else "validation"
    samples: list[Sample] = []
    for folder_name, intact in (("00-damage", 0), ("01-whole", 1)):
        for image_path in sorted((directory / folder_name).glob("*")):
            if image_path.is_file():
                samples.append(Sample(image_path, intact, split_name, "data1a", _digest(image_path)))
    return samples


def collect_samples() -> tuple[list[Sample], dict[str, int]]:
    train_csv, train_missing = _csv_samples("train")
    validation_csv, validation_missing = _csv_samples("val")
    test_csv, test_missing = _csv_samples("test")
    raw_samples = train_csv + _folder_samples("training") + validation_csv + _folder_samples("validation") + test_csv
    unique_samples: list[Sample] = []
    seen: dict[str, Sample] = {}
    duplicate_count = 0
    conflicting_duplicates = 0
    for sample in raw_samples:
        previous = seen.get(sample.digest)
        if previous is not None:
            duplicate_count += 1
            if previous.intact != sample.intact:
                conflicting_duplicates += 1
            continue
        seen[sample.digest] = sample
        unique_samples.append(sample)
    return unique_samples, {
        "train_csv_missing": train_missing,
        "validation_csv_missing": validation_missing,
        "test_csv_missing": test_missing,
        "exact_duplicates_removed": duplicate_count,
        "conflicting_duplicates": conflicting_duplicates,
    }


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.2, 0.8, 121)
    scored = [
        (balanced_accuracy_score(labels, probabilities >= threshold), -abs(threshold - 0.5), threshold)
        for threshold in candidates
    ]
    return round(float(max(scored)[2]), 3)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        "f1": round(float(f1_score(labels, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "threshold": threshold,
    }


def _classifier(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )


def _extract_embeddings(runtime: VisionRuntime, samples: list[Sample], batch_size: int) -> tuple[np.ndarray, list[Sample], list[str]]:
    embeddings: list[np.ndarray] = []
    valid_samples: list[Sample] = []
    unreadable: list[str] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        images: list[Image.Image] = []
        readable_samples: list[Sample] = []
        for sample in batch_samples:
            try:
                with Image.open(sample.path) as image:
                    images.append(image.copy())
                readable_samples.append(sample)
            except Exception as error:
                unreadable.append(f"{sample.path.relative_to(PROJECT_ROOT)}: {error}")
        if images:
            embeddings.append(runtime.encode_images(images, batch_size=batch_size))
            valid_samples.extend(readable_samples)
        completed = min(start + batch_size, len(samples))
        print(f"Encoded images: {completed}/{len(samples)}", flush=True)
    return np.vstack(embeddings), valid_samples, unreadable


def train(model_path: Path, seed: int = 42) -> dict[str, object]:
    samples, dataset_report = collect_samples()
    runtime = VisionRuntime(
        model_id=_required_env("VISION_MODEL_ID"),
        cache_dir=_configured_path("VISION_MODEL_CACHE_DIR"),
        device_name=_required_env("VISION_DEVICE"),
    )
    batch_size = int(_required_env("VISION_BATCH_SIZE"))
    matrix, valid_samples, unreadable = _extract_embeddings(runtime, samples, batch_size)
    train_mask = np.asarray([sample.split == "train" for sample in valid_samples])
    validation_mask = np.asarray([sample.split == "validation" for sample in valid_samples])
    labels = np.asarray([sample.intact for sample in valid_samples], dtype=np.int8)

    candidates: list[tuple[float, float, float, float, dict[str, float]]] = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = _classifier(c_value, seed)
        model.fit(matrix[train_mask], labels[train_mask])
        probabilities = model.predict_proba(matrix[validation_mask])[:, 1]
        threshold = _best_threshold(labels[validation_mask], probabilities)
        metrics = _metrics(labels[validation_mask], probabilities, threshold)
        candidates.append((metrics["balanced_accuracy"], -abs(c_value - 1.0), c_value, threshold, metrics))

    _, _, selected_c, intact_threshold, validation_metrics = max(candidates, key=lambda item: (item[0], item[1]))
    final_model = _classifier(selected_c, seed)
    final_model.fit(matrix, labels)

    metadata: dict[str, object] = {
        "artifact_version": ARTIFACT_VERSION,
        "vision_model_id": runtime.model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "integrity_training_images": len(valid_samples),
        "cleanliness_engine": "CLIP zero-shot prompt ensemble; repository clean labels rejected after audit",
        "dataset": dataset_report,
        "unreadable_images": unreadable,
        "thresholds": {"clean": 0.5, "intact": intact_threshold},
        "integrity_validation": validation_metrics,
        "integrity_logistic_c": selected_c,
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "intact_model": final_model,
        "metadata": metadata,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    model_path.with_suffix(".metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the integrity head on CLIP image embeddings.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output.resolve(), args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
