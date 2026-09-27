from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from ml_model.features import FEATURE_VERSION, extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _configured_path(name: str) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DATA_ROOT = _configured_path("CAR_DATA_ROOT")
DEFAULT_MODEL_PATH = _configured_path("CAR_MODEL_PATH")


@dataclass(frozen=True)
class Sample:
    path: Path
    clean: int | None
    intact: int
    split: str


def _csv_samples(split: str) -> tuple[list[Sample], int]:
    directory = DATA_ROOT / split
    samples: list[Sample] = []
    missing = 0
    split_name = "validation" if split == "val" else split
    with (directory / "labels.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = directory / row["image"]
            if not image_path.is_file():
                missing += 1
                continue
            samples.append(Sample(image_path, int(row["clean"]), int(row["intact"]), split_name))
    return samples, missing


def _folder_samples(split: str) -> list[Sample]:
    directory = DATA_ROOT / "data1a" / split
    samples: list[Sample] = []
    split_name = "train" if split == "training" else "validation"
    for folder_name, intact in (("00-damage", 0), ("01-whole", 1)):
        for image_path in sorted((directory / folder_name).glob("*")):
            if image_path.is_file():
                samples.append(Sample(image_path, None, intact, split_name))
    return samples


def collect_samples() -> tuple[list[Sample], dict[str, int]]:
    train_csv, train_missing = _csv_samples("train")
    validation_csv, validation_missing = _csv_samples("val")
    test_csv, test_missing = _csv_samples("test")
    samples = train_csv + validation_csv + test_csv + _folder_samples("training") + _folder_samples("validation")
    return samples, {
        "train_csv_missing": train_missing,
        "validation_csv_missing": validation_missing,
        "test_csv_missing": test_missing,
    }


def _make_classifier(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


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


def train(model_path: Path, seed: int = 42) -> dict[str, object]:
    samples, missing = collect_samples()
    features: list[np.ndarray] = []
    valid_samples: list[Sample] = []
    unreadable: list[str] = []

    for index, sample in enumerate(samples, start=1):
        try:
            features.append(extract_features(sample.path))
            valid_samples.append(sample)
        except Exception as error:
            unreadable.append(f"{sample.path.relative_to(PROJECT_ROOT)}: {error}")
        if index % 500 == 0 or index == len(samples):
            print(f"Extracted features: {index}/{len(samples)}", flush=True)

    matrix = np.vstack(features)
    train_indexes = np.asarray([sample.split == "train" for sample in valid_samples])
    validation_indexes = np.asarray([sample.split == "validation" for sample in valid_samples])
    clean_indexes = np.asarray([sample.clean is not None for sample in valid_samples])
    intact_labels = np.asarray([sample.intact for sample in valid_samples], dtype=np.int8)
    clean_labels = np.asarray([sample.clean if sample.clean is not None else -1 for sample in valid_samples], dtype=np.int8)

    clean_validation_model = _make_classifier(seed)
    clean_validation_model.fit(matrix[train_indexes & clean_indexes], clean_labels[train_indexes & clean_indexes])
    clean_validation_labels = clean_labels[validation_indexes & clean_indexes]
    clean_validation_probabilities = clean_validation_model.predict_proba(matrix[validation_indexes & clean_indexes])[:, 1]
    clean_threshold = _best_threshold(clean_validation_labels, clean_validation_probabilities)
    clean_metrics = _metrics(clean_validation_labels, clean_validation_probabilities, clean_threshold)

    intact_validation_model = _make_classifier(seed)
    intact_validation_model.fit(matrix[train_indexes], intact_labels[train_indexes])
    intact_validation_labels = intact_labels[validation_indexes]
    intact_validation_probabilities = intact_validation_model.predict_proba(matrix[validation_indexes])[:, 1]
    intact_threshold = _best_threshold(intact_validation_labels, intact_validation_probabilities)
    intact_metrics = _metrics(intact_validation_labels, intact_validation_probabilities, intact_threshold)

    clean_model = _make_classifier(seed)
    clean_model.fit(matrix[clean_indexes], clean_labels[clean_indexes])
    intact_model = _make_classifier(seed)
    intact_model.fit(matrix, intact_labels)

    metadata: dict[str, object] = {
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "available_images": len(valid_samples),
        "clean_training_images": int(clean_indexes.sum()),
        "intact_training_images": len(valid_samples),
        "missing_images": missing,
        "unreadable_images": unreadable,
        "thresholds": {"clean": clean_threshold, "intact": intact_threshold},
        "validation": {"clean": clean_metrics, "intact": intact_metrics},
    }
    artifact = {
        "feature_version": FEATURE_VERSION,
        "clean_model": clean_model,
        "intact_model": intact_model,
        "metadata": metadata,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    model_path.with_suffix(".metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifiers on every available labeled car image.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = train(args.output.resolve(), args.seed)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
