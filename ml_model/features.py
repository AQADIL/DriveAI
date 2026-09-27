from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


FEATURE_VERSION = "car-condition-v1"
IMAGE_SIZE = (160, 160)


def _open_image(source: str | Path | bytes | BinaryIO) -> Image.Image:
    if isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)
    return ImageOps.exif_transpose(image).convert("RGB")


def _summary(values: np.ndarray) -> list[float]:
    return [
        float(values.mean()),
        float(values.std()),
        *[float(value) for value in np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])],
    ]


def extract_features(source: str | Path | bytes | BinaryIO) -> np.ndarray:
    image = _open_image(source).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    features: list[float] = []
    for channel in range(3):
        features.extend(_summary(rgb[:, :, channel]))
        histogram, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
        features.extend(histogram.astype(float).tolist())

    features.extend(_summary(gray))
    gray_histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
    features.extend(gray_histogram.astype(float).tolist())
    features.extend(_summary(hsv[:, :, 1]))
    features.extend(_summary(hsv[:, :, 2]))

    gradient_x = np.abs(np.diff(gray, axis=1)).ravel()
    gradient_y = np.abs(np.diff(gray, axis=0)).ravel()
    gradients = np.concatenate((gradient_x, gradient_y))
    features.extend(_summary(gradients))
    features.extend(float(np.mean(gradients > threshold)) for threshold in (0.05, 0.1, 0.2, 0.3))

    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    features.extend(_summary(np.abs(laplacian).ravel()))

    for row in np.array_split(rgb, 4, axis=0):
        for cell in np.array_split(row, 4, axis=1):
            features.extend(cell.mean(axis=(0, 1)).astype(float).tolist())
            features.append(float(cell.std()))

    return np.asarray(features, dtype=np.float32)
