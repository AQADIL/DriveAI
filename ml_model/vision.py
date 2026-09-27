from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, CLIPModel


CLEAN_PROMPTS = (
    "a spotless clean car with a freshly washed exterior",
    "a clean polished car body with no visible dirt",
)
DIRTY_PROMPTS = (
    "a dirty muddy car covered in mud and road grime",
    "a filthy car with visible dirt, dust, stains, or road salt",
)
INTACT_PROMPTS = (
    "an intact undamaged car body with no dents or scratches",
    "a car in normal condition without visible body damage",
)
DAMAGE_PROMPTS = (
    "a crashed car with visible collision damage and broken panels",
    "a damaged car with dents, scratches, broken lights, or missing parts",
)
REGION_PROMPTS = (
    "clean intact car paint with an even glossy surface",
    "car body covered in dirt mud dust grime or road salt",
    "a visible scratch scrape or scuff on a car body panel",
    "a dented or deformed car body panel",
    "chipped cracked or peeling automotive paint",
    "rust or corrosion on a car body panel",
    "a broken cracked or missing exterior car part",
)
REGION_LABELS = (
    "clear",
    "dirt",
    "scratch",
    "dent",
    "paint_chip",
    "rust",
    "broken_part",
)


class VisionRuntime:
    def __init__(self, model_id: str, cache_dir: Path, device_name: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device_name)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        self._cleanliness_text_features = self.encode_texts(CLEAN_PROMPTS + DIRTY_PROMPTS)
        self._damage_text_features = self.encode_texts(INTACT_PROMPTS + DAMAGE_PROMPTS)
        self._region_text_features = self.encode_texts(REGION_PROMPTS)

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VISION_DEVICE is cuda, but CUDA is unavailable")
        return torch.device(device_name)

    @staticmethod
    def _normalized(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(prompts), padding=True, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.text_model(**inputs)
            features = self.model.text_projection(outputs.pooler_output)
        return self._normalized(features)

    def encode_images(self, images: Iterable[Image.Image], batch_size: int = 16) -> np.ndarray:
        prepared = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        batches: list[np.ndarray] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                outputs = self.model.vision_model(pixel_values=pixel_values)
                features = self.model.visual_projection(outputs.pooler_output)
                features = self._normalized(features)
            batches.append(features.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, self.model.projection_dim), dtype=np.float32)
        return np.vstack(batches)

    def _group_probability(self, image_features: np.ndarray, text_features: torch.Tensor, positive_start: int) -> float:
        features = torch.from_numpy(image_features).to(self.device).reshape(1, -1)
        with torch.inference_mode():
            scale = self.model.logit_scale.exp().clamp(max=100)
            logits = scale * features @ text_features.T
            probabilities = logits.softmax(dim=-1)[0]
        return float(probabilities[positive_start:].sum().cpu())

    def cleanliness_probability(self, image_features: np.ndarray) -> float:
        dirty_probability = self._group_probability(
            image_features,
            self._cleanliness_text_features,
            len(CLEAN_PROMPTS),
        )
        return 1.0 - dirty_probability

    def damage_probability(self, image_features: np.ndarray) -> float:
        return self._group_probability(
            image_features,
            self._damage_text_features,
            len(INTACT_PROMPTS),
        )

    def region_probabilities(self, image_features: np.ndarray) -> np.ndarray:
        features = torch.from_numpy(image_features).to(self.device)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        with torch.inference_mode():
            scale = self.model.logit_scale.exp().clamp(max=100)
            logits = scale * features @ self._region_text_features.T
            probabilities = logits.softmax(dim=-1)
        return probabilities.cpu().numpy().astype(np.float32)
