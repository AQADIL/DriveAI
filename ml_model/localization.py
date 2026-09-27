from __future__ import annotations

from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import torch
from cardamage import AutoModel
from PIL import Image, ImageOps
from transformers import AutoProcessor, CLIPSegForImageSegmentation


DAMAGE_TYPE_BY_LABEL = {
    "Dent": "dent",
    "Paint scratch": "scratch",
    "Tear": "tear",
    "Missing part": "missing_part",
    "Puncture": "puncture",
    "Broken lamp": "lamp_broken",
    "Broken glass": "glass_shatter",
}


FINDING_COPY = {
    "dirt": (
        "Dirt or road grime",
        "The dense segmentation model found concentrated surface contamination in this area.",
        "Wash and dry the panel before assessing the paint underneath.",
    ),
    "scratch": (
        "Possible scratch",
        "The segmented surface pattern resembles a scratch or paint abrasion.",
        "Inspect under diffuse light and check whether the mark catches a fingernail.",
    ),
    "dent": (
        "Possible dent",
        "The segmented panel area resembles a local deformation or dent.",
        "Inspect the panel from a low angle and compare reflections across it.",
    ),
    "crack": (
        "Possible crack",
        "The model found an irregular region consistent with a crack or fracture.",
        "Check the area closely and have structural cracks assessed professionally.",
    ),
    "glass_shatter": (
        "Possible glass damage",
        "The segmented region resembles shattered or heavily cracked vehicle glass.",
        "Do not drive if visibility or glass retention may be compromised.",
    ),
    "lamp_broken": (
        "Possible lamp damage",
        "The model found a region consistent with a cracked or broken exterior lamp.",
        "Check lamp operation, housing seals, and mounting points.",
    ),
    "tire_flat": (
        "Possible tire-area damage",
        "The model found an abnormal region around a tire or wheel area.",
        "Check tire pressure, sidewall condition, and wheel clearance before driving.",
    ),
    "missing_part": (
        "Missing exterior part",
        "The segmentation model found exposed internal structure where an exterior part should be.",
        "Do not drive until the mounts, crash structure, cooling components, and wiring are inspected.",
    ),
    "tear": (
        "Torn body material",
        "The segmented region is consistent with torn, split, or severely deformed body material.",
        "Have the panel and the structure behind it inspected before driving.",
    ),
    "puncture": (
        "Punctured body panel",
        "The segmentation model found a hole or puncture in the vehicle body.",
        "Inspect the panel and any components behind it for secondary damage.",
    ),
}


class LocalizationRuntime:
    def __init__(
        self,
        *,
        cache_dir: Path,
        device_name: str,
        damage_model_id: str,
        damage_confidence: float,
        damage_duplicate_overlap: float,
        max_damage_regions: int,
        dirt_model_id: str,
        dirt_threshold: float,
        dirt_quantile: float,
        dirt_min_area: float,
        max_dirt_regions: int,
    ):
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device_name)
        self.damage_model_id = damage_model_id
        self.damage_confidence = damage_confidence
        self.damage_duplicate_overlap = damage_duplicate_overlap
        self.max_damage_regions = max_damage_regions
        self.dirt_model_id = dirt_model_id
        self.dirt_threshold = dirt_threshold
        self.dirt_quantile = dirt_quantile
        self.dirt_min_area = dirt_min_area
        self.max_dirt_regions = max_dirt_regions

    @staticmethod
    def _resolve_device(device_name: str) -> str:
        if device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VISION_DEVICE is cuda, but CUDA is unavailable")
        return device_name

    @cached_property
    def damage_model(self):
        model = AutoModel.from_pretrained(
            self.damage_model_id,
            cache_dir=self.cache_dir,
        )
        model.roi_heads.score_thresh = self.damage_confidence
        return model.to(torch.device(self.device)).eval()

    @cached_property
    def dirt_processor(self):
        return AutoProcessor.from_pretrained(self.dirt_model_id, cache_dir=self.cache_dir)

    @cached_property
    def dirt_model(self):
        return CLIPSegForImageSegmentation.from_pretrained(
            self.dirt_model_id,
            cache_dir=self.cache_dir,
        ).to(self.device).eval()

    @staticmethod
    def _severity(confidence: float) -> str:
        if confidence >= 0.78:
            return "high"
        if confidence >= 0.55:
            return "medium"
        return "low"

    @staticmethod
    def _geometry(points: np.ndarray) -> dict[str, object]:
        points = np.clip(points.astype(np.float64), 0.0, 1.0)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return {
            "polygon": [
                {"x": round(float(x), 5), "y": round(float(y), 5)}
                for x, y in points
            ],
            "bounds": {
                "x": round(float(x_min), 5),
                "y": round(float(y_min), 5),
                "width": round(float(x_max - x_min), 5),
                "height": round(float(y_max - y_min), 5),
            },
        }

    def _payload(self, finding_id: str, finding_type: str, confidence: float, points: np.ndarray) -> dict[str, object]:
        label, description, recommendation = FINDING_COPY[finding_type]
        return {
            "id": finding_id,
            "type": finding_type,
            "label": label,
            "description": description,
            "recommendation": recommendation,
            "severity": self._severity(confidence),
            "confidence": round(confidence * 100, 1),
            "geometry": self._geometry(points),
        }

    @staticmethod
    def _mask_polygon(mask: np.ndarray) -> np.ndarray | None:
        binary = np.where(mask, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 16:
            return None
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, 0.003 * perimeter, True).reshape(-1, 2)
        if len(simplified) < 3:
            return None
        return simplified / np.asarray([binary.shape[1] - 1, binary.shape[0] - 1])

    def _damage_findings(self, image: Image.Image) -> list[dict[str, object]]:
        result = self.damage_model.predict(ImageOps.exif_transpose(image).convert("RGB"))
        findings = []
        accepted_masks = []
        detections = zip(
            result["labels_en"],
            result["scores"],
            result["masks"],
            strict=True,
        )
        for label, score, mask in detections:
            confidence = float(score)
            finding_type = DAMAGE_TYPE_BY_LABEL.get(label)
            if finding_type is None or confidence < self.damage_confidence:
                continue
            binary_mask = np.asarray(mask, dtype=bool)
            duplicate = False
            for accepted_type, accepted_mask in accepted_masks:
                if accepted_type != finding_type:
                    continue
                smaller_area = min(binary_mask.sum(), accepted_mask.sum())
                if smaller_area == 0:
                    continue
                overlap = np.logical_and(binary_mask, accepted_mask).sum() / smaller_area
                if overlap >= self.damage_duplicate_overlap:
                    duplicate = True
                    break
            if duplicate:
                continue
            points = self._mask_polygon(binary_mask)
            if points is None:
                continue
            finding_index = len(findings) + 1
            findings.append(self._payload(f"damage-{finding_index}", finding_type, confidence, points))
            accepted_masks.append((finding_type, binary_mask))
            if len(findings) >= self.max_damage_regions:
                break
        return findings

    def _dirt_findings(self, image: Image.Image) -> list[dict[str, object]]:
        source = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.dirt_processor(
            text=["mud splashes and dirty stains"],
            images=[source],
            padding=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            mask = self.dirt_model(**inputs).logits.sigmoid()[0].cpu().numpy()

        threshold = max(self.dirt_threshold, float(np.quantile(mask, self.dirt_quantile)))
        binary = np.where(mask >= threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = binary.shape[0] * binary.shape[1] * self.dirt_min_area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        findings = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, 0.012 * perimeter, True).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            normalized = simplified / np.asarray([binary.shape[1] - 1, binary.shape[0] - 1])
            contour_mask = np.zeros_like(binary)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            confidence = float(mask[contour_mask > 0].mean())
            findings.append(self._payload(f"dirt-{len(findings) + 1}", "dirt", confidence, normalized))
            if len(findings) >= self.max_dirt_regions:
                break
        return findings

    def localize(self, image: Image.Image, *, include_dirt: bool, include_damage: bool) -> list[dict[str, object]]:
        findings = []
        if include_damage:
            findings.extend(self._damage_findings(image))
        if include_dirt:
            findings.extend(self._dirt_findings(image))
        return findings
