from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .vision import REGION_LABELS, VisionRuntime


FINDING_COPY = {
    "dirt": {
        "label": "Dirt or road grime",
        "description": "Surface contamination is visually concentrated in this area.",
        "recommendation": "Wash and dry the panel before assessing the paint underneath.",
    },
    "scratch": {
        "label": "Possible scratch",
        "description": "The texture resembles a scrape or a linear paint defect.",
        "recommendation": "Inspect under diffuse light and check whether the mark catches a fingernail.",
    },
    "dent": {
        "label": "Possible dent",
        "description": "The panel geometry or reflections look locally distorted.",
        "recommendation": "Inspect the panel from a low angle and compare reflections across it.",
    },
    "paint_chip": {
        "label": "Possible paint damage",
        "description": "The area resembles chipped, cracked, or peeling automotive paint.",
        "recommendation": "Check for exposed metal and seal the area promptly if confirmed.",
    },
    "rust": {
        "label": "Possible corrosion",
        "description": "Color and texture in this area resemble rust or corrosion.",
        "recommendation": "Inspect the metal edge closely and assess whether corrosion has spread below paint.",
    },
    "broken_part": {
        "label": "Possible broken part",
        "description": "The area resembles a cracked, displaced, or missing exterior component.",
        "recommendation": "Check the component mounts and have safety-critical parts inspected professionally.",
    },
}


@dataclass(frozen=True)
class ScanBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def scan_boxes(image_width: int, image_height: int, columns: int, rows: int, window_ratio: float) -> list[ScanBox]:
    window_width = max(1, round(image_width * window_ratio))
    window_height = max(1, round(image_height * window_ratio))
    left_positions = _axis_positions(image_width, window_width, columns)
    top_positions = _axis_positions(image_height, window_height, rows)
    return [
        ScanBox(left, top, min(image_width, left + window_width), min(image_height, top + window_height))
        for top in top_positions
        for left in left_positions
    ]


def _axis_positions(length: int, window: int, count: int) -> list[int]:
    if count <= 1 or length <= window:
        return [0]
    return [round(index * (length - window) / (count - 1)) for index in range(count)]


def intersection_over_union(first: ScanBox, second: ScanBox) -> float:
    intersection_width = max(0, min(first.right, second.right) - max(first.left, second.left))
    intersection_height = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = intersection_width * intersection_height
    if not intersection:
        return 0.0
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union


def _severity(score: float) -> str:
    if score >= 0.78:
        return "high"
    if score >= 0.58:
        return "medium"
    return "low"


def _finding_payload(index: int, finding_type: str, score: float, box: ScanBox, image: Image.Image) -> dict[str, object]:
    copy = FINDING_COPY[finding_type]
    return {
        "id": f"finding-{index}",
        "type": finding_type,
        "label": copy["label"],
        "description": copy["description"],
        "recommendation": copy["recommendation"],
        "severity": _severity(score),
        "confidence": round(score * 100, 1),
        "geometry": {
            "x": round(box.left / image.width, 5),
            "y": round(box.top / image.height, 5),
            "width": round(box.width / image.width, 5),
            "height": round(box.height / image.height, 5),
        },
    }


def localize_findings(
    runtime: VisionRuntime,
    image: Image.Image,
    *,
    include_dirt: bool,
    include_damage: bool,
    columns: int,
    rows: int,
    window_ratio: float,
    dirt_threshold: float,
    damage_threshold: float,
    overlap_threshold: float,
    max_regions: int,
    batch_size: int,
) -> list[dict[str, object]]:
    if not include_dirt and not include_damage:
        return []

    source = image.convert("RGB")
    boxes = scan_boxes(source.width, source.height, columns, rows, window_ratio)
    crops = [source.crop((box.left, box.top, box.right, box.bottom)) for box in boxes]
    features = runtime.encode_images(crops, batch_size=batch_size)
    probabilities = runtime.region_probabilities(features)
    candidates: list[tuple[float, str, ScanBox]] = []

    for box, scores in zip(boxes, probabilities, strict=True):
        if include_dirt:
            dirt_score = float(scores[REGION_LABELS.index("dirt")])
            if dirt_score >= dirt_threshold:
                candidates.append((dirt_score, "dirt", box))
        if include_damage:
            finding_type = max(
                ("scratch", "dent", "paint_chip", "rust", "broken_part"),
                key=lambda label: float(scores[REGION_LABELS.index(label)]),
            )
            damage_score = float(scores[REGION_LABELS.index(finding_type)])
            if damage_score >= damage_threshold:
                candidates.append((damage_score, finding_type, box))

    selected: list[tuple[float, str, ScanBox]] = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(
            candidate[1] == existing[1]
            and intersection_over_union(candidate[2], existing[2]) > overlap_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_regions:
            break

    return [
        _finding_payload(index + 1, finding_type, score, box, source)
        for index, (score, finding_type, box) in enumerate(selected)
    ]
