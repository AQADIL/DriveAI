from __future__ import annotations

from functools import lru_cache

import joblib
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image

from .forms import PredictionUploadForm
from .localization import LocalizationRuntime
from .training import ARTIFACT_VERSION
from .vision import VisionRuntime


def home(request):
    return render(
        request,
        "ml_model/home.html",
        {
            "max_image_bytes": settings.MAX_IMAGE_BYTES,
            "max_image_mb": f"{settings.MAX_IMAGE_BYTES / (1024 * 1024):g}",
            "allowed_formats": ", ".join(sorted(settings.ALLOWED_IMAGE_FORMATS)),
        },
    )


@lru_cache(maxsize=1)
def load_model():
    model_path = settings.CAR_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    artifact = joblib.load(model_path)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError("The model artifact version does not match the application")
    if artifact["metadata"]["vision_model_id"] != settings.VISION_MODEL_ID:
        raise RuntimeError("The configured vision model does not match the trained integrity head")
    return artifact


@lru_cache(maxsize=1)
def load_vision_runtime():
    return VisionRuntime(
        model_id=settings.VISION_MODEL_ID,
        cache_dir=settings.VISION_MODEL_CACHE_DIR,
        device_name=settings.VISION_DEVICE,
    )


@lru_cache(maxsize=1)
def load_localization_runtime():
    return LocalizationRuntime(
        cache_dir=settings.VISION_MODEL_CACHE_DIR,
        device_name=settings.VISION_DEVICE,
        damage_model_id=settings.DAMAGE_SEGMENTATION_MODEL_ID,
        damage_model_file=settings.DAMAGE_SEGMENTATION_MODEL_FILE,
        damage_confidence=settings.DAMAGE_SEGMENTATION_CONFIDENCE,
        damage_image_size=settings.DAMAGE_SEGMENTATION_IMAGE_SIZE,
        max_damage_regions=settings.MAX_DAMAGE_REGIONS,
        dirt_model_id=settings.DIRT_SEGMENTATION_MODEL_ID,
        dirt_threshold=settings.DIRT_SEGMENTATION_THRESHOLD,
        dirt_quantile=settings.DIRT_SEGMENTATION_QUANTILE,
        dirt_min_area=settings.DIRT_SEGMENTATION_MIN_AREA,
        max_dirt_regions=settings.MAX_DIRT_REGIONS,
    )


def _explanation(clean: bool, integrity_status: str) -> str:
    if integrity_status == "inconclusive":
        return "The car is too dirty for a reliable body inspection. Clean it and upload a new photo."
    if clean and integrity_status == "intact":
        return "The car appears clean, and the model found no visible body damage."
    if not clean and integrity_status == "intact":
        return "The model detected dirt but found no visible body damage."
    if clean and integrity_status == "possible_damage":
        return "The car appears clean, but the model detected signs of body damage."
    return "The model detected dirt and signs of body damage. A manual inspection is recommended."


def process_image(image_file) -> dict[str, object]:
    artifact = load_model()
    runtime = load_vision_runtime()
    image_file.seek(0)
    with Image.open(image_file) as uploaded_image:
        image = uploaded_image.convert("RGB")
    image_features = runtime.encode_images([image], batch_size=1)[0]
    clean_probability = runtime.cleanliness_probability(image_features)
    damage_probability = runtime.damage_probability(image_features)
    intact_probability = float(artifact["intact_model"].predict_proba(image_features.reshape(1, -1))[0, 1])
    thresholds = artifact["metadata"]["thresholds"]
    clean = clean_probability >= thresholds["clean"]
    if (
        clean_probability < settings.INTEGRITY_MIN_CLEAN_PROBABILITY
        and damage_probability < settings.DAMAGE_OVERRIDE_PROBABILITY
    ):
        intact = None
        intact_score = None
        integrity_status = "inconclusive"
    elif clean_probability < settings.INTEGRITY_MIN_CLEAN_PROBABILITY:
        intact = False
        intact_score = round((1 - damage_probability) * 100, 1)
        integrity_status = "possible_damage"
    else:
        intact = intact_probability >= thresholds["intact"]
        intact_score = round(intact_probability * 100, 1)
        integrity_status = "intact" if intact else "possible_damage"
    regions = load_localization_runtime().localize(
        image,
        include_dirt=not clean,
        include_damage=integrity_status == "possible_damage",
    )
    return {
        "clean": clean,
        "intact": intact,
        "integrity_status": integrity_status,
        "clean_score": round(clean_probability * 100, 1),
        "intact_score": intact_score,
        "explanation": _explanation(clean, integrity_status),
        "regions": regions,
        "localization_note": "Contours are segmentation estimates and should be confirmed by physical inspection.",
    }


def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)
    form = PredictionUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        message = form.errors.get("image", ["Invalid upload."])[0]
        return JsonResponse({"error": str(message)}, status=400)

    try:
        return JsonResponse(process_image(form.cleaned_data["image"]))
    except FileNotFoundError as error:
        return JsonResponse({"error": str(error)}, status=503)
    except Exception as error:
        return JsonResponse({"error": f"Image analysis failed: {error}"}, status=500)


def health(request):
    try:
        metadata = load_model()["metadata"]
        return JsonResponse({"status": "ok", "model": metadata})
    except Exception as error:
        return JsonResponse({"status": "unavailable", "error": str(error)}, status=503)
