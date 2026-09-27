from __future__ import annotations

from functools import lru_cache

import joblib
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from .features import FEATURE_VERSION, extract_features
from .forms import PredictionUploadForm


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
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("The model feature version does not match the application")
    return artifact


def _explanation(clean: bool, intact: bool) -> str:
    if clean and intact:
        return "The car appears clean, and the model found no visible body damage."
    if not clean and intact:
        return "The model detected dirt but found no visible body damage."
    if clean and not intact:
        return "The car appears clean, but the model detected signs of body damage."
    return "The model detected dirt and signs of body damage. A manual inspection is recommended."


def process_image(image_file) -> dict[str, object]:
    features = extract_features(image_file.read()).reshape(1, -1)
    artifact = load_model()
    clean_probability = float(artifact["clean_model"].predict_proba(features)[0, 1])
    intact_probability = float(artifact["intact_model"].predict_proba(features)[0, 1])
    thresholds = artifact["metadata"]["thresholds"]
    clean = clean_probability >= thresholds["clean"]
    intact = intact_probability >= thresholds["intact"]
    return {
        "clean": clean,
        "intact": intact,
        "clean_score": round(clean_probability * 100, 1),
        "intact_score": round(intact_probability * 100, 1),
        "explanation": _explanation(clean, intact),
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
