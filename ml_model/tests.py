import io
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from .localization import LocalizationRuntime
from .training import ARTIFACT_VERSION, collect_samples


class FixedClassifier:
    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def predict_proba(self, features):
        probability = self.positive_probability
        return np.asarray([[1 - probability, probability]], dtype=np.float64)


class FixedVisionRuntime:
    def __init__(
        self,
        clean_probability: float,
        damage_probability: float = 0.1,
    ):
        self.clean_probability = clean_probability
        self.damage_probability_value = damage_probability

    def encode_images(self, images, batch_size=1):
        return np.ones((len(images), 512), dtype=np.float32)

    def cleanliness_probability(self, image_features):
        return self.clean_probability

    def damage_probability(self, image_features):
        return self.damage_probability_value



class FixedLocalizationRuntime:
    def localize(self, image, *, include_dirt, include_damage):
        finding_type = "scratch" if include_damage else "dirt"
        if not include_dirt and not include_damage:
            return []
        return [{
            "id": "finding-1",
            "type": finding_type,
            "geometry": {
                "polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.2, "y": 0.3}],
                "bounds": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            },
        }]


def image_upload(name: str = "car.jpg") -> SimpleUploadedFile:
    output = io.BytesIO()
    Image.new("RGB", (80, 60), color=(120, 140, 160)).save(output, format="JPEG")
    return SimpleUploadedFile(name, output.getvalue(), content_type="image/jpeg")


def artifact(intact_probability: float = 0.8):
    return {
        "artifact_version": ARTIFACT_VERSION,
        "intact_model": FixedClassifier(intact_probability),
        "metadata": {
            "vision_model_id": "openai/clip-vit-base-patch32",
            "thresholds": {"clean": 0.5, "intact": 0.5},
        },
    }


class DatasetTests(TestCase):
    def test_duplicate_images_are_removed(self):
        samples, report = collect_samples()

        self.assertEqual(len(samples), 4347)
        self.assertEqual(report["exact_duplicates_removed"], 23)
        self.assertEqual(report["conflicting_duplicates"], 0)
        self.assertEqual(report["test_csv_missing"], 230)


class LocalizationTests(TestCase):
    def test_polygon_geometry_includes_normalized_bounds(self):
        geometry = LocalizationRuntime._geometry(np.asarray([[0.1, 0.2], [0.7, 0.3], [0.4, 0.8]]))

        self.assertEqual(len(geometry["polygon"]), 3)
        self.assertEqual(geometry["bounds"], {"x": 0.1, "y": 0.2, "width": 0.6, "height": 0.6})


class PredictionViewTests(TestCase):
    def test_home_is_english_and_has_csrf_token(self):
        response = self.client.get(reverse("home"))

        self.assertContains(response, "Analyze photo")
        self.assertContains(response, "Drop your vehicle photo here")
        self.assertContains(response, "Detected attention zones")
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_invalid_upload_is_rejected(self):
        upload = SimpleUploadedFile("car.txt", b"not an image", content_type="text/plain")
        response = self.client.post(reverse("predict"), {"image": upload})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    @patch("ml_model.views.load_localization_runtime")
    @patch("ml_model.views.load_vision_runtime")
    @patch("ml_model.views.load_model")
    def test_clean_car_uses_integrity_head(self, mocked_load_model, mocked_vision, mocked_localization):
        mocked_load_model.return_value = artifact(0.82)
        mocked_vision.return_value = FixedVisionRuntime(0.91)
        mocked_localization.return_value = FixedLocalizationRuntime()

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["clean"])
        self.assertTrue(payload["intact"])
        self.assertEqual(payload["integrity_status"], "intact")
        self.assertEqual(payload["clean_score"], 91.0)
        self.assertEqual(payload["intact_score"], 82.0)

    @patch("ml_model.views.load_localization_runtime")
    @patch("ml_model.views.load_vision_runtime")
    @patch("ml_model.views.load_model")
    def test_heavy_dirt_withholds_integrity_claim(self, mocked_load_model, mocked_vision, mocked_localization):
        mocked_load_model.return_value = artifact(0.05)
        mocked_vision.return_value = FixedVisionRuntime(0.01)
        mocked_localization.return_value = FixedLocalizationRuntime()

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["clean"])
        self.assertIsNone(payload["intact"])
        self.assertIsNone(payload["intact_score"])
        self.assertEqual(payload["integrity_status"], "inconclusive")
        self.assertGreater(len(payload["regions"]), 0)
        self.assertTrue(all(region["type"] == "dirt" for region in payload["regions"]))

    @patch("ml_model.views.load_localization_runtime")
    @patch("ml_model.views.load_vision_runtime")
    @patch("ml_model.views.load_model")
    def test_obvious_damage_overrides_dirt_gate(self, mocked_load_model, mocked_vision, mocked_localization):
        mocked_load_model.return_value = artifact(0.05)
        mocked_vision.return_value = FixedVisionRuntime(0.1, damage_probability=0.8)
        mocked_localization.return_value = FixedLocalizationRuntime()

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["intact"])
        self.assertEqual(payload["intact_score"], 20.0)
        self.assertEqual(payload["integrity_status"], "possible_damage")
        self.assertTrue(any(region["type"] == "scratch" for region in payload["regions"]))

    @patch("ml_model.views.load_model")
    def test_health_exposes_training_metadata(self, mocked_load_model):
        mocked_load_model.return_value = {"metadata": {"integrity_training_images": 4347}}

        response = self.client.get(reverse("health"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"]["integrity_training_images"], 4347)
