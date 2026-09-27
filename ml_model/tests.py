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
    def __init__(self, damage_confidence: float | None = None):
        self.damage_confidence = damage_confidence

    def localize(self, image, *, include_dirt, include_damage):
        findings = []
        if include_damage and self.damage_confidence is not None:
            findings.append(self._finding("scratch", self.damage_confidence))
        if include_dirt:
            findings.append(self._finding("dirt", 88.0))
        return findings

    @staticmethod
    def _finding(finding_type, confidence):
        return {
            "id": f"{finding_type}-1",
            "type": finding_type,
            "confidence": confidence,
            "geometry": {
                "polygon": [{"x": 0.1, "y": 0.1}, {"x": 0.3, "y": 0.1}, {"x": 0.2, "y": 0.3}],
                "bounds": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            },
        }


class FixedDamageModel:
    def predict(self, image):
        missing_part = np.zeros((60, 80), dtype=bool)
        missing_part[25:55, 12:70] = True
        broken_lamp = np.zeros((60, 80), dtype=bool)
        broken_lamp[18:32, 12:30] = True
        return {
            "labels_en": ["Missing part", "Broken lamp"],
            "scores": np.asarray([0.98, 0.91]),
            "masks": np.asarray([missing_part, broken_lamp]),
        }


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

    def test_damage_masks_include_missing_parts_and_broken_lamps(self):
        runtime = object.__new__(LocalizationRuntime)
        runtime.damage_confidence = 0.35
        runtime.damage_duplicate_overlap = 0.65
        runtime.max_damage_regions = 8
        runtime.__dict__["damage_model"] = FixedDamageModel()

        findings = runtime._damage_findings(Image.new("RGB", (80, 60)))

        self.assertEqual([finding["type"] for finding in findings], ["missing_part", "lamp_broken"])
        self.assertTrue(all(len(finding["geometry"]["polygon"]) >= 4 for finding in findings))
        self.assertNotIn("glass_shatter", [finding["type"] for finding in findings])


class PredictionViewTests(TestCase):
    def test_home_is_english_and_has_csrf_token(self):
        response = self.client.get(reverse("home"))

        self.assertContains(response, "Analyze photo")
        self.assertContains(response, "Drop your vehicle photo here")
        self.assertContains(response, "Detected issues")
        self.assertContains(response, "csrfmiddlewaretoken")
        self.assertNotContains(response, "Condition scoring")
        self.assertNotContains(response, "Vision system online")

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
        mocked_localization.return_value = FixedLocalizationRuntime(damage_confidence=94.0)

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["intact"])
        self.assertEqual(payload["intact_score"], 20.0)
        self.assertEqual(payload["integrity_status"], "possible_damage")
        self.assertTrue(any(region["type"] == "scratch" for region in payload["regions"]))

    @patch("ml_model.views.load_localization_runtime")
    @patch("ml_model.views.load_vision_runtime")
    @patch("ml_model.views.load_model")
    def test_strong_damage_mask_overrides_inconclusive_global_score(
        self,
        mocked_load_model,
        mocked_vision,
        mocked_localization,
    ):
        mocked_load_model.return_value = artifact(0.05)
        mocked_vision.return_value = FixedVisionRuntime(0.1, damage_probability=0.2)
        mocked_localization.return_value = FixedLocalizationRuntime(damage_confidence=98.0)

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["intact"])
        self.assertEqual(payload["integrity_status"], "possible_damage")
        self.assertEqual(payload["intact_score"], 2.0)
        self.assertTrue(any(region["type"] == "scratch" for region in payload["regions"]))

    @patch("ml_model.views.load_model")
    def test_health_exposes_training_metadata(self, mocked_load_model):
        mocked_load_model.return_value = {"metadata": {"integrity_training_images": 4347}}

        response = self.client.get(reverse("health"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"]["integrity_training_images"], 4347)
