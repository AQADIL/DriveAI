import io
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from .features import extract_features
from .training import collect_samples


class FixedClassifier:
    def __init__(self, positive_probability: float):
        self.positive_probability = positive_probability

    def predict_proba(self, features):
        probability = self.positive_probability
        return np.asarray([[1 - probability, probability]], dtype=np.float64)


def image_upload(name: str = "car.jpg") -> SimpleUploadedFile:
    output = io.BytesIO()
    Image.new("RGB", (80, 60), color=(120, 140, 160)).save(output, format="JPEG")
    return SimpleUploadedFile(name, output.getvalue(), content_type="image/jpeg")


class FeatureTests(TestCase):
    def test_features_are_deterministic_and_finite(self):
        image_bytes = image_upload().read()
        first = extract_features(image_bytes)
        second = extract_features(image_bytes)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(first.size, 100)


class DatasetTests(TestCase):
    def test_all_available_images_are_collected(self):
        samples, missing = collect_samples()

        self.assertEqual(len(samples), 4370)
        self.assertEqual(sum(sample.clean is not None for sample in samples), 2070)
        self.assertEqual(missing["test_csv_missing"], 230)


class PredictionViewTests(TestCase):
    def test_home_is_english_and_has_csrf_token(self):
        response = self.client.get(reverse("home"))

        self.assertContains(response, "Analyze photo")
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_invalid_upload_is_rejected(self):
        upload = SimpleUploadedFile("car.txt", b"not an image", content_type="text/plain")
        response = self.client.post(reverse("predict"), {"image": upload})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    @patch("ml_model.views.load_model")
    def test_prediction_uses_model_scores_and_saved_thresholds(self, mocked_load_model):
        mocked_load_model.return_value = {
            "clean_model": FixedClassifier(0.82),
            "intact_model": FixedClassifier(0.34),
            "metadata": {"thresholds": {"clean": 0.61, "intact": 0.48}},
        }

        response = self.client.post(reverse("predict"), {"image": image_upload()})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["clean"])
        self.assertFalse(payload["intact"])
        self.assertEqual(payload["clean_score"], 82.0)
        self.assertEqual(payload["intact_score"], 34.0)
        self.assertNotIn("confidence", payload)

    @patch("ml_model.views.load_model")
    def test_health_exposes_training_metadata(self, mocked_load_model):
        mocked_load_model.return_value = {"metadata": {"available_images": 4370}}

        response = self.client.get(reverse("health"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"]["available_images"], 4370)
