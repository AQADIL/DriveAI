from django import forms
from django.conf import settings


class PredictionUploadForm(forms.Form):
    image = forms.ImageField(
        error_messages={
            "required": "Upload a car image.",
            "invalid_image": "The uploaded file is not a valid image.",
        }
    )

    def clean_image(self):
        image = self.cleaned_data["image"]
        if image.size > settings.MAX_IMAGE_BYTES:
            max_size_mb = settings.MAX_IMAGE_BYTES / (1024 * 1024)
            raise forms.ValidationError(f"The image must be no larger than {max_size_mb:g} MB.")
        if image.image.format not in settings.ALLOWED_IMAGE_FORMATS:
            formats = ", ".join(sorted(settings.ALLOWED_IMAGE_FORMATS))
            raise forms.ValidationError(f"Unsupported image format. Use one of: {formats}.")
        image.seek(0)
        return image
