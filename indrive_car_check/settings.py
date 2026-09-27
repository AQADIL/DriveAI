import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_bool(name: str) -> bool:
    value = required_env(name).lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"{name} must be either true or false")
    return value == "true"


def env_list(name: str) -> list[str]:
    return [item.strip() for item in required_env(name).split(",") if item.strip()]


def project_path(name: str) -> Path:
    value = Path(required_env(name))
    return value if value.is_absolute() else (BASE_DIR / value).resolve()


SECRET_KEY = required_env("DJANGO_SECRET_KEY")
DEBUG = env_bool("DJANGO_DEBUG")
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS")
CSRF_TRUSTED_ORIGINS = [
    item.strip()
    for item in os.getenv("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if item.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ml_model",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "indrive_car_check.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "indrive_car_check.wsgi.application"
ASGI_APPLICATION = "indrive_car_check.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": project_path("DJANGO_DATABASE_PATH"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = required_env("DJANGO_TIME_ZONE")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MAX_IMAGE_BYTES = int(required_env("MAX_IMAGE_BYTES"))
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_IMAGE_BYTES
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_IMAGE_BYTES
DATA_UPLOAD_MAX_NUMBER_FILES = 1
ALLOWED_IMAGE_FORMATS = frozenset(env_list("ALLOWED_IMAGE_FORMATS"))
CAR_MODEL_PATH = project_path("CAR_MODEL_PATH")
VISION_MODEL_ID = required_env("VISION_MODEL_ID")
VISION_MODEL_CACHE_DIR = project_path("VISION_MODEL_CACHE_DIR")
VISION_DEVICE = required_env("VISION_DEVICE")
VISION_BATCH_SIZE = int(required_env("VISION_BATCH_SIZE"))
INTEGRITY_MIN_CLEAN_PROBABILITY = float(required_env("INTEGRITY_MIN_CLEAN_PROBABILITY"))
DAMAGE_OVERRIDE_PROBABILITY = float(required_env("DAMAGE_OVERRIDE_PROBABILITY"))
DAMAGE_SEGMENTATION_MODEL_ID = required_env("DAMAGE_SEGMENTATION_MODEL_ID")
DAMAGE_SEGMENTATION_CONFIDENCE = float(required_env("DAMAGE_SEGMENTATION_CONFIDENCE"))
DAMAGE_LOCALIZATION_OVERRIDE_CONFIDENCE = float(required_env("DAMAGE_LOCALIZATION_OVERRIDE_CONFIDENCE"))
DAMAGE_MASK_DUPLICATE_OVERLAP = float(required_env("DAMAGE_MASK_DUPLICATE_OVERLAP"))
MAX_DAMAGE_REGIONS = int(required_env("MAX_DAMAGE_REGIONS"))
DIRT_SEGMENTATION_MODEL_ID = required_env("DIRT_SEGMENTATION_MODEL_ID")
DIRT_SEGMENTATION_THRESHOLD = float(required_env("DIRT_SEGMENTATION_THRESHOLD"))
DIRT_SEGMENTATION_QUANTILE = float(required_env("DIRT_SEGMENTATION_QUANTILE"))
DIRT_SEGMENTATION_MIN_AREA = float(required_env("DIRT_SEGMENTATION_MIN_AREA"))
MAX_DIRT_REGIONS = int(required_env("MAX_DIRT_REGIONS"))

SECURE_HSTS_SECONDS = int(required_env("DJANGO_SECURE_HSTS_SECONDS"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS")
SECURE_HSTS_PRELOAD = env_bool("DJANGO_SECURE_HSTS_PRELOAD")
SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT")
SESSION_COOKIE_SECURE = env_bool("DJANGO_SESSION_COOKIE_SECURE")
CSRF_COOKIE_SECURE = env_bool("DJANGO_CSRF_COOKIE_SECURE")
