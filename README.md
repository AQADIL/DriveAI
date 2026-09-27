# DriveAI

DriveAI is a Django application that estimates two exterior car conditions from a photo:

- cleanliness;
- visible body integrity.

The prediction endpoint uses two trained scikit-learn classifiers. It does not contain random answers, filename rules, or a heuristic fallback. If the model artifact is missing or incompatible, the API returns an explicit service error.

## What the model actually uses

The training pipeline reads every available labeled image in `fresh_data`:

| Source | Available images | Labels used |
| --- | ---: | --- |
| `train/labels.csv` | 1,610 | cleanliness and integrity |
| `val/labels.csv` | 460 | cleanliness and integrity |
| `data1a/training` | 1,840 | integrity |
| `data1a/validation` | 460 | integrity |
| `test/labels.csv` | 0 of 230 | none; referenced image files are missing |

The pipeline first evaluates on the supplied validation splits, selects a decision threshold for each task by balanced accuracy, and then fits deployable models on all 4,370 available images. The generated metrics report records counts, skipped files, thresholds, accuracy, balanced accuracy, F1, and ROC AUC.

The model is an image classifier, not a safety inspection. Its ability to generalize is limited by the cars, camera angles, lighting, damage types, and label quality represented in the repository.

## Setup

The commands below target PowerShell on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Paste the generated secret into `DJANGO_SECRET_KEY` in `.env`, then set the remaining required values. A local configuration can use:

```dotenv
DJANGO_DEBUG=true
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
DJANGO_CSRF_TRUSTED_ORIGINS=
DJANGO_DATABASE_PATH=db.sqlite3
DJANGO_TIME_ZONE=UTC
MAX_IMAGE_BYTES=5242880
ALLOWED_IMAGE_FORMATS=JPEG,PNG,WEBP
CAR_DATA_ROOT=fresh_data
CAR_MODEL_PATH=ml_model/car_condition.joblib
APP_HOST=127.0.0.1
APP_PORT=8000
DJANGO_SECURE_HSTS_SECONDS=0
DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS=false
DJANGO_SECURE_HSTS_PRELOAD=false
DJANGO_SECURE_SSL_REDIRECT=false
DJANGO_SESSION_COOKIE_SECURE=false
DJANGO_CSRF_COOKIE_SECURE=false
```

Do not commit `.env`; it is ignored by Git.

## Train

```powershell
.\.venv\Scripts\python.exe -m ml_model.training
```

Training writes two ignored local artifacts:

- `ml_model/car_condition.joblib` — fitted classifiers and metadata;
- `ml_model/car_condition.metrics.json` — readable validation and dataset report.

Set `CAR_DATA_ROOT` and `CAR_MODEL_PATH` to different locations when the data or artifacts live outside the repository.

## Run

```powershell
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe run_server.py
```

Open the host and port configured through `APP_HOST` and `APP_PORT`.

## Endpoints

- `GET /` — upload interface;
- `POST /predict/` — same-origin multipart prediction request with an `image` field;
- `GET /health/` — model availability, dataset counts, thresholds, and validation metrics.

Successful predictions return deterministic model scores:

```json
{
  "clean": true,
  "intact": false,
  "clean_score": 78.4,
  "intact_score": 31.7,
  "explanation": "The car appears clean, but the model detected signs of body damage."
}
```

## Project layout

```text
indrive_car_check/       Django configuration
ml_model/features.py    Deterministic image feature extraction
ml_model/training.py    Dataset loading, validation, threshold selection, final fit
ml_model/forms.py       Upload and image validation
ml_model/views.py       Model loading and prediction API
fresh_data/             Repository datasets
run_server.py           Environment-driven local server entry point
```

## Quality checks

```powershell
.\.venv\Scripts\python.exe -m compileall -q indrive_car_check ml_model
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test
```

For a production deployment, use a production WSGI/ASGI server, set `DJANGO_DEBUG=false`, provide appropriate host and CSRF origin lists, enable the secure cookie and HTTPS settings from `.env.example`, and run Django's deployment checks. Enable HSTS only after HTTPS is working for every relevant hostname.
