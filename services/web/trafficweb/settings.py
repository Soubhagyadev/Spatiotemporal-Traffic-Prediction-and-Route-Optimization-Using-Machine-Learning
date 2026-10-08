"""Django settings for the traffic portal.

The portal is a pure presentation layer: it owns no models and reads every
number from the Flask API through ``services.common.TrafficApiClient``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = WEB_DIR.parent.parent  # repository root

# The project is importable as ``services.web.trafficweb.settings`` from the
# repository root, but the app label stays the plain ``portal``; the repository
# root is needed for the shared ``services.common`` API client.
for path in (WEB_DIR, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "django-insecure-dev-only-spatiotemporal-traffic-portal"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() not in {"0", "false", "no"}
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "0.0.0.0"]

INSTALLED_APPS = ["django.contrib.staticfiles", "portal"]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "trafficweb.urls"
WSGI_APPLICATION = "trafficweb.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.debug",
            ]
        },
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "data" / "django.sqlite3",
    }
}

STATIC_URL = "static/"
USE_TZ = False

TRAFFIC_API_URL = os.environ.get("TRAFFIC_API_URL", "http://127.0.0.1:5001")
TRAFFIC_DASHBOARD_URL = os.environ.get("TRAFFIC_DASHBOARD_URL", "http://127.0.0.1:8050")
