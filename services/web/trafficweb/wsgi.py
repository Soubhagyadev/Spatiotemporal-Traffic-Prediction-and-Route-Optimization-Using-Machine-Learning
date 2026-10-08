"""WSGI entry point for the traffic portal."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trafficweb.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
