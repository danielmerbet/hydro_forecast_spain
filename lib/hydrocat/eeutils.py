"""Google Earth Engine helpers."""
from __future__ import annotations

import time

import ee

from .config import load_settings

_INITIALISED = False


def init_ee() -> None:
    """Initialise Earth Engine once, with the Cloud project from settings.yaml.

    Credentials come from `earthengine authenticate` / application-default
    login, or — in GitHub Actions — from a service-account key passed in the
    environment variable EE_SERVICE_ACCOUNT_KEY (see OPERATIONS.md).
    """
    global _INITIALISED
    if not _INITIALISED:
        import json
        import os
        key = os.environ.get("EE_SERVICE_ACCOUNT_KEY")          # GitHub Actions secret (JSON text)
        project = load_settings()["ee_project"]
        if key:
            info = json.loads(key)
            creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key)
            ee.Initialize(creds, project=project)
        else:
            ee.Initialize(project=project)
        _INITIALISED = True


def retry(fn, *args, tries: int = 6, wait: float = 10.0, **kwargs):
    """Call an Earth Engine request, retrying on transient errors (429/5xx)."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except ee.EEException as e:
            msg = str(e)
            transient = any(s in msg.lower() for s in ("429", "too many", "timed out", "500", "503", "internal error",
                                                       "deadline", "unavailable"))
            if not transient or i == tries - 1:
                raise
            print(f"    EE transient error ({msg[:80]}...), retry {i + 1}/{tries}", flush=True)
            time.sleep(wait * (i + 1))
