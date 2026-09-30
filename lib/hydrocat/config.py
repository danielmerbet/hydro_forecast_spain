"""Settings and canonical file locations.

Every file the pipeline reads or writes is named here, so a reader can see the
whole data flow in one place and scripts never disagree about a path.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"


class P:
    """Canonical paths of the pipeline's products (in the order they are made)."""

    # 01 — gauges
    stations = PROCESSED / "stations.csv"
    q_daily = PROCESSED / "q_obs_daily.parquet"            # m3/s, wide: date x gauge
    # 02 — catchments
    flowdir = INTERIM / "hydrosheds_03dir.tif"
    catchments = PROCESSED / "catchments.gpkg"
    # 03 — attributes
    attributes = PROCESSED / "catchment_attributes.csv"
    # 04 — forcing
    forcing = PROCESSED / "forcing_era5land.parquet"         # long: date, gauge, variables
    # 05 — GR4J
    gr4j_params = PROCESSED / "gr4j" / "parameters.csv"
    gr4j_sim = PROCESSED / "gr4j" / "q_sim_daily.parquet"   # m3/s, wide
    # 06/07 — Google
    google_dir = PROCESSED / "google"
    google_sim = PROCESSED / "google" / "q_sim_daily_{run}.parquet"
    # 08 — comparison
    metrics = TABLES / "metrics_test_period.csv"


for _d in (RAW, INTERIM, PROCESSED, FIGURES, TABLES, PROCESSED / "gr4j", PROCESSED / "google"):
    _d.mkdir(parents=True, exist_ok=True)


@lru_cache
def load_settings() -> dict:
    with open(CONFIG / "settings.yaml") as f:
        s = yaml.safe_load(f)
    # YAML parses bare dates to datetime.date; normalise to ISO strings, and
    # resolve the "today" sentinel.
    for k, v in s["periods"].items():
        if v == "today":
            v = dt.date.today()
        s["periods"][k] = str(v)
    for k, v in s["gauges"].items():
        if isinstance(v, dt.date):
            s["gauges"][k] = str(v)
    return s
