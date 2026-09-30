"""Load the catchment forcing, applying the configured precipitation correction."""
from __future__ import annotations

import pandas as pd

from .config import PROCESSED, ROOT, P, load_settings


def load_forcing(domain: str = "catalonia", mode: str | None = None, columns=None) -> pd.DataFrame:
    """ERA5-Land catchment forcing (script 03), with `total_precipitation`
    rescaled by the monthly EMO-1 factors of script 03d when
    settings.yaml forcing.precip_correction == "emo1_monthly_scaling".
    The uncorrected value is kept as `total_precipitation_raw`."""
    S = load_settings()
    f = pd.read_parquet(P.forcing if domain == "catalonia" else ROOT / S["domains"][domain]["forcing"],
                        columns=columns)
    mode = mode or S["forcing"].get("precip_correction", "none")
    if mode == "none" or "total_precipitation" not in f:
        return f
    if mode != "emo1_monthly_scaling":
        raise ValueError(f"unknown precip_correction {mode!r}")
    fac = pd.read_csv((PROCESSED if domain == "catalonia" else PROCESSED / domain) / "precip_scaling_emo1.csv")
    f["month"] = f.date.dt.month
    f = f.merge(fac, on=["gauge_id", "month"], how="left")
    if f.factor.isna().any():
        missing = sorted(f.loc[f.factor.isna(), "gauge_id"].unique())
        raise RuntimeError(f"no EMO-1 scaling factors for {missing}: run codes/03d first")
    f["total_precipitation_raw"] = f.total_precipitation
    f["total_precipitation"] = f.total_precipitation * f.factor
    return f.drop(columns=["month", "factor"])
