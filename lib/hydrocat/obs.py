"""Observed daily discharge (m3/s, date x gauge) as used by the forecast scripts."""
from __future__ import annotations

import pandas as pd

from .config import PROCESSED, P

# SAIH tables of script 01c (columns already renamed to CEDEX gauge ids)
SAIH_BASINS = ["ebro", "jucar", "guadalquivir", "segura"]


def observations(domain: str) -> pd.DataFrame:
    """Catalonia: the ACA table of script 01 (export + open data).
    Spain: the SAIH tables of script 01c, only for gauges whose SAIH series was
    joined to CEDEX (script 01d: "verified" or "unverified"; "rejected" joins,
    where the two sources disagree, are left out)."""
    if domain == "catalonia":
        q = pd.read_parquet(P.q_daily)
        q.index = pd.DatetimeIndex(q.index).as_unit("ns")
        return q
    base = PROCESSED / domain
    st = pd.read_csv(base / "stations.csv", usecols=["gauge_id", "saih_join"])
    ok = set(st.gauge_id[st.saih_join.isin(["verified", "unverified"])])
    parts = []
    for b in SAIH_BASINS:
        f = base / f"q_obs_daily_saih_{b}.parquet"
        if f.exists():
            q = pd.read_parquet(f)
            parts.append(q[[c for c in q.columns if c in ok]])
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    q = pd.concat(parts, axis=1)
    q = q.loc[:, ~q.columns.duplicated()]
    q.index = pd.DatetimeIndex(q.index).as_unit("ns")
    return q.sort_index()
