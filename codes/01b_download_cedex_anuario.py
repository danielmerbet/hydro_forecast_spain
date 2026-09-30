#!/usr/bin/env python3
"""
01b — Historical daily streamflow at every Spanish ROEA/SAIH gauge (CEDEX Anuario de Aforos).

SOURCE
  CEDEX (Centro de Estudios Hidrográficos) — Anuario de Aforos, edition
  2021-2022 (published April 2026), one set of CSV tables per river basin
  district (demarcación hidrográfica):
      https://ceh.cedex.es/anuarioaforos/demarcaciones.asp
  Tables used (see Descripción_Tablas_Mto.pdf in each folder):
      estaf.csv   river gauging stations: code (indroea), name, drained area
                  (suprest, km²), coordinates, SAIH code (cod_saih)
      afliq.csv   daily data: indroea; fecha (dd/mm/yyyy); altura (m); caudal (m³/s)
  Coverage ends on 2022-09-30 for every station. More recent data come from
  each basin authority's SAIH (scripts 01c_*).

  The internal basins of Catalonia (ACA), the Basque Country, Galicia-Costa and
  Andalusia are managed by regional agencies; Catalonia comes from script 01.

COORDINATES
  longwgs84 / latwgs84 are packed sexagesimal integers: -25720 = -2°57'20",
  424116 = 42°41'16". Decoded here to decimal degrees.

Writes
  data/processed/spain/stations_cedex.csv       one row per gauge with data
  data/processed/spain/q_obs_daily_cedex.parquet daily discharge, m³/s (date x gauge), 1980-10-01 → 2022-09-30
Caches
  data/raw/cedex/<DEMARCACION>/{estaf,afliq}.csv
"""
from __future__ import annotations

import re
import subprocess
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from hydrocat.config import PROCESSED, RAW, ROOT

BASE = "https://ceh.cedex.es/anuarioaforos/"
DEMARCACIONES = ["CANTABRICO", "DUERO", "EBRO", "GALICIA COSTA", "GUADALQUIVIR", "GUADIANA",
                 "JUCAR", "MINO_SIL", "SEGURA", "TAJO"]
START = "1980-10-01"
OUT = PROCESSED / "spain"
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def get(url: str, timeout: int = 120) -> bytes:
    """Download with curl: the CEDEX server resets Python's TLS handshake
    (old cipher suite) but works with curl, which every Linux runner has."""
    r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "4", "--max-time", str(timeout), url],
                       capture_output=True, check=True)
    return r.stdout


def page_links(dem: str) -> dict[str, str]:
    """Scrape the per-demarcación page for its CSV links (their folder names vary)."""
    for page in (f"{dem}_csv.asp", f"{dem}.asp"):
        try:
            html = get(BASE + urllib.parse.quote(page), 60).decode("latin1")
        except Exception:
            continue
        links = re.findall(r'href="([^"]+\.csv)"', html, flags=re.I)
        if links:
            return {l.replace("\\", "/").rsplit("/", 1)[-1].lower(): l.replace("\\", "/") for l in links}
    raise RuntimeError(f"no CSV links found for {dem}")


def fetch(dem: str, name: str) -> pd.DataFrame:
    f = RAW / "cedex" / dem.replace(" ", "_") / name
    if not f.exists():
        url = page_links(dem)[name]
        f.parent.mkdir(parents=True, exist_ok=True)
        log(f"  downloading {dem}/{name}")
        f.write_bytes(get(urllib.parse.quote(url, safe=":/"), 900))
    return pd.read_csv(f, sep=";", encoding="latin1", low_memory=False)


def dms(v) -> float:
    """-25720 -> -(2 + 57/60 + 20/3600)."""
    if pd.isna(v):
        return np.nan
    v = int(v)
    a = abs(v)
    return np.sign(v) * (a // 10000 + (a // 100 % 100) / 60 + (a % 100) / 3600)


def main():
    stations, series = [], []
    for dem in DEMARCACIONES:
        log(f"== {dem}")
        st = fetch(dem, "estaf.csv")
        q = fetch(dem, "afliq.csv")
        q["fecha"] = pd.to_datetime(q["fecha"], format="%d/%m/%Y", errors="coerce")
        q["caudal"] = pd.to_numeric(q["caudal"], errors="coerce")
        q = q[(q.fecha >= START) & q.caudal.notna()]
        w = q.pivot_table(index="fecha", columns="indroea", values="caudal")
        w.columns = [f"ES{c}" for c in w.columns]
        series.append(w)
        st = st.assign(
            gauge_id="ES" + st["indroea"].astype(str),
            name=st["lugar"].astype(str).str.strip().str.title(),
            demarcacion=dem.replace("_", "-").title(),
            lon=st["longwgs84"].map(dms), lat=st["latwgs84"].map(dms),
            area_official_km2=pd.to_numeric(st["suprest"], errors="coerce"),
            cod_saih=st["cod_saih"].astype(str).str.strip().replace({"nan": ""}),
        )
        st = st[st.gauge_id.isin(w.columns)]
        n = w.notna().sum()
        st["first_obs"] = st.gauge_id.map(lambda g: w[g].first_valid_index().date())
        st["last_obs"] = st.gauge_id.map(lambda g: w[g].last_valid_index().date())
        st["n_days"] = st.gauge_id.map(n)
        st["n_days_2003_2022"] = st.gauge_id.map(w.loc["2003-10-01":].notna().sum())
        stations.append(st[["gauge_id", "indroea", "name", "demarcacion", "lon", "lat", "area_official_km2",
                            "cod_saih", "first_obs", "last_obs", "n_days", "n_days_2003_2022"]])
        log(f"   {len(st)} gauges with discharge since {START[:4]}")

    st = pd.concat(stations, ignore_index=True)
    q = pd.concat(series, axis=1).sort_index()
    q = q.reindex(pd.date_range(q.index.min(), "2022-09-30", freq="D")).clip(lower=0)
    q.index.name = "date"
    q.to_parquet(OUT / "q_obs_daily_cedex.parquet")
    st.to_csv(OUT / "stations_cedex.csv", index=False)
    log(f"-> {len(st)} gauges, {OUT.relative_to(ROOT)}/")
    log(st.groupby("demarcacion").agg(gauges=("gauge_id", "size"),
                                      with_5yr_since_2003=("n_days_2003_2022", lambda s: int((s >= 1825).sum())),
                                      with_saih_code=("cod_saih", lambda s: int((s != "").sum()))).to_string())


if __name__ == "__main__":
    main()
