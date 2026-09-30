#!/usr/bin/env python3
"""
12 — GR4J forecast for today's runs (WeatherNext 3 and AIFS).

1. STATES. GR4J is run from the start of the warm-up with ERA5-Land. ERA5-Land
   ends ~7 days before today, so the gap up to the forecast start is BRIDGED
   with WeatherNext 3's first forecast day of each daily 00 UTC run ("short
   forecast as analysis"), cached per date in data/forecasts/<domain>/bridge/.
   The same bridge is used for every forecast product, so all start from the
   same model states.
2. FORECASTS, 15 days, per gauge:
     wn3        WeatherNext 3 ensemble-mean forcing
     aifs       AIFS deterministic
     aifs_mNN   50 AIFS-ENS members: member rain, other variables from AIFS
                deterministic (only rain is published for the members here)
   The member runs are summarised as quantiles q05/q25/q50/q75/q95.
   (WeatherNext's hourly p10/p90 are NOT used: summing hourly percentiles gave
   "p90" floods 50x the ensemble mean — not a meaningful daily percentile.)
   STATE UPDATING: on the last day with an observation (up to 3 days before the
   forecast start) the routing store is reset so that GR4J reproduces the
   observed discharge (GRP-style updating, Berthet et al., 2009). Without it,
   forecasts after a dry summer start from a nearly empty store and
   underestimate the response to rain.
3. Parameters from the calibration with RAW ERA5-Land rain
   (data/processed/gr4j/run_era5l_raw), i.e. the forcing type the forecasts
   also are; zero-flow thresholds (q0) applied as in calibration.

Writes  data/forecasts/<domain>/gr4j/<init>/q_forecast.parquet
        long: date, gauge_id, product, q (m3/s) [+ quantiles for aifs_ens]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

from hydrocat.config import DATA, PROCESSED, ROOT, load_settings
from hydrocat.eeutils import init_ee
from hydrocat.forcing import load_forcing
from hydrocat.gr4j import CatchmentModel

S = load_settings()
GR = S["gr4j"]
VARS = ["total_precipitation", "temperature_2m", "temperature_2m_min", "temperature_2m_max", "pet_fao56"]


def log(*a):
    print(*a, flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "codes" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bridge(domain, first: dt.date, last: dt.date) -> pd.DataFrame:
    """Daily forcing for first..last from WeatherNext 3 day 0 of each date's 00 UTC run."""
    s10 = _load("10_fetch_weathernext3")
    cache = DATA / "forecasts" / domain / "bridge"
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for d in pd.date_range(first, last, freq="D").date:
        f = cache / f"{d:%Y%m%d}.parquet"
        if not f.exists():
            run = DATA / "forecasts" / domain / "weathernext3" / f"{d:%Y%m%d}00" / "forcing.parquet"
            if not run.exists():
                import subprocess
                subprocess.run(["python", str(ROOT / "codes" / "10_fetch_weathernext3.py"),
                                "--domain", domain, "--init", d.isoformat()], check=True)
            x = pd.read_parquet(run)
            x[x.date == pd.Timestamp(d)].to_parquet(f, index=False)
        out.append(pd.read_parquet(f))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--init", default=None, help="YYYYMMDDHH of the forecast runs (default: latest common)")
    args = ap.parse_args()
    D = S["domains"][args.domain]
    base = PROCESSED if args.domain == "catalonia" else PROCESSED / args.domain
    fdir = DATA / "forecasts" / args.domain
    inits = sorted(set(p.name for p in (fdir / "weathernext3").iterdir()) & set(p.name for p in (fdir / "aifs").iterdir()))
    init = args.init or inits[-1]
    d0 = pd.Timestamp(dt.datetime.strptime(init, "%Y%m%d%H").date())
    log(f"GR4J forecast, runs of {init}, domain {args.domain}")

    era = load_forcing(args.domain, "none")
    era = era[era.date >= S["periods"]["warmup_start"]]
    last_era = era.date.max()
    br = bridge(args.domain, (last_era + pd.Timedelta(days=1)).date(), (d0 - pd.Timedelta(days=1)).date())
    log(f"   ERA5-Land to {last_era.date()}, bridge {len(br.date.unique()) if len(br) else 0} days")
    past = pd.concat([era[["date", "gauge_id", *VARS]], br[["date", "gauge_id", *VARS]]], ignore_index=True)

    wn3 = pd.read_parquet(fdir / "weathernext3" / init / "forcing.parquet")
    aifs = pd.read_parquet(fdir / "aifs" / init / "forcing.parquet")
    ens_f = fdir / "aifs" / init / "precip_members.parquet"
    ens = pd.read_parquet(ens_f) if ens_f.exists() else None

    rawdir = base / "gr4j" / "run_era5l_raw"
    par = pd.read_csv((rawdir if rawdir.exists() else base / "gr4j") / "parameters.csv").set_index("gauge_id")
    hy = pd.read_csv(base / "hypsometry.csv").set_index("gauge_id")
    cat = gpd.read_file(ROOT / D["catchments"]).set_index("gauge_id")
    q_obs = pd.read_parquet(base / "q_obs_daily.parquet")

    rows = []
    for gid, p in par.iterrows():
        k = cat.area_km2[gid] / 86.4
        hist = past[past.gauge_id == gid].sort_values("date")
        scen = {"wn3": wn3, "aifs": aifs}
        if ens is not None:
            for mnum, e in ens[ens.gauge_id == gid].groupby("member"):
                a = aifs[aifs.gauge_id == gid].drop(columns="total_precipitation").merge(
                    e[["date", "total_precipitation"]], on="date")
                scen[f"aifs_m{mnum:02d}"] = a
        params = [p.X1, p.X2, p.X3, p.X4, p.CTG, p.Kf]
        o = q_obs[gid][(d0 - pd.Timedelta(days=3)):(d0 - pd.Timedelta(days=1))].dropna() if gid in q_obs else []
        upd_date, upd_mm = (o.index[-1], float(o.iloc[-1]) / k) if len(o) else (None, 0.0)
        for name, fc in scen.items():
            fc = fc[fc.gauge_id == gid] if "gauge_id" in fc else fc
            s = pd.concat([hist, fc[["date", "gauge_id", *VARS]]]).sort_values("date")
            m = CatchmentModel(s.total_precipitation.values, s.temperature_2m.values, s.temperature_2m_min.values,
                               s.temperature_2m_max.values, s.pet_fao56.values, hy.loc[gid].values.astype(float),
                               GR["n_elevation_layers"])
            ui = int(np.searchsorted(s.date.values, np.datetime64(upd_date))) if upd_date is not None else -1
            q = m.run(params, ui, upd_mm) * k
            q = np.where(q < p.q0_m3s, 0.0, q)
            sel = s.date.values >= (d0 - pd.Timedelta(days=60)).to_datetime64()       # keep 60 past days for context
            rows.append(pd.DataFrame({"date": s.date.values[sel], "gauge_id": gid, "product": name, "q": q[sel]}))
    q = pd.concat(rows, ignore_index=True)
    mem = q[q["product"].str.startswith("aifs_m")]
    if len(mem):
        qs = mem.groupby(["date", "gauge_id"]).q.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        qs.columns = ["q05", "q25", "q50", "q75", "q95"]
        qs = qs.reset_index().assign(product="aifs_ens")
        q = pd.concat([q[~q["product"].str.startswith("aifs_m")], qs], ignore_index=True)
    out = fdir / "gr4j" / init
    out.mkdir(parents=True, exist_ok=True)
    q.to_parquet(out / "q_forecast.parquet", index=False)
    log(f"-> {out.relative_to(ROOT)}/q_forecast.parquet ({q.gauge_id.nunique()} gauges, "
        f"products {sorted(q['product'].unique())})")


if __name__ == "__main__":
    main()
