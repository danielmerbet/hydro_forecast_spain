#!/usr/bin/env python3
"""
12 — GR4J forecast for today's runs (WeatherNext 3 and AIFS).

1. STATES. GR4J is run from the start of the warm-up with ERA5-Land. ERA5-Land
   ends ~7 days before today, so the gap up to the forecast start is BRIDGED
   with the first forecast day of each daily 00 UTC run ("short forecast as
   analysis"): WeatherNext 3 when this account can read it, otherwise AIFS
   (past runs from ECMWF's AWS mirror, script 11 --date --days 1). Cached per
   date in data/forecasts/<domain>/bridge/ (column bridge_source). The same
   bridge is used for every forecast product, so all start from the same
   model states.
2. FORECASTS, 15 days, per gauge:
     wn3        WeatherNext 3 ensemble-mean forcing (optional: skipped when the
                WeatherNext 3 run of the same init is not available)
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

   Gauges without a recent observation (most of Spain: no live SAIH feed)
   run without updating (column `updated` = False).
4. Speed: every input is pivoted once to (days x gauges) arrays; ~1 ms per
   model run, so Spain's 863 gauges x 52 runs take a few minutes.

Writes  data/forecasts/<domain>/gr4j/<init>/q_forecast.parquet
        long: date (forecast days only), gauge_id, product, q (m3/s)
        [q05..q95 for aifs_ens], updated (state updating applied)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from hydrocat.config import DATA, PROCESSED, ROOT, load_settings
from hydrocat.forcing import load_forcing
from hydrocat.gr4j import CatchmentModel
from hydrocat.obs import observations

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
    """Daily forcing for first..last from day 0 of each date's 00 UTC run (WeatherNext 3, else AIFS)."""
    fdir = DATA / "forecasts" / domain
    cache = fdir / "bridge"
    cache.mkdir(parents=True, exist_ok=True)
    wn3_ok = None
    out = []
    for d in pd.date_range(first, last, freq="D").date:
        f = cache / f"{d:%Y%m%d}.parquet"
        if not f.exists():
            run = fdir / "weathernext3" / f"{d:%Y%m%d}00" / "forcing.parquet"
            if not run.exists():
                if wn3_ok is None:
                    wn3_ok = _load("10_fetch_weathernext3").available()
                if wn3_ok:
                    subprocess.run([sys.executable, str(ROOT / "codes" / "10_fetch_weathernext3.py"),
                                    "--domain", domain, "--init", d.isoformat()], check=False)
            src = "weathernext3"
            if not run.exists():
                src = "aifs"
                run = fdir / "aifs" / f"{d:%Y%m%d}00" / "forcing.parquet"
                if not run.exists():
                    run = fdir / "bridge_aifs" / f"{d:%Y%m%d}00" / "forcing.parquet"
                    if not run.exists():
                        subprocess.run([sys.executable, str(ROOT / "codes" / "11_fetch_aifs.py"), "--domain", domain,
                                        "--date", d.isoformat(), "--days", "1", "--no-ens",
                                        "--out", str(fdir / "bridge_aifs")], check=True)
            x = pd.read_parquet(run)
            x[x.date == pd.Timestamp(d)].assign(bridge_source=src).to_parquet(f, index=False)
        out.append(pd.read_parquet(f))
    b = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(b):
        b["bridge_source"] = b.get("bridge_source", pd.Series(index=b.index, dtype=object)).fillna("weathernext3")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--init", default=None, help="YYYYMMDDHH of the forecast runs (default: latest AIFS)")
    args = ap.parse_args()
    D = S["domains"][args.domain]
    base = PROCESSED if args.domain == "catalonia" else PROCESSED / args.domain
    fdir = DATA / "forecasts" / args.domain
    init = args.init or sorted(p.name for p in (fdir / "aifs").iterdir())[-1]        # AIFS is required
    aifs = pd.read_parquet(fdir / "aifs" / init / "forcing.parquet")
    fc_dates = pd.DatetimeIndex(sorted(aifs.date.unique()))              # 15 days (00 UTC run) or 14 (12 UTC)
    d0 = fc_dates[0]                                                      # first forecast day
    log(f"GR4J forecast, runs of {init}, domain {args.domain}, {d0.date()} .. {fc_dates[-1].date()}")

    rawdir = base / "gr4j" / "run_era5l_raw"
    par = pd.read_csv((rawdir if rawdir.exists() else base / "gr4j") / "parameters.csv").set_index("gauge_id")
    gids = list(par.index)
    hy = pd.read_csv(base / "hypsometry.csv").set_index("gauge_id")
    cat = gpd.read_file(ROOT / D["catchments"]).set_index("gauge_id")
    q_obs = observations(args.domain)

    era = load_forcing(args.domain, "none", columns=["date", "gauge_id", *VARS])
    era = era[(era.date >= S["periods"]["warmup_start"]) & era.gauge_id.isin(gids)]
    last_era = era.date.max()
    br = bridge(args.domain, (last_era + pd.Timedelta(days=1)).date(), (d0 - pd.Timedelta(days=1)).date())
    log(f"   ERA5-Land to {last_era.date()}, bridge {len(br.date.unique()) if len(br) else 0} days"
        + (f" ({br.drop_duplicates('date').bridge_source.value_counts().to_dict()})" if len(br) else ""))
    past = pd.concat([era, br[br.gauge_id.isin(gids)][["date", "gauge_id", *VARS]]], ignore_index=True)
    del era

    # every input as a (days, gauges) array, built once
    def grid(df, dates):
        return {v: df.pivot(index="date", columns="gauge_id", values=v).reindex(index=dates, columns=gids)
                .to_numpy(float) for v in VARS}
    past_dates = pd.date_range(past.date.min(), d0 - pd.Timedelta(days=1), freq="D")
    H = grid(past, past_dates)
    if np.isnan(H["total_precipitation"]).any():
        raise RuntimeError("gaps in the ERA5-Land + bridge forcing before the forecast start")
    wf = fdir / "weathernext3" / init / "forcing.parquet"
    scen = {"aifs": grid(aifs, fc_dates)}
    if wf.exists():
        scen = {"wn3": grid(pd.read_parquet(wf), fc_dates)} | scen
    else:
        log(f"   no WeatherNext 3 run for {init}: AIFS products only")
    ens_f = fdir / "aifs" / init / "precip_members.parquet"
    ens = None
    if ens_f.exists():
        e = pd.read_parquet(ens_f)
        members = sorted(e.member.unique())
        ens = np.stack([e[e.member == m].pivot(index="date", columns="gauge_id", values="total_precipitation")
                        .reindex(index=fc_dates, columns=gids).to_numpy(float) for m in members])  # (m, days, g)
    dates = past_dates.append(fc_dates)
    n_past = len(past_dates)

    rows = []
    for j, gid in enumerate(gids):
        p = par.loc[gid]
        k = cat.area_km2[gid] / 86.4
        params = [p.X1, p.X2, p.X3, p.X4, p.CTG, p.Kf]
        o = q_obs[gid][(d0 - pd.Timedelta(days=3)):(d0 - pd.Timedelta(days=1))].dropna() if gid in q_obs else []
        ui, upd_mm = ((dates.get_loc(o.index[-1]), float(o.iloc[-1]) / k) if len(o) else (-1, 0.0))
        hyp = hy.loc[gid].values.astype(float)

        def run(fc: dict, precip=None):
            x = {v: np.concatenate([H[v][:, j], fc[v][:, j] if (v != "total_precipitation" or precip is None)
                                    else precip]) for v in VARS}
            m = CatchmentModel(x["total_precipitation"], x["temperature_2m"], x["temperature_2m_min"],
                               x["temperature_2m_max"], x["pet_fao56"], hyp, GR["n_elevation_layers"])
            q = m.run(params, ui, upd_mm)[n_past:] * k
            return np.where(q < p.q0_m3s, 0.0, q)

        for name, fc in scen.items():
            rows.append(pd.DataFrame({"date": fc_dates, "gauge_id": gid, "product": name, "q": run(fc),
                                      "updated": ui >= 0}))
        if ens is not None:
            qm = np.stack([run(scen["aifs"], ens[i, :, j]) for i in range(len(ens))])
            qs = np.quantile(qm, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
            rows.append(pd.DataFrame({"date": fc_dates, "gauge_id": gid, "product": "aifs_ens",
                                      **{c: qs[i] for i, c in enumerate(["q05", "q25", "q50", "q75", "q95"])},
                                      "updated": ui >= 0}))
        if (j + 1) % 200 == 0:
            log(f"   {j + 1}/{len(gids)} gauges")
    q = pd.concat(rows, ignore_index=True)
    out = fdir / "gr4j" / init
    out.mkdir(parents=True, exist_ok=True)
    q.to_parquet(out / "q_forecast.parquet", index=False)
    n_upd = q.drop_duplicates("gauge_id").updated.sum()
    log(f"-> {out.relative_to(ROOT)}/q_forecast.parquet ({q.gauge_id.nunique()} gauges, {n_upd} updated to an "
        f"observation, products {sorted(q['product'].unique())})")


if __name__ == "__main__":
    main()
