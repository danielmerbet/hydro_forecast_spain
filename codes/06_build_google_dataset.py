#!/usr/bin/env python3
"""
06 — Build the input dataset for Google's hydrology model (Caravan-MultiMet format).

THE MODEL
  Google's "mean embedding forecast LSTM" (the Flood Hub model), released with
  pretrained weights in github.com/google-research/flood-forecasting. It was
  trained on ~16,000 basins worldwide (Caravan + MultiMet), 1982-2023-09-30.
  We use the weights as released (no fine-tuning): a *global* model applied
  to Catalan catchments it was not calibrated on — as Google's Flood Hub does.

WHAT IT EXPECTS (config.yml of the pretrained run)
  hindcast inputs  HRES (5 vars), GRAPHCAST (2), IMERG (1), CPC (1), 365 days
  forecast inputs  HRES (5 vars), GRAPHCAST (2), lead times 1..7 days
  statics          10 Caravan climate indices + 74 HydroATLAS attributes
  union_mapping    any missing HRES/GRAPHCAST/IMERG/CPC value is replaced by
                   the ERA5-Land value of the same variable (for forecast
                   products: ERA5-Land on the forecast's VALID date).
  Units            precipitation mm/day, temperature degC, pressure kPa,
                   radiation W/m2 (daily mean), streamflow mm/day.

HOW WE USE IT HERE (simulation / "hindcast" mode)
  HRES, GRAPHCAST, IMERG and CPC are written as all-NaN, so through
  union_mapping the model sees ERA5-Land everywhere — at every lead time,
  i.e. "perfect" weather. That is the same forcing GR4J gets, which is what
  makes the comparison fair. In the operational forecast (later scripts) the
  HRES slots receive ECMWF (IFS/AIFS) forecasts and the GRAPHCAST slots
  receive WeatherNext 3 (GraphCast's successor at Google DeepMind), while the
  past stays ERA5-Land.

DOMAINS
  catalonia  ACA gauges with a modelled catchment, basin ids "aca_<gauge_id>"
  spain      the CEDEX gauges with calibrated GR4J parameters (the forecast
             ones), basin ids "es_<gauge_id>"; statics from
             04_catchment_attributes.py --domain spain --google

Writes (data/processed/google/ or data/processed/spain/google/)
  dynamics/{ERA5_LAND,HRES,GRAPHCAST,IMERG,CPC}/timeseries.zarr
  statics/attributes.zarr         basin x 84 attributes
  statics/streamflow.zarr         basin x date, observed streamflow (mm/day)
  basins.txt                      basin ids
"""
from __future__ import annotations

import shutil
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from hydrocat.config import PROCESSED, ROOT, P, load_settings

S = load_settings()
warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
LEADS = pd.to_timedelta(np.arange(1, 11), unit="D")   # same as Google's MultiMet HRES/GRAPHCAST
VARS = ["total_precipitation", "temperature_2m", "surface_pressure",
        "surface_net_solar_radiation", "surface_net_thermal_radiation"]


def log(*a):
    print(*a, flush=True)


PREFIX = {"catalonia": "aca_", "spain": "es_"}


def google_dir(domain: str):
    return P.google_dir if domain == "catalonia" else PROCESSED / domain / "google"


def basin_id(gid: str, domain: str = "catalonia") -> str:
    return f"{PREFIX[domain]}{gid}"


def write_zarr(ds: xr.Dataset, path):
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", consolidated=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(PREFIX))
    args = ap.parse_args()
    OUT = google_dir(args.domain)
    cfg = yaml.safe_load((ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["baseline"]
                          / "config.yml").read_text())
    if args.domain == "catalonia":
        att = pd.read_csv(P.attributes)
        st = pd.read_csv(P.stations).set_index("gauge_id")
        att = att[att.gauge_id.map(st["modelled"])]
        forcing_path, q_path = P.forcing, P.q_daily
    else:
        base = PROCESSED / args.domain
        att = pd.read_csv(base / "catchment_attributes.csv")
        att = att[att.gauge_id.isin(pd.read_csv(base / "gr4j" / "parameters.csv").gauge_id)]
        forcing_path, q_path = ROOT / S["domains"][args.domain]["forcing"], base / "q_obs_daily.parquet"
    gids = list(att.gauge_id)
    basins = [basin_id(g, args.domain) for g in gids]
    log(f"{len(basins)} basins")

    forcing = pd.read_parquet(forcing_path, columns=["date", "gauge_id", *VARS])
    forcing = forcing[forcing.gauge_id.isin(gids)]
    dates = pd.date_range(forcing.date.min(), forcing.date.max(), freq="D")

    # --- ERA5_LAND (basin, date)
    era = {}
    for v in VARS:
        w = forcing.pivot(index="date", columns="gauge_id", values=v).reindex(index=dates, columns=gids)
        era[f"era5land_{v}"] = (("basin", "date"), w.values.T.astype("float32"))
    era = xr.Dataset(era, coords={"basin": basins, "date": dates})
    write_zarr(era, OUT / "dynamics" / "ERA5_LAND" / "timeseries.zarr")

    # --- forecast products: all NaN in hindcast mode (filled from ERA5-Land by union_mapping)
    def nan_fc(names):
        a = np.full((len(basins), len(dates), len(LEADS)), np.nan, "float32")
        return xr.Dataset({n: (("basin", "date", "lead_time"), a) for n in names},
                          coords={"basin": basins, "date": dates, "lead_time": LEADS})

    def nan_obs(names):
        a = np.full((len(basins), len(dates)), np.nan, "float32")
        return xr.Dataset({n: (("basin", "date"), a) for n in names}, coords={"basin": basins, "date": dates})

    write_zarr(nan_fc(cfg["forecast_inputs"]["hres"]), OUT / "dynamics" / "HRES" / "timeseries.zarr")
    write_zarr(nan_fc(cfg["forecast_inputs"]["graphcast"]), OUT / "dynamics" / "GRAPHCAST" / "timeseries.zarr")
    write_zarr(nan_obs(cfg["hindcast_inputs"]["imerg"]), OUT / "dynamics" / "IMERG" / "timeseries.zarr")
    write_zarr(nan_obs(cfg["hindcast_inputs"]["cpc"]), OUT / "dynamics" / "CPC" / "timeseries.zarr")

    # --- statics
    names = cfg["static_attributes"]
    a = att.set_index("gauge_id").loc[gids, names].astype("float32")
    stat = xr.Dataset({n: ("basin", a[n].values) for n in names}, coords={"basin": basins})
    write_zarr(stat, OUT / "statics" / "attributes.zarr")

    # --- targets: m3/s -> mm/day (Caravan unit)
    q = pd.read_parquet(q_path).reindex(index=dates, columns=gids)
    area = att.set_index("gauge_id").loc[gids, "area_km2"].values
    mm = q.values * 86.4 / area[None, :]
    tgt = xr.Dataset({"streamflow": (("basin", "date"), mm.T.astype("float32"))},
                     coords={"basin": basins, "date": dates})
    write_zarr(tgt, OUT / "statics" / "streamflow.zarr")

    (OUT / "basins.txt").write_text("\n".join(basins) + "\n")
    log(f"-> {OUT.relative_to(ROOT)}  ({dates[0].date()} .. {dates[-1].date()})")

    # sanity check: our inputs vs the global training distribution (scaler)
    sc = xr.open_dataset(ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["baseline"] / "scaler.nc")
    log("input check (ours: mean over basins/days | training set mean, std):")
    for v in VARS:
        n = f"era5land_{v}"
        log(f"  {n:42s} {float(era[n].mean()):8.2f} | {float(sc[n].sel(parameter='mean')):8.2f} "
            f"{float(sc[n].sel(parameter='std')):8.2f}")
    for n in ["p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND", "ele_mt_sav"]:
        log(f"  {n:42s} {float(stat[n].mean()):8.2f} | {float(sc[n].sel(parameter='mean')):8.2f} "
            f"{float(sc[n].sel(parameter='std')):8.2f}")


if __name__ == "__main__":
    main()
