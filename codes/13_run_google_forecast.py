#!/usr/bin/env python3
"""
13 — Google's hydrology model, forecast mode (7 days), for today's runs.

INPUTS (Caravan-MultiMet format, as in script 06, but for one issue date)
  issue date I = forecast start D - 1 day; lead L valid on I + L (MultiMet
  convention: HRES/GRAPHCAST lead_time 1..10 days, union_mapping fills a
  missing forecast value with ERA5-Land on the valid date).
  hindcast (365 days up to I): ERA5-Land, plus the WeatherNext 3 "bridge" days
                               that GR4J also uses (ERA5-Land lags ~7 days).
                               For the 10 issue dates before I, leads valid after
                               I take today's forecast (the framework needs every
                               lead present over the hindcast year).
  HRES slots,      leads 1..10: ECMWF AIFS deterministic (script 11)
  GRAPHCAST slots, leads 1..10: WeatherNext 3 ensemble mean (script 10) —
                               GraphCast's successor at Google DeepMind
  IMERG, CPC: missing (the model is built to average over available products).
MODELS
  released   the pretrained weights as published (script 07)
  finetuned  the ACA fine-tuned weights (script 07b; Catalonia only)
  Each forecast runs in a copy of the run directory, so the historical test
  results are never overwritten.
OUTPUT
  time steps 0..7 of the CMAL distribution -> median and 5/25/75/95 % per
  valid day (step 0 = issue date, i.e. yesterday's nowcast).

Writes  data/forecasts/<domain>/google/<init>/q_forecast.parquet
        long: date, gauge_id, product, q05, q25, q50, q75, q95 (m3/s)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from hydrocat.config import DATA, PROCESSED, ROOT, P, load_settings
from hydrocat.forcing import load_forcing

S = load_settings()
LEADS = pd.to_timedelta(np.arange(1, 11), unit="D")
V = ["total_precipitation", "temperature_2m", "surface_pressure",
     "surface_net_solar_radiation", "surface_net_thermal_radiation"]
QU = [0.05, 0.25, 0.5, 0.75, 0.95]


def log(*a):
    print(*a, flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "codes" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write(ds, path):
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", consolidated=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=["catalonia"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--gpu", type=int, default=-1)
    args = ap.parse_args()
    fdir = DATA / "forecasts" / args.domain
    inits = sorted(set(p.name for p in (fdir / "weathernext3").iterdir()) & set(p.name for p in (fdir / "aifs").iterdir()))
    init = args.init or inits[-1]
    D0 = pd.Timestamp(dt.datetime.strptime(init, "%Y%m%d%H").date())
    I = D0 - pd.Timedelta(days=1)                                              # issue date
    work = fdir / "google" / init
    inp = work / "input"
    log(f"Google model forecast, runs of {init}, issue date {I.date()}")

    basins = (P.google_dir / "basins.txt").read_text().split()
    gids = [b.removeprefix("aca_") for b in basins]
    dates = pd.date_range(I - pd.Timedelta(days=420), I + pd.Timedelta(days=10), freq="D")   # noqa

    # --- past: ERA5-Land + bridge (same as GR4J)
    s12 = _load("12_run_gr4j_forecast")
    era = load_forcing(args.domain, "none")
    br = s12.bridge(args.domain, (era.date.max() + pd.Timedelta(days=1)).date(), I.date())
    past = pd.concat([era, br], ignore_index=True)
    past = past[past.gauge_id.isin(gids) & (past.date >= dates[0]) & (past.date <= I)]
    era_ds = xr.Dataset({f"era5land_{v}": (("basin", "date"), past.pivot(index="date", columns="gauge_id", values=v)
                                           .reindex(index=dates, columns=gids).values.T.astype("float32")) for v in V},
                        coords={"basin": basins, "date": dates})
    write(era_ds, inp / "dynamics" / "ERA5_LAND" / "timeseries.zarr")

    # --- forecasts on issue date I, leads 1..10
    cfg0 = yaml.safe_load((ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["baseline"]
                           / "config.yml").read_text())
    i_idx = dates.get_loc(I)

    def fc_product(src: pd.DataFrame, names: dict) -> xr.Dataset:
        out = {}
        for tgt, col in names.items():
            a = np.full((len(basins), len(dates), len(LEADS)), np.nan, "float32")
            w = src.pivot(index="date", columns="gauge_id", values=col).reindex(columns=gids)
            # Issue date I gets today's forecast at every lead. The previous 10
            # issue dates also need values at leads whose valid date lies after
            # I (the framework requires complete leads over the whole hindcast
            # year); they get today's forecast for that valid date. Valid dates
            # up to I stay NaN and are filled from ERA5-Land/bridge by union_mapping.
            for back in range(0, 11):
                d = I - pd.Timedelta(days=back)
                di = dates.get_loc(d)
                for j, L in enumerate(LEADS):
                    vd = d + L
                    if vd in w.index and (back == 0 or vd > I):
                        a[:, di, j] = w.loc[vd].values
            out[tgt] = (("basin", "date", "lead_time"), a)
        return xr.Dataset(out, coords={"basin": basins, "date": dates, "lead_time": LEADS})

    aifs = pd.read_parquet(fdir / "aifs" / init / "forcing.parquet")
    wn3 = pd.read_parquet(fdir / "weathernext3" / init / "forcing.parquet")
    hres = fc_product(aifs, {f"hres_{v}": v for v in V})
    write(hres[cfg0["forecast_inputs"]["hres"]], inp / "dynamics" / "HRES" / "timeseries.zarr")
    gc = fc_product(wn3, {"graphcast_temperature_2m": "temperature_2m",
                          "graphcast_total_precipitation": "total_precipitation"})
    write(gc, inp / "dynamics" / "GRAPHCAST" / "timeseries.zarr")
    nan2 = lambda n: xr.Dataset({n: (("basin", "date"), np.full((len(basins), len(dates)), np.nan, "float32"))},  # noqa: E731
                                coords={"basin": basins, "date": dates})
    write(nan2("imerg_precipitation"), inp / "dynamics" / "IMERG" / "timeseries.zarr")
    write(nan2("cpc_precipitation"), inp / "dynamics" / "CPC" / "timeseries.zarr")

    # --- run both model variants
    area = pd.read_csv(P.attributes).set_index("gauge_id").loc[gids, "area_km2"].values
    # released weights: straight from Google's repository (cloned by 00_setup.sh);
    # fine-tuned weights: the committed copy in models/ (exported after script 07b)
    runs = {"google_released": ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["baseline"]}
    ft = ROOT / "models" / "google_finetuned"
    if not (ft / "config.yml").exists():
        found = sorted((P.google_dir / "runs" / "finetuned").glob("finetuned_*/config.yml"))
        ft = found[-1].parent if found else None
    if ft is not None:
        runs["google_finetuned"] = ft
    frames = []
    fmt = lambda d: d.strftime("%d/%m/%Y")  # noqa: E731
    for name, src in runs.items():
        rd = work / name
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        for f in src.glob("model_epoch*.pt"):
            shutil.copy(f, rd / f.name)
        for f in ("scaler.nc",):
            if (src / f).exists():
                shutil.copy(src / f, rd / f)
        if (src / "train_data").exists():
            shutil.copytree(src / "train_data", rd / "train_data")
        cfg = yaml.safe_load((src / "config.yml").read_text())
        if cfg.get("base_run_dir"):   # fine-tuned run: scaler lives in the base run -> this machine's path
            cfg["base_run_dir"] = str(ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["filtered"])
        bf = str(P.google_dir / "basins.txt")
        cfg.update(run_dir=str(rd), train_dir=str(rd / "train_data"), dynamics_data_dir=str(inp / "dynamics"),
                   train_basin_file=bf, validation_basin_file=bf, test_basin_file=bf,
                   statics_data_dir=str(P.google_dir / "statics"), targets_data_dir=str(P.google_dir / "statics"),
                   # the framework needs start < end: issue I-1 and I, keep only I below
                   test_start_date=fmt(I - pd.Timedelta(days=1)), test_end_date=fmt(I), n_samples=500,
                   device="cpu" if args.gpu < 0 else f"cuda:{args.gpu}", img_log_dir=str(rd / "img_log"),
                   tester_skip_obs_all_nan=False)          # the future has no observations
        (rd / "config.yml").write_text(yaml.safe_dump(cfg))
        subprocess.run(["run", "infer", "--run-dir", str(rd), "--gpu", str(args.gpu)], check=True,
                       stdout=subprocess.DEVNULL)
        res = sorted(rd.glob("test/model_epoch*/test_results.zarr"))[-1]
        sim = xr.open_zarr(res, decode_timedelta=True).streamflow_sim.isel(freq=0).sel(date=I)
        qs = sim.quantile(QU, dim="samples").compute()
        qs = qs.transpose("basin", "time_step", "quantile").values                     # (basin, step, q)
        k = (area / 86.4)[:, None, None]
        qs = qs * k
        for b, g in enumerate(gids):
            for L in range(qs.shape[1]):
                frames.append(dict(date=I + pd.Timedelta(days=L), gauge_id=g, product=name,
                                   **{f"q{int(q * 100):02d}": float(qs[b, L, j]) for j, q in enumerate(QU)}))
        log(f"   {name}: done")
    out = pd.DataFrame(frames)
    out.to_parquet(work / "q_forecast.parquet", index=False)
    log(f"-> {(work / 'q_forecast.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
