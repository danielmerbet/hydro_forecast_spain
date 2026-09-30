#!/usr/bin/env python3
"""
07 — Run Google's pretrained hydrology model on the Catalan catchments.

For each released pretrained run (settings.yaml: google.pretrained_runs):
  1. make a run directory with the released weights + scaler, untouched;
  2. rewrite only the data paths, basin list, test period, device and the
     number of output samples in its config.yml;
  3. call the framework's own CLI:  run infer --run-dir <dir>
  4. condense the probabilistic output to daily median + 5/25/75/95 %
     quantiles at every lead time, in m3/s.

THE OUTPUT OF THE MODEL
  For each issue date d and time step L = 0..7 the model returns samples of a
  Countable Mixture of Asymmetric Laplacians (CMAL) — a full predictive
  distribution of streamflow on day d+L. With ERA5-Land at every lead (script
  06), L = 0 is the model's best estimate of "today's" flow given all weather
  up to today: the analogue of a continuous GR4J simulation, and what is
  compared with GR4J in script 08. (Other leads are kept for later.)

TEST PERIOD
  2023-10-01 .. last ERA5-Land day minus the 7-day forecast horizon.
  NOT earlier: the weights were trained on 1982-01-01 .. 2023-09-30 and
  Google's README warns explicitly that scoring inside that window is data
  leakage.

Writes
  data/processed/google/runs/<run>/...                       run directory (config, raw results)
  data/processed/google/q_sim_daily_<run>.parquet            lead-0 median, m3/s (date x gauge)
  data/processed/google/q_sim_quantiles_<run>.parquet        long: date, gauge_id, lead, q05..q95 (m3/s)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from hydrocat.config import ROOT, P, load_settings

S = load_settings()
G = S["google"]
OUT = P.google_dir
QUANTS = [0.05, 0.25, 0.5, 0.75, 0.95]


def log(*a):
    print(*a, flush=True)


def prepare_run(name: str, src_rel: str, t0: str, t1: str, n_samples: int, gpu: int,
                gauges: list[str] | None = None, tag: str = ""):
    src = ROOT / G["repo_dir"] / src_rel
    run = OUT / "runs" / (name + tag)
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    for f in src.glob("model_epoch*.pt"):
        shutil.copy(f, run / f.name)
    shutil.copy(src / "scaler.nc", run / "scaler.nc")
    cfg = yaml.safe_load((src / "config.yml").read_text())
    basins = str(OUT / "basins.txt")
    if gauges:                                   # preview on a subset of gauges
        (run / "basins.txt").write_text("".join(f"aca_{g}\n" for g in gauges))
        basins = str(run / "basins.txt")
    fmt = lambda d: pd.Timestamp(d).strftime("%d/%m/%Y")  # noqa: E731
    cfg.update(
        run_dir=str(run), train_dir=str(run / "train_data"), img_log_dir=str(run / "img_log"),
        dynamics_data_dir=str(OUT / "dynamics"),
        statics_data_dir=str(OUT / "statics"), targets_data_dir=str(OUT / "statics"),
        train_basin_file=basins, validation_basin_file=basins, test_basin_file=basins,
        test_start_date=fmt(t0), test_end_date=fmt(t1),
        device="cpu" if gpu < 0 else f"cuda:{gpu}",
        n_samples=n_samples, num_workers=0,
    )
    (run / "config.yml").write_text(yaml.safe_dump(cfg))
    return run


def condense(run, name: str, lead_sim: int, tag: str = ""):
    res = sorted(run.glob("test/model_epoch*/test_results.zarr"))[-1]
    ds = xr.open_zarr(res, decode_timedelta=True)
    sim = ds.streamflow_sim.isel(freq=0)                     # basin, date, time_step, samples (mm/day)
    qs = sim.quantile(QUANTS, dim="samples").compute()
    att = pd.read_csv(P.attributes).set_index("gauge_id")
    gids = [b.removeprefix("aca_") for b in qs.basin.values]
    k = (att.loc[gids, "area_km2"].values / 86.4)[:, None, None, None]   # mm/day -> m3/s
    arr = qs.transpose("basin", "date", "time_step", "quantile").values * k
    rows = []
    dates = pd.DatetimeIndex(qs.date.values)
    for i, g in enumerate(gids):
        for L in qs.time_step.values:
            df = pd.DataFrame(arr[i, :, int(L), :], columns=[f"q{int(q * 100):02d}" for q in QUANTS])
            df.insert(0, "lead", int(L))
            df.insert(0, "gauge_id", g)
            df.insert(0, "date", dates + pd.Timedelta(days=int(L)))     # valid date
            rows.append(df)
    long = pd.concat(rows, ignore_index=True)
    name = name + tag
    long.to_parquet(OUT / f"q_sim_quantiles_{name}.parquet", index=False)
    wide = long[long.lead == lead_sim].pivot(index="date", columns="gauge_id", values="q50")
    wide.index.name = "date"
    wide.to_parquet(str(P.google_sim).format(run=name))
    log(f"   -> {str(P.google_sim).format(run=name).replace(str(ROOT) + '/', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(G["pretrained_runs"]))
    ap.add_argument("--gpu", type=int, default=-1, help="CUDA device id, -1 = CPU")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="CMAL samples per prediction (7500 in the release; 200 keeps output small)")
    ap.add_argument("--condense-only", action="store_true")
    ap.add_argument("--gauges", nargs="*", help="preview: only these gauges (outputs get a _preview suffix)")
    args = ap.parse_args()

    last = pd.read_parquet(P.forcing, columns=["date"]).date.max()
    t0, t1 = S["periods"]["test_start"], (last - pd.Timedelta(days=8)).date().isoformat()
    for name in args.runs:
        log(f"== {name}: test {t0} .. {t1}")
        tag = "_preview" if args.gauges else ""
        run = OUT / "runs" / (name + tag)
        if not args.condense_only:
            run = prepare_run(name, G["pretrained_runs"][name], t0, t1, args.n_samples, args.gpu,
                              args.gauges, tag)
            subprocess.run(["run", "infer", "--run-dir", str(run), "--gpu", str(args.gpu)], check=True)
        condense(run, name, G["simulation_lead"], tag)


if __name__ == "__main__":
    main()
