#!/usr/bin/env python3
"""
07b — Fine-tune Google's pretrained model on ACA gauges ("Google fine-tuned" arm).

WHY
  Used as released (script 07), Google's global model reproduces the timing
  and shape of Catalan floods but carries too much water: Catalan catchments
  lose a large share of ERA5-Land precipitation (abstractions, karst,
  ERA5-Land's rain bias in the Pre-Pyrenees), which a model trained on
  ~16,000 basins worldwide cannot know. Google's README names fine-tuning on
  local data as the intended use of the released weights, and it is the fair
  counterpart of calibrating GR4J per gauge.

HOW (the framework's own `run finetune`)
  * start from the "filtered" pretrained run (trained on basins with NSE > 0.5,
    which Google recommends as the starting point for fine-tuning);
  * the pretrained scaler is kept (inputs normalised exactly as in training);
  * all model parts are trainable, with a low learning rate (1e-4), so the
    network adapts without forgetting what it learned globally;
  * train 2008-10-01 .. 2021-09-30 on all modelled ACA gauges,
    validate 2021-10-01 .. 2023-09-30 after every epoch; the epoch with the
    best median validation NSE is kept. The test period (2023-10-01 → ) is
    never seen — same rule as for GR4J.
  * GPU (cuda:0) if available: needs the `hydrocat-gpu` environment
    (environment.yml + CUDA build of PyTorch, see README). A 4 GB laptop GPU
    fits batch size 128.

Then inference on the test period exactly as in script 07.

Writes
  data/processed/google/runs/finetuned/                 run directory (weights per epoch, logs)
  data/processed/google/q_sim_daily_finetuned.parquet   lead-0 median, m3/s
  data/processed/google/q_sim_quantiles_finetuned.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hydrocat.config import ROOT, P, load_settings

S = load_settings()
G = S["google"]
OUT = P.google_dir
FT = S.get("google_finetune", {})

# reuse condense() from script 07 so both arms are post-processed identically
_spec = importlib.util.spec_from_file_location("s07", Path(__file__).with_name("07_run_google_model.py"))
s07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s07)


def log(*a):
    print(*a, flush=True)


def fmt(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device id, -1 = CPU (very slow)")
    ap.add_argument("--epochs", type=int, default=FT.get("epochs", 15))
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    base = ROOT / G["repo_dir"] / G["pretrained_runs"][FT.get("base", "filtered")]
    runs = OUT / "runs"
    basins = str(OUT / "basins.txt")
    cfg = {
        "base_run_dir": str(base),
        "finetune_modules": ["static_embedding_fc", "hindcast_embeddings_fc", "forecast_embeddings_fc",
                             "shared_embeddings_fc", "hindcast_lstm", "forecast_lstm", "head"],
        "experiment_name": "finetuned",
        "run_dir": str(runs / "finetuned"),
        "train_dir": str(runs / "finetuned" / "train_data"),
        "img_log_dir": str(runs / "finetuned" / "img_log"),
        "dynamics_data_dir": str(OUT / "dynamics"),
        "statics_data_dir": str(OUT / "statics"),
        "targets_data_dir": str(OUT / "statics"),
        "train_basin_file": basins, "validation_basin_file": basins, "test_basin_file": basins,
        "train_start_date": fmt(FT.get("train_start", "2008-10-01")),
        "train_end_date": fmt(FT.get("train_end", "2021-09-30")),
        "validation_start_date": fmt(FT.get("val_start", "2021-10-01")),
        "validation_end_date": fmt(FT.get("val_end", "2023-09-30")),
        "epochs": args.epochs,
        "initial_learning_rate": FT.get("learning_rate", 1e-4),
        "learning_rate_strategy": "ConstantLR",
        "batch_size": FT.get("batch_size", 128),
        "max_updates_per_epoch": FT.get("max_updates_per_epoch", 500),
        "validate_every": 1, "validate_n_random_basins": 10_000,
        "metrics": ["NSE", "KGE"],
        "n_samples": 100, "num_workers": 0, "save_validation_results": False,
        "device": "cpu" if args.gpu < 0 else f"cuda:{args.gpu}",
    }
    cfg_file = OUT / "finetune_config.yml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    run = runs / "finetuned"
    if not args.skip_training:
        if run.exists():
            shutil.rmtree(run)
        log(f"fine-tuning from {base.name}: train {cfg['train_start_date']}–{cfg['train_end_date']}, "
            f"validation {cfg['validation_start_date']}–{cfg['validation_end_date']}, {args.epochs} epochs")
        subprocess.run(["run", "finetune", "--config-file", str(cfg_file), "--gpu", str(args.gpu)], check=True)

    # The framework creates <run_dir>/<experiment_name>_<ddmm_HHMMSS>/; use the newest one.
    made = sorted([p for p in run.glob("finetuned_*") if (p / "config.yml").exists()],
                  key=lambda p: p.stat().st_mtime)
    if not made:
        raise RuntimeError(f"no fine-tuning run found under {run}")
    run = made[-1]
    log(f"run directory: {run.relative_to(ROOT)}")

    # --- pick the epoch with the best median validation NSE
    scores = {}
    for f in sorted(run.glob("validation/model_epoch*/validation_metrics.csv")):
        m = pd.read_csv(f)                      # columns: basin, NSE, KGE
        col = "NSE"
        scores[int(f.parent.name.replace("model_epoch", ""))] = float(m[col].median())
    if scores:
        best = max(scores, key=scores.get)
        log("median validation NSE by epoch: " + ", ".join(f"{e}:{v:.2f}" for e, v in sorted(scores.items())))
        log(f"best epoch {best}")
        for f in run.glob("model_epoch*.pt"):     # keep only the chosen weights for inference
            if int(f.stem.replace("model_epoch", "")) != best:
                f.rename(f.with_suffix(".pt.unused"))
    else:
        log("WARNING: no validation metrics found; using the last epoch")

    # --- inference on the test period (same as script 07)
    last = pd.read_parquet(P.forcing, columns=["date"]).date.max()
    c = yaml.safe_load((run / "config.yml").read_text())
    c.update(test_start_date=fmt(S["periods"]["test_start"]),
             test_end_date=fmt((last - pd.Timedelta(days=8)).date()), n_samples=200,
             device="cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    (run / "config.yml").write_text(yaml.safe_dump(c))
    subprocess.run(["run", "infer", "--run-dir", str(run), "--gpu", str(args.gpu)], check=True)
    s07.condense(run, "finetuned", G["simulation_lead"])


if __name__ == "__main__":
    main()
