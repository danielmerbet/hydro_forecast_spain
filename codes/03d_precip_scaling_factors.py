#!/usr/bin/env python3
"""
03d — Monthly precipitation scaling factors: ERA5-Land -> EMO-1 climatology.

WHY
  Script 03c shows ERA5-Land precipitation is higher than the gauge-based EMO-1
  in every ACA catchment (median +29 %, up to +66 % in the Garrotxa /
  Pre-Pyrenees), and that with EMO-1 no catchment has an impossible water
  balance (P - Q > ET0), against 14 with ERA5-Land. EMO-1 itself is published
  once a year, so it cannot drive a daily forecast; ERA5-Land (and the weather
  forecasts) can.

METHOD — linear scaling (e.g. Teutschbein & Seibert, 2012, J. Hydrol. 456-457)
  For each catchment c and calendar month m, over the overlap 2003-2024:
      f(c, m) = mean(P_EMO1 in month m) / mean(P_ERA5L in month m)
  Corrected precipitation = f(c, month(t)) * P_ERA5L(t).
  Timing and day-to-day variability stay ERA5-Land's; the monthly climatology
  becomes EMO-1's. Factors are bounded to [0.3, 2.0] as a safeguard (none
  reaches the bounds in Catalonia; they are reported).

Used when settings.yaml has forcing.precip_correction: emo1_monthly_scaling
(scripts 05 and 06 via hydrocat.forcing.load_forcing).

Writes data/processed/precip_scaling_emo1.csv   (gauge_id, month, factor)
"""
from __future__ import annotations

import pandas as pd

from hydrocat.config import PROCESSED, P

OUT = PROCESSED / "precip_scaling_emo1.csv"


def main():
    era = pd.read_parquet(P.forcing, columns=["date", "gauge_id", "total_precipitation"])
    emo = pd.read_parquet(PROCESSED / "forcing_emo1_pr.parquet")
    d = era.merge(emo, on=["date", "gauge_id"], suffixes=("_era5l", "_emo1")).dropna()
    d["month"] = d.date.dt.month
    m = d.groupby(["gauge_id", "month"])[["total_precipitation_era5l", "total_precipitation_emo1"]].mean()
    f = (m.total_precipitation_emo1 / m.total_precipitation_era5l).clip(0.3, 2.0).rename("factor").reset_index()
    f.to_csv(OUT, index=False)
    print(f"{f.gauge_id.nunique()} catchments x 12 months, {d.date.min().date()} .. {d.date.max().date()}")
    print(f"factor median {f.factor.median():.2f}, range {f.factor.min():.2f}–{f.factor.max():.2f}; "
          f"at bounds: {int(((f.factor <= 0.3) | (f.factor >= 2.0)).sum())}")
    print(f.pivot(index="gauge_id", columns="month", values="factor").median().round(2).to_string())


if __name__ == "__main__":
    main()
