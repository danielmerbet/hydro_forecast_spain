#!/usr/bin/env python3
"""
03c — Is ERA5-Land's rain too high? ERA5-Land vs EMO-1 (gauge-based), per catchment.

For the GR4J calibration period (2008-10-01 .. 2023-09-30), on days with an
observed discharge, per catchment:
  P_era5l, P_emo1   mean annual precipitation (mm/yr)
  Q_obs             mean annual observed runoff (mm/yr)
  ET0               mean annual FAO-56 reference ET from ERA5-Land (mm/yr)
  "impossible balance": P - Q > ET0, i.e. more water must leave than even a
  well-watered surface could evaporate -> either P is too high or water is
  abstracted / lost to groundwater before the gauge.
Also the daily correlation between the two rain products.

Writes
  outputs/tables/03c_precip_era5land_vs_emo1.csv
  outputs/figures/03c_precip_era5land_vs_emo1.png
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrocat.config import FIGURES, PROCESSED, TABLES, P, load_settings

S = load_settings()
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"
mpl.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def main():
    c0, c1 = S["periods"]["calib_start"], S["periods"]["calib_end"]
    era = pd.read_parquet(P.forcing)
    era = era[(era.date >= c0) & (era.date <= c1)]
    emo = pd.read_parquet(PROCESSED / "forcing_emo1_pr.parquet")
    emo = emo[(emo.date >= c0) & (emo.date <= c1)]
    q = pd.read_parquet(P.q_daily)[c0:c1]
    cat = gpd.read_file(P.catchments).set_index("gauge_id")
    st = pd.read_csv(P.stations).set_index("gauge_id")
    reg = pd.read_csv(P.attributes).set_index("gauge_id")["regulated"]
    gauges = [g for g in st.index[st.modelled] if g in cat.index]

    rows = []
    for g in gauges:
        e = era[era.gauge_id == g].set_index("date")
        m = emo[emo.gauge_id == g].set_index("date")["total_precipitation"]
        qq = q[g] * 86.4 / cat.area_km2[g]
        ok = qq.notna() & qq.index.isin(e.index) & qq.index.isin(m.index)
        d = ok[ok].index
        yr = 365.25
        pe, pm = e.total_precipitation[d].mean() * yr, m[d].mean() * yr
        et0, qo = e.pet_fao56[d].mean() * yr, qq[d].mean() * yr
        rows.append(dict(gauge_id=g, regulated=bool(reg.get(g, False)), P_era5land=pe, P_emo1=pm,
                         ratio_era5l_emo1=pe / pm, Q_obs=qo, ET0=et0,
                         impossible_era5l=pe - qo > et0, impossible_emo1=pm - qo > et0,
                         daily_corr=float(np.corrcoef(e.total_precipitation[d], m[d])[0, 1]),
                         runoff_ratio_era5l=qo / pe, runoff_ratio_emo1=qo / pm))
    t = pd.DataFrame(rows).round(3)
    t.to_csv(TABLES / "03c_precip_era5land_vs_emo1.csv", index=False)
    print(f"{len(t)} catchments, 2008-10 .. 2023-09")
    print(f"  median P ERA5-Land {t.P_era5land.median():.0f} mm/yr, EMO-1 {t.P_emo1.median():.0f} mm/yr, "
          f"median ratio {t.ratio_era5l_emo1.median():.2f} (range {t.ratio_era5l_emo1.min():.2f}–"
          f"{t.ratio_era5l_emo1.max():.2f})")
    print(f"  median daily correlation {t.daily_corr.median():.2f}")
    print(f"  'impossible' water balance (P - Q > ET0): ERA5-Land {int(t.impossible_era5l.sum())}, "
          f"EMO-1 {int(t.impossible_emo1.sum())} of {len(t)}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axs[0]
    lim = [300, max(t.P_era5land.max(), t.P_emo1.max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=0.8)
    for flag, mk, lab in [(False, "o", "near-natural"), (True, "s", "regulated")]:
        s = t[t.regulated == flag]
        ax.scatter(s.P_emo1, s.P_era5land, s=30, marker=mk, color=C1, facecolor="none" if flag else C1, label=lab)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("EMO-1 (gauge-based) precipitation, mm/yr")
    ax.set_ylabel("ERA5-Land precipitation, mm/yr")
    ax.set_title(f"Catchment mean annual rain 2008–2023 (median ratio {t.ratio_era5l_emo1.median():.2f})",
                 loc="left", fontsize=9, color=INK)
    ax.legend(loc="upper left")

    ax = axs[1]
    x = np.arange(len(t))
    tt = t.sort_values("P_era5land")
    ax.bar(x - 0.2, tt.P_era5land - tt.Q_obs, 0.4, color=C1, label="P − Q with ERA5-Land")
    ax.bar(x + 0.2, tt.P_emo1 - tt.Q_obs, 0.4, color=C2, label="P − Q with EMO-1")
    ax.plot(x, tt.ET0, color=INK, lw=1.2, marker="_", ls="none", ms=8, label="FAO-56 ET0 (upper limit of evaporation)")
    ax.set_xticks(x, tt.gauge_id, rotation=90, fontsize=5)
    ax.set_ylabel("mm/yr")
    ax.set_title("Water that must leave the catchment other than as river flow", loc="left", fontsize=9, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "03c_precip_era5land_vs_emo1.png", dpi=170)


if __name__ == "__main__":
    main()
