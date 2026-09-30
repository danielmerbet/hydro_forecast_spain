#!/usr/bin/env python3
"""
05b — Figures of the GR4J calibration (reads the outputs of 05; no model runs).

Writes
  outputs/figures/05b_gr4j_kge_by_gauge.png   KGE in calibration vs test period, every gauge
  outputs/figures/05b_gr4j_kge_map.png        calibration / test KGE on the catchment map
  outputs/figures/05b_gr4j_parameters.png     distribution of calibrated parameters
  outputs/figures/gr4j/<gauge>.png            per-gauge diagnostics:
                                              hydrograph 2008 → today (calib + test shaded),
                                              mean monthly regime, flow-duration curve,
                                              daily scatter (log-log)
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrocat.config import FIGURES, P, load_settings

S = load_settings()
PER = S["periods"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_SIM, C_TEST = "#2a78d6", "#eb6834"          # categorical slots 1-2
mpl.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def fdc(x):
    x = np.sort(x[np.isfinite(x)])[::-1]
    return np.arange(1, len(x) + 1) / (len(x) + 1) * 100, x


def gauge_panel(gid, meta, obs, sim, par, out):
    c0, c1, t0 = pd.Timestamp(PER["calib_start"]), pd.Timestamp(PER["calib_end"]), pd.Timestamp(PER["test_start"])
    obs, sim = obs[c0:], sim[c0:]
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(t0, sim.index.max(), color=C_TEST, alpha=0.07, lw=0)
    ax.plot(sim.index, sim.values, color=C_SIM, lw=0.8, label="GR4J")
    ax.plot(obs.index, obs.values, color=INK, lw=0.8, label="Observed (ACA)")
    ax.set_yscale("symlog", linthresh=max(np.nanpercentile(obs, 10), 0.01))
    ax.set_ylabel("Discharge (m³/s)")
    ax.legend(loc="upper left", ncol=2)
    ax.text(c0 + (c1 - c0) / 2, 1.0, f"calibration   KGE {par.cal_KGE:.2f}   NSE {par.cal_NSE:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2)
    ax.text(t0 + (sim.index.max() - t0) / 2, 1.0, f"test   KGE {par.test_KGE:.2f}   NSE {par.test_NSE:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=C_TEST)
    reg = "regulated (" + meta.reservoirs_upstream + ")" if meta.regulated else "near-natural"
    snow = ", snow calibrated" if par.snow_calibrated else ""
    rain = ("rain: ERA5-Land × EMO-1 monthly scaling" if S["forcing"].get("precip_correction", "none") != "none"
            else "rain: ERA5-Land")
    fig.suptitle(f"{gid} — {meta['name']} ({meta.river}) · {meta.area_km2:.0f} km² · {reg}{snow} · {rain}",
                 x=0.01, ha="left", fontsize=11, color=INK)

    both = pd.concat([obs.rename("o"), sim.rename("s")], axis=1).dropna()
    ax = fig.add_subplot(gs[1, 0])
    reg_o = both.o.groupby(both.index.month).mean()
    reg_s = both.s.groupby(both.index.month).mean()
    ax.plot(reg_o.index, reg_o.values, color=INK, lw=1.6, marker="o", ms=4, label="Observed")
    ax.plot(reg_s.index, reg_s.values, color=C_SIM, lw=1.6, marker="o", ms=4, label="GR4J")
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylabel("Mean discharge (m³/s)")
    ax.set_title("Monthly regime", loc="left", fontsize=9, color=INK)

    ax = fig.add_subplot(gs[1, 1])
    p, v = fdc(both.o.values)
    ax.plot(p, v, color=INK, lw=1.4)
    p, v = fdc(both.s.values)
    ax.plot(p, v, color=C_SIM, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Exceedance (%)")
    ax.set_title("Flow-duration curve", loc="left", fontsize=9, color=INK)

    ax = fig.add_subplot(gs[1, 2])
    cal = both[:c1]
    tst = both[t0:]
    lo = max(np.nanpercentile(both.values, 0.5), 1e-3)
    hi = np.nanmax(both.values) * 1.2
    ax.scatter(cal.o.clip(lo), cal.s.clip(lo), s=4, color=C_SIM, alpha=0.25, lw=0, label="calibration")
    ax.scatter(tst.o.clip(lo), tst.s.clip(lo), s=5, color=C_TEST, alpha=0.5, lw=0, label="test")
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed (m³/s)"); ax.set_ylabel("GR4J (m³/s)")
    ax.legend(loc="upper left", markerscale=3)
    ax.set_title(f"X1 {par.X1:.0f}  X2 {par.X2:.2f}  X3 {par.X3:.0f}  X4 {par.X4:.2f}",
                 loc="left", fontsize=8, color=INK2)
    fig.savefig(out)
    plt.close(fig)


def main():
    par = pd.read_csv(P.gr4j_params).set_index("gauge_id")
    sim = pd.read_parquet(P.gr4j_sim)
    obs = pd.read_parquet(P.q_daily)
    att = pd.read_csv(P.attributes).set_index("gauge_id")
    st = pd.read_csv(P.stations).set_index("gauge_id")
    att["name"] = st["name"]
    att["reservoirs_upstream"] = att["reservoirs_upstream"].fillna("")

    # --- 1. KGE by gauge
    d = par.join(att[["regulated", "area_ok", "name"]]).sort_values("cal_KGE")
    fig, ax = plt.subplots(figsize=(7, 0.19 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.hlines(y, d.cal_KGE.clip(-1), d.test_KGE.clip(-1), color=GRID, lw=1.2)
    ax.scatter(d.cal_KGE.clip(-1), y, s=34, color=C_SIM, label="calibration 2008–2023", zorder=3)
    ax.scatter(d.test_KGE.clip(-1), y, s=34, facecolor="white", edgecolor=C_TEST, lw=1.6,
               label=f"test {PER['test_start'][:7]} → today", zorder=3)
    labels = [f"{g} {'■' if r else ' '} {n[:26]}" for g, r, n in zip(d.index, d.regulated, d.name)]
    ax.set_yticks(y, labels, fontsize=7, family="monospace")
    ax.set_xlim(-1.02, 1)
    ax.axvline(0, color=INK2, lw=0.7, ls=":")
    ax.set_xlabel("KGE  (values < −1 drawn at −1)")
    ax.legend(loc="lower right")
    ax.set_title(f"GR4J + CemaNeige at {len(d)} ACA gauges — median KGE calib {d.cal_KGE.median():.2f}, "
                 f"test {d.test_KGE.median():.2f}\n■ = regulated (reservoir upstream)",
                 loc="left", fontsize=9, color=INK)
    fig.savefig(FIGURES / "05b_gr4j_kge_by_gauge.png")
    plt.close(fig)

    # --- 2. maps
    cat = gpd.read_file(P.catchments).set_index("gauge_id")
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, title in [(axs[0], "cal_KGE", "Calibration (2008–2023)"), (axs[1], "test_KGE", "Test (2023 → today)")]:
        cat.boundary.plot(ax=ax, color=GRID, lw=0.5)
        g = cat.loc[par.index].join(par[[col]]).sort_values("area_km2", ascending=False)
        g.plot(ax=ax, column=col, cmap="Blues", vmin=0, vmax=1, alpha=0.75, edgecolor="white", lw=0.4)
        reg = att.loc[g.index, "regulated"]
        ax.scatter(g.outlet_lon, g.outlet_lat, s=14, c=np.where(reg, INK, "white"), edgecolor=INK, lw=0.6, zorder=3)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    sm = mpl.cm.ScalarMappable(cmap="Blues", norm=mpl.colors.Normalize(0, 1))
    fig.colorbar(sm, ax=axs, shrink=0.7, label="KGE (≤ 0 shown as white)")
    fig.suptitle("GR4J skill per catchment (dots = gauges; black = regulated)", x=0.01, ha="left", color=INK)
    fig.savefig(FIGURES / "05b_gr4j_kge_map.png")
    plt.close(fig)

    # --- 3. parameters
    fig, axs = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, p in zip(axs, ["X1", "X2", "X3", "X4", "CTG", "Kf"]):
        v = par[p] if p in ("X1", "X2", "X3", "X4") else par.loc[par.snow_calibrated, p]
        lo, hi = S["gr4j"]["bounds"][p]
        bins = np.logspace(np.log10(lo), np.log10(hi), 16) if p in ("X1", "X3") else np.linspace(lo, hi, 16)
        ax.hist(v, bins=bins, color=C_SIM, edgecolor="white")
        if p in ("X1", "X3"):
            ax.set_xscale("log")
        ax.set_title(p + (" (snow gauges)" if p in ("CTG", "Kf") else ""), loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("gauges")
    fig.savefig(FIGURES / "05b_gr4j_parameters.png")
    plt.close(fig)

    # --- 4. per gauge
    out = FIGURES / "gr4j"
    out.mkdir(exist_ok=True)
    for gid in par.index:
        gauge_panel(gid, att.loc[gid], obs[gid], sim[gid], par.loc[gid], out / f"{gid}.png")
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
