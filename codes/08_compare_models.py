#!/usr/bin/env python3
"""
08 — GR4J (calibrated per gauge) vs Google's model (global, not calibrated here).

Both are driven by the same ERA5-Land forcing and scored on the same days:
the test period (2023-10-01 → last common day), on days with an observation.

Metrics (hydrocat.metrics): NSE, KGE and its components (r, alpha = variability
ratio, beta = bias ratio), PBIAS, NSE of log flows (low flows), and the
relative error of the mean of the top 1 % observed flows (peaks).

Writes
  outputs/tables/metrics_test_period.csv        one row per gauge x model
  outputs/tables/metrics_summary.csv            medians by model and regulation class
  outputs/figures/08_skill_cdf.png              CDF of KGE / NSE across gauges
  outputs/figures/08_gr4j_vs_google_scatter.png per-gauge KGE, GR4J vs each Google arm
  outputs/figures/08_skill_map.png              which model wins where
  outputs/figures/gauges/<gauge>.png            hydrograph + flow-duration curve per gauge
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrocat import metrics
from hydrocat.config import FIGURES, ROOT, TABLES, P, load_settings

S = load_settings()

# Colour = model identity, fixed order (validated categorical slots 1-3); obs in text ink.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = {  # key -> (label, colour); fixed categorical order, never re-assigned
    "gr4j": ("GR4J (calibrated per gauge)", "#2a78d6"),
    "google_baseline": ("Google (global, as released)", "#eb6834"),
    "google_finetuned": ("Google (fine-tuned on ACA)", "#1baf7a"),
    "gr4j_emo1": ("GR4J, rain scaled to EMO-1 (sensitivity)", "#eda100"),
}
# "gr4j" is the run with RAW ERA5-Land rain (data/processed/gr4j/run_era5l_raw/),
# i.e. exactly the forcing Google's model sees: the fair head-to-head. The run
# with EMO-1-scaled rain is shown as a sensitivity test only.

mpl.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
})


def log(*a):
    print(*a, flush=True)


def load_sims():
    raw = P.gr4j_sim.parent / "run_era5l_raw" / "q_sim_daily.parquet"
    if S["forcing"].get("precip_correction", "none") == "none":
        sims = {"gr4j": pd.read_parquet(P.gr4j_sim)}
    else:
        sims = {"gr4j": pd.read_parquet(raw), "gr4j_emo1": pd.read_parquet(P.gr4j_sim)}
    for run in ["baseline", "finetuned"]:
        f = ROOT / str(P.google_sim).format(run=run)
        if f.exists():
            sims[f"google_{run}"] = pd.read_parquet(f)
    return sims


def fdc(x):
    x = np.sort(x[np.isfinite(x)])[::-1]
    return np.arange(1, len(x) + 1) / (len(x) + 1) * 100, x


def gauge_figure(gid, meta, obs, sims, band, m, out):
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2.2, 2.2, 1.7], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[:, :2])
    if band is not None:
        ax.fill_between(band.index, band.q25, band.q75, color=MODELS["google_baseline"][1], alpha=0.18,
                        lw=0, label="Google baseline 25–75 %")
    for k, s in sims.items():
        ax.plot(s.index, s.values, lw=1.1, color=MODELS[k][1], label=MODELS[k][0])
    ax.plot(obs.index, obs.values, lw=1.3, color=INK, label="Observed (ACA)")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    reg = "regulated" if meta.regulated else "near-natural"
    ax.set_title(f"{gid} — {meta['name']} ({meta.river}), {meta.area_km2:.0f} km², {reg}",
                 loc="left", fontsize=10, color=INK)

    ax2 = fig.add_subplot(gs[0, 2])
    p, v = fdc(obs.values)
    ax2.plot(p, v, color=INK, lw=1.3)
    for k, s in sims.items():
        p, v = fdc(s.reindex(obs.dropna().index).values)
        ax2.plot(p, v, color=MODELS[k][1], lw=1.1)
    ax2.set_yscale("log")
    ax2.set_xlabel("Exceedance (%)")
    ax2.set_title("Flow-duration curve", loc="left", fontsize=9, color=INK)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis("off")
    short = {"gr4j": "GR4J", "gr4j_emo1": "GR4J (EMO-1 rain)", "google_baseline": "Google released",
             "google_finetuned": "Google fine-tuned"}
    rows = [["", "KGE", "NSE", "bias"]]
    for k in sims:
        r = m[m.model == k].iloc[0]
        rows.append([short[k], f"{r.KGE:.2f}", f"{r.NSE:.2f}", f"{r.PBIAS:+.0f} %"])
    t = ax3.table(cellText=rows, loc="center", cellLoc="right", edges="horizontal",
                  colWidths=[0.46, 0.17, 0.17, 0.2])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1.15, 1.35)
    ax3.set_title("Test-period skill", loc="left", fontsize=9, color=INK)
    fig.savefig(out)
    plt.close(fig)


def main():
    q_obs = pd.read_parquet(P.q_daily)
    att = pd.read_csv(P.attributes).set_index("gauge_id")
    st = pd.read_csv(P.stations).set_index("gauge_id")
    sims = load_sims()
    log(f"models: {list(sims)}")
    gauges = sorted(set.intersection(*[set(s.columns) for s in sims.values()]))
    t0 = pd.Timestamp(S["periods"]["test_start"])
    t1 = min(s.dropna(how="all").index.max() for s in sims.values())
    log(f"test period {t0.date()} .. {t1.date()}, {len(gauges)} gauges")

    bands = {}
    qf = P.google_dir / "q_sim_quantiles_baseline.parquet"
    if qf.exists():
        qq = pd.read_parquet(qf)
        qq = qq[qq.lead == S["google"]["simulation_lead"]]
        bands = {g: d.set_index("date")[["q25", "q75"]] for g, d in qq.groupby("gauge_id")}

    rows = []
    gdir = FIGURES / "gauges"
    gdir.mkdir(exist_ok=True)
    for gid in gauges:
        obs = q_obs[gid][t0:t1]
        sub = {k: s[gid][t0:t1] for k, s in sims.items()}
        # score only days where every model and the observation exist
        ok = obs.notna()
        for s in sub.values():
            ok &= s.reindex(obs.index).notna()
        mrows = []
        for k, s in sub.items():
            r = metrics.all_metrics(obs[ok].values, s.reindex(obs.index)[ok].values)
            mrows.append(dict(gauge_id=gid, model=k, **r))
        m = pd.DataFrame(mrows)
        rows.append(m)
        meta = att.loc[gid].copy()
        meta["name"] = st.loc[gid, "name"]
        band = bands.get(gid)
        gauge_figure(gid, meta, obs, sub, band[t0:t1] if band is not None else None, m, gdir / f"{gid}.png")

    met = pd.concat(rows, ignore_index=True)
    met = met.merge(att[["regulated", "area_km2", "area_ok", "reservoirs_upstream"]], left_on="gauge_id",
                    right_index=True)
    met.to_csv(P.metrics, index=False)
    summ = (met[met.area_ok].groupby(["regulated", "model"])[["KGE", "NSE", "logNSE", "PBIAS", "peak_err_pct"]]
            .median().round(3))
    summ.to_csv(TABLES / "metrics_summary.csv")
    log("median test-period skill (gauges with correct catchment area):\n" + summ.to_string())

    # --- CDFs
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axs, ["KGE", "NSE"]):
        for k, (lab, c) in MODELS.items():
            v = np.sort(met.loc[(met.model == k) & met.area_ok, col].dropna().values)
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=1.8, label=lab)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(f"{col} (test period)")
        ax.set_ylabel("Fraction of gauges ≤ x")
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.16), ncol=2)
    fig.suptitle("Skill across ACA gauges, Oct 2023 → present (same ERA5-Land forcing)", x=0.01, ha="left")
    fig.savefig(FIGURES / "08_skill_cdf.png")
    plt.close(fig)

    # --- scatter GR4J vs each Google arm, per gauge (one panel per arm)
    w = met.pivot(index="gauge_id", columns="model", values="KGE")
    reg = att.loc[w.index, "regulated"]
    arms = [k for k in MODELS if k.startswith("google") and k in w]
    fig, axs = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 5), squeeze=False)
    for ax, arm in zip(axs[0], arms):
        col = MODELS[arm][1]
        for flag, mk, lab in [(False, "o", "near-natural"), (True, "s", "regulated")]:
            sel = reg == flag
            ax.scatter(w.loc[sel, "gr4j"].clip(-1), w.loc[sel, arm].clip(-1), s=34, marker=mk,
                       facecolor="none" if flag else col, edgecolor=col, label=lab)
        for gid, r in w.iterrows():
            ax.annotate(gid, (max(r.gr4j, -1), max(r[arm], -1)), fontsize=5.5, color=INK2,
                        xytext=(3, 2), textcoords="offset points")
        ax.plot([-1, 1], [-1, 1], color=INK2, lw=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("KGE — GR4J (calibrated per gauge)")
        ax.set_ylabel(f"KGE — {MODELS[arm][0]}")
        ax.legend(loc="upper left")
        better = int((w[arm] > w["gr4j"]).sum())
        ax.set_title(f"Above the line: Google better ({better}/{len(w)} gauges)", loc="left",
                     fontsize=9, color=INK2)
    fig.savefig(FIGURES / "08_gr4j_vs_google_scatter.png")
    plt.close(fig)

    # --- map: KGE difference (best available Google arm - GR4J); diverging, neutral grey at 0
    arm = "google_finetuned" if "google_finetuned" in w else arms[0]
    cat = gpd.read_file(P.catchments).set_index("gauge_id")
    d = (w[arm] - w["gr4j"]).clip(-0.6, 0.6)
    fig, ax = plt.subplots(figsize=(8, 7))
    cat.loc[w.index].sort_values("area_km2", ascending=False).boundary.plot(ax=ax, color=GRID, lw=0.6)
    sc = ax.scatter(att.loc[w.index].outlet_lon, att.loc[w.index].outlet_lat, c=d, cmap="RdBu_r",
                    vmin=-0.6, vmax=0.6, s=46, edgecolor=INK2, linewidth=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label(f"KGE({MODELS[arm][0]}) − KGE(GR4J)\n(red: Google better, blue: GR4J better)")
    ax.set_title("Where each model does better (test period)", loc="left", color=INK)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.savefig(FIGURES / "08_skill_map.png")
    plt.close(fig)
    log(f"-> {P.metrics.relative_to(ROOT)}, figures in outputs/figures/")


if __name__ == "__main__":
    main()
