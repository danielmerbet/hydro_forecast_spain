#!/usr/bin/env python3
"""
05 — Calibrate GR4J + CemaNeige at every gauge.

SET-UP
  Model      GR4J (4 parameters) + CemaNeige snow (2 parameters, 5 elevation
             bands), hydrocat.gr4j — verified against airGR (tests/).
  Forcing    ERA5-Land catchment means (script 03), precipitation rescaled to the
             gauge-based EMO-1 monthly climatology (scripts 03b-03d; configurable),
             mean/min/max
             temperature, and FAO-56 Penman-Monteith reference evapotranspiration
             computed from ERA5-Land radiation, humidity, pressure and wind
             (no temperature-only PET formula).
  Periods    warm-up  2003-10-01 .. 2008-09-30   (5 years; stores spin up, not scored)
             calib    2008-10-01 .. 2023-09-30   (parameters fitted here)
             test     2023-10-01 .. today        (never seen; shared with Google's model)
  Objective  mean of KGE(Q) and KGE(sqrt Q) (Gupta et al., 2009), only on days
             with observations: peaks dominate but low flows keep weight. Plain
             KGE(Q) produced degenerate fits (routing store -> 1 mm, baseflow -> 0)
             at a quarter of the gauges. Configurable (gr4j.objective).
  Bounds     X3 >= 10 mm for the same reason (see settings.yaml).
  Optimiser  scipy differential evolution; X1 and X3 are searched in log space.
  Snow       CTG and Kf are calibrated only where the catchment reaches above
             `snow_if_max_elev_m` (Pyrenees / Montseny); elsewhere they are
             fixed at airGR's median values and CemaNeige is essentially inert.

ZERO FLOW (intermittent rivers)
  GR4J never produces exactly zero flow. After calibration, a per-gauge
  threshold q0 is derived (settings.yaml gr4j.zero_flow): observed zero is
  Q < 0.001 m3/s (ACA's reporting resolution), counted only in spells the
  river visibly receded into (see credible_zeros); a NEAR-NATURAL gauge is
  intermittent if it has >= 1 such zero day per year on average in the
  calibration period (Messager et al., 2021) — below dams and weirs zeros are
  management decisions GR4J cannot know about; q0 is then the simulated flow whose non-exceedance frequency in
  the calibration period equals the observed zero-flow frequency, and
  simulated flows below q0 are set to 0. Perennial gauges only get the
  0.001 m3/s floor. All reported metrics use the thresholded series. The
  threshold choice matters for intermittence metrics (Yu et al., 2024,
  Hydrol. Process. 38, e15300), hence the explicit, data-based rule.

The whole 2003 → today series is simulated with the final parameters, so the
test-period simulation starts from states produced by the model itself — the
same way the operational forecast will run.

Writes
  data/processed/gr4j/parameters.csv       parameters + calibration & test metrics per gauge
  data/processed/gr4j/q_sim_daily.parquet  simulated discharge, m3/s (date x gauge)
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from hydrocat import metrics
from hydrocat.config import PROCESSED, ROOT, P, load_settings
from hydrocat.forcing import load_forcing
from hydrocat.gr4j import CatchmentModel

S = load_settings()
PER = S["periods"]
GR = S["gr4j"]
B = GR["bounds"]


def log(*a):
    print(*a, flush=True)


def objective_fn(name):
    if name == "kge":
        return lambda o, s: metrics.kge(o, s)
    if name == "nse":
        return lambda o, s: metrics.nse(o, s)
    if name == "kge_sqrt":
        return lambda o, s: 0.5 * (metrics.kge(o, s) + metrics.kge(np.sqrt(o), np.sqrt(np.clip(s, 0, None))))
    raise ValueError(name)


def to_params(u, snow: bool):
    x = [np.exp(u[0]), u[1], np.exp(u[2]), u[3]]
    x += [u[4], u[5]] if snow else list(GR["default_snow_params"])
    return x


def credible_zeros(obs: np.ndarray, zf: dict) -> np.ndarray:
    """Observed zero-flow days that look like real drying, not recording dropouts.

    A spell of Q < obs_zero_m3s counts only if the last valid value before it
    was already low (< drying_fraction x the gauge's median flow), i.e. the
    river receded into it. Spells that start right after a data gap or out of
    normal flow (e.g. a 12-day block of zeros on the lower Llobregat, EA099)
    are treated as missing data, not as dry river bed.
    """
    z = np.isfinite(obs) & (obs < zf["obs_zero_m3s"])
    med = np.nanmedian(obs)
    out = np.zeros_like(z)
    i, n = 0, len(obs)
    while i < n:
        if z[i]:
            j = i
            while j < n and z[j]:
                j += 1
            prev = obs[i - 1] if i > 0 else np.nan
            if np.isfinite(prev) and prev < zf["drying_fraction"] * med:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def calibrate_one(job):
    gid, forc, hypso, q_obs_mm, area_km2, snow, regulated, calib_end = job
    t0 = time.time()
    model = CatchmentModel(forc.total_precipitation.values, forc.temperature_2m.values,
                           forc.temperature_2m_min.values, forc.temperature_2m_max.values,
                           forc.pet_fao56.values, hypso, GR["n_elevation_layers"])
    dates = forc.date.values
    cal = (dates >= np.datetime64(PER["calib_start"])) & (dates <= np.datetime64(calib_end))
    obs = q_obs_mm.copy()
    obs_cal = np.where(cal, obs, np.nan)
    ok = np.isfinite(obs_cal)
    f = objective_fn(GR["objective"])

    def cost(u):
        q = model.run(to_params(u, snow))
        v = f(obs_cal[ok], q[ok])
        return 10.0 if not np.isfinite(v) else 1.0 - v

    bounds = [np.log(B["X1"]), B["X2"], np.log(B["X3"]), B["X4"]]
    if snow:
        bounds += [B["CTG"], B["Kf"]]
    opt = GR["optimizer"]
    res = differential_evolution(cost, bounds, maxiter=opt["maxiter"], popsize=opt["popsize"],
                                 seed=opt["seed"], tol=1e-6, polish=True, init="sobol")
    par = to_params(res.x, snow)
    k = area_km2 / 86.4                                   # mm/day -> m3/s
    q = model.run(par) * k
    obs = obs * k
    tst = dates >= np.datetime64(PER["test_start"])

    # --- zero-flow threshold (see module docstring)
    zf = GR["zero_flow"]
    oc = cal & np.isfinite(obs)
    f0 = float(np.mean(credible_zeros(obs, zf)[oc]))
    intermittent = (f0 * 365.25 >= zf["min_zero_days_per_year"]) and not regulated
    q0 = float(np.quantile(q[oc], f0)) if intermittent else zf["obs_zero_m3s"]
    q0 = max(q0, zf["obs_zero_m3s"])
    q = np.where(q < q0, 0.0, q)

    def zero_frac(x, m):
        m = m & np.isfinite(obs)
        return float(np.mean(x[m] < zf["obs_zero_m3s"])) if m.any() else np.nan

    m_cal = metrics.all_metrics(obs[cal], q[cal])
    m_tst = metrics.all_metrics(obs[tst], q[tst])
    row = dict(gauge_id=gid, X1=par[0], X2=par[1], X3=par[2], X4=par[3], CTG=par[4], Kf=par[5],
               snow_calibrated=snow, intermittent=intermittent, zero_days_per_year_obs_cal=round(f0 * 365.25, 1),
               q0_m3s=q0, zero_frac_obs_test=zero_frac(obs, tst), zero_frac_sim_test=zero_frac(q, tst),
               n_evals=res.nfev, seconds=round(time.time() - t0, 1))
    row |= {f"cal_{k}": v for k, v in m_cal.items()} | {f"test_{k}": v for k, v in m_tst.items()}
    return row, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--precip", default="config", choices=["config", "none", "emo1_monthly_scaling"],
                    help="precipitation correction (default: settings.yaml forcing.precip_correction)")
    ap.add_argument("--out", default=None, help="output folder (default: the domain's gr4j folder)")
    ap.add_argument("--gauges", nargs="*", help="only these gauge ids (default: all modelled)")
    args = ap.parse_args()
    D = S["domains"][args.domain]
    spain = args.domain == "spain"
    base = PROCESSED / "spain" if spain else PROCESSED
    out = ROOT / args.out if args.out else (base / "gr4j")
    out.mkdir(parents=True, exist_ok=True)
    # Spain: CEDEX (the calibration data) ends on 2022-09-30
    calib_end = min(PER["calib_end"], "2022-09-30") if spain else PER["calib_end"]

    # GR4J needs only area (script 02) and hypsometry (script 04), not the
    # Google-specific attributes.
    att = gpd.read_file(ROOT / D["catchments"]).drop(columns="geometry")
    st = pd.read_csv(base / "stations.csv").set_index("gauge_id")
    ok_col = "calibratable" if spain else "modelled"
    hy = pd.read_csv(base / "hypsometry.csv").set_index("gauge_id")
    att["elev_max_m"] = att.gauge_id.map(hy["p100"])
    # zero-flow thresholds are only applied at near-natural gauges (script 04)
    regulated = pd.read_csv(base / "catchment_attributes.csv").set_index("gauge_id")["regulated"].to_dict()
    forcing = load_forcing(args.domain, None if args.precip == "config" else args.precip)
    forcing = forcing[forcing.date >= PER["warmup_start"]]
    q_obs = pd.read_parquet(base / ("q_obs_daily.parquet"))

    gauges = [g for g in att.gauge_id if g in st.index and bool(st.loc[g, ok_col]) and g in hy.index]
    if args.gauges:
        gauges = [g for g in gauges if g in args.gauges]
    fg = dict(tuple(forcing.groupby("gauge_id")))
    jobs = []
    for gid in gauges:
        a = att.set_index("gauge_id").loc[gid]
        f = fg[gid].sort_values("date").reset_index(drop=True)
        q = q_obs[gid].reindex(f.date.values).values * 86.4 / a.area_km2      # m3/s -> mm/day
        snow = bool(a.elev_max_m >= GR["snow_if_max_elev_m"])
        jobs.append((gid, f, hy.loc[gid].values.astype(float), q, float(a.area_km2), snow,
                     bool(regulated.get(gid, False)), calib_end))
    log(f"{args.domain}: calibrating {len(jobs)} gauges ({sum(j[5] for j in jobs)} with snow) "
        f"on {PER['calib_start']} .. {calib_end}, objective {GR['objective']}")

    rows, sims = [], {}
    with Pool(GR["optimizer"]["workers"]) as pool:
        for row, qs in pool.imap_unordered(calibrate_one, jobs):
            rows.append(row)
            sims[row["gauge_id"]] = qs
            log(f"  {row['gauge_id']:7s} KGE cal {row['cal_KGE']:5.2f}  test {row['test_KGE']:5.2f}  "
                f"NSE test {row['test_NSE']:5.2f}  logNSE test {row['test_logNSE']:5.2f}"
                + (f"  intermittent ({row['zero_days_per_year_obs_cal']:.0f} d/yr, q0 {row['q0_m3s']:.3g})"
                   if row["intermittent"] else "") + f"  ({row['seconds']:.0f} s)")

    par = pd.DataFrame(rows).sort_values("gauge_id")
    par.to_csv(out / "parameters.csv", index=False)
    dates = forcing.date.drop_duplicates().sort_values().values
    sim = pd.DataFrame(sims, index=pd.DatetimeIndex(dates, name="date"))[sorted(sims)]
    sim.to_parquet(out / "q_sim_daily.parquet")
    log(f"-> {(out / 'parameters.csv').relative_to(ROOT)}\n-> {(out / 'q_sim_daily.parquet').relative_to(ROOT)}")
    log(f"median KGE  calib {par.cal_KGE.median():.2f}   test {par.test_KGE.median():.2f}")


if __name__ == "__main__":
    main()
