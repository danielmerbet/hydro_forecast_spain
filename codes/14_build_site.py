#!/usr/bin/env python3
"""
14 — Build the public forecast website (GitHub Pages, folder docs/).

Reads today's forecasts of every domain that has them (scripts 12, 13 with
--domain catalonia / spain), the flood levels and the skill of each model, and
writes one static site for all of them. Plots start on the forecast day
(today): past days are not shown. Products are published only if they exist
for the run (WeatherNext 3 is optional; the Google model's forcing is recorded
as AIFS or AIFS + WN3).

  docs/index.html                 map + hydrograph panel (Leaflet + Plotly from CDNs)
  docs/data/latest.json           everything the page shows for the latest run
  docs/data/gauges.geojson        gauge points (drawn on load)
  docs/data/catchments.geojson    simplified catchments (loaded after the map;
                                  only the selected gauge's catchment is drawn)
  docs/data/archive/<init>.json   one file per daily run (the site's history)

FLOOD LEVELS (per gauge, like Flood Hub's return-period warnings)
  Annual maxima of observed daily mean discharge (water years, >= 300 days
  each, >= 8 years) are fitted with a Gumbel distribution by the method of
  moments; the 2-, 5- and 20-year return levels are the thresholds. Daily
  means are lower than instantaneous peaks, so these are levels of daily flow,
  stated as such on the page. Gauges without enough record have no levels
  (has_levels = false) and are drawn as hollow circles, never as "safe".
  Levels are recomputed whenever the full observed record is on disk and are
  stored in data/processed[/<domain>]/flood_levels.csv; the daily GitHub run
  of Spain (no CEDEX history on the runner) reads that file.

ALERT COLOUR (decided 2026-09-30)
  level     highest threshold reached by the MEDIAN of the skilful products
            over the whole forecast (15 days GR4J, 7 days Google):
            GR4J with WeatherNext 3, AIFS and the AIFS ensemble, and Google's
            fine-tuned model. Google's released model is shown in the plots for
            comparison but does not set the colour: over the test period it
            roughly doubles the observed volume in Catalonia (median bias +98 %),
            so its median and especially its 95 % bound are strongly inflated.
  possible  highest threshold reached by the 95 % bound of GR4J driven by the
            50 AIFS ensemble members, the only range that comes from weather
            uncertainty; drawn as a ring when higher than `level`.
  skill     a model sets the colour at a gauge only if it has skill there:
            KGE > -0.41, the value of the mean-flow benchmark (Knoben et al.,
            2019, HESS 23, 4323-4331), on the test period when available,
            otherwise calibration. Where no model qualifies (e.g. ES9026 below
            the Ebro dam, test KGE -1.6) the gauge is drawn hollow, "no
            skilful model", and its forecasts are plotted for information only.
"""
from __future__ import annotations

import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from hydrocat.config import DATA, P, PROCESSED, ROOT, load_settings

S = load_settings()
SITE = ROOT / "docs"
EULER = 0.5772156649
T_KEYS = ("T2", "T5", "T20")
ALERT_MEDIANS = ["gr4j_wn3", "gr4j_aifs", "gr4j_aifs_ens", "google_finetuned"]
ALERT_UPPER = "gr4j_aifs_ens"
KGE_BENCHMARK = -0.41                     # Knoben et al. (2019): KGE of the mean-flow benchmark


def log(*a):
    print(*a, flush=True)


def gumbel_levels(q: pd.Series, T=(2, 5, 20)) -> dict | None:
    am = q.groupby(q.index.year + (q.index.month >= 10)).agg(["max", "count"])
    am = am[am["count"] >= 300]["max"]
    if len(am) < 8:
        return None
    beta = am.std() * np.sqrt(6) / np.pi
    mu = am.mean() - EULER * beta
    return {f"T{t}": float(mu - beta * np.log(-np.log(1 - 1 / t))) for t in T} | {"n_years": len(am)}


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def base_dir(domain):
    return PROCESSED if domain == "catalonia" else PROCESSED / domain


def flood_levels(domain) -> pd.DataFrame:
    """Gumbel levels from the full observed record if present, else the stored table."""
    f = base_dir(domain) / "flood_levels.csv"
    full = P.q_daily if domain == "catalonia" else base_dir(domain) / "q_obs_daily.parquet"
    if full.exists():
        obs = pd.read_parquet(full)
        rows = [{"gauge_id": g, **lv} for g in obs.columns if (lv := gumbel_levels(obs[g].dropna()))]
        lv = pd.DataFrame(rows, columns=["gauge_id", *T_KEYS, "n_years"])
        lv.round(3).to_csv(f, index=False)
        return lv.set_index("gauge_id")
    return pd.read_csv(f).set_index("gauge_id")


def domain_info(domain) -> tuple[pd.DataFrame, dict]:
    """Per-gauge name / label / regulation, and skill {gauge: {model: {KGE, NSE}}}."""
    b = base_dir(domain)
    att = pd.read_csv(b / "catchment_attributes.csv").set_index("gauge_id")
    skill: dict = {}
    if domain == "catalonia":
        st = pd.read_csv(P.stations).set_index("gauge_id")
        info = pd.DataFrame({"name": st.name, "label": st.river, "authority": "ACA"})
        if P.metrics.exists():
            for _, m in pd.read_csv(P.metrics).iterrows():
                skill.setdefault(m.gauge_id, {})[m.model] = {"KGE": r(m.KGE, 2), "NSE": r(m.NSE, 2)}
    else:
        st = pd.read_csv(b / "stations.csv").set_index("gauge_id")
        info = pd.DataFrame({"name": st.name, "label": st.demarcacion, "authority": st.demarcacion})
        par = pd.read_csv(b / "gr4j" / "parameters.csv").set_index("gauge_id")
        for g, p in par.iterrows():
            s = {"gr4j_cal": {"KGE": r(p.cal_KGE, 2), "NSE": r(p.cal_NSE, 2)}}
            if np.isfinite(p.test_KGE):
                s["gr4j"] = {"KGE": r(p.test_KGE, 2), "NSE": r(p.test_NSE, 2)}
            skill[g] = s
    info["regulated"] = att.regulated.reindex(info.index).astype("boolean").fillna(False).astype(bool)
    return info, skill


def build_domain(domain):
    fdir = DATA / "forecasts" / domain
    init = sorted(p.name for p in (fdir / "gr4j").iterdir())[-1]
    d0 = pd.Timestamp(dt.datetime.strptime(init, "%Y%m%d%H").date())
    gr = pd.read_parquet(fdir / "gr4j" / init / "q_forecast.parquet")
    gf = fdir / "google" / init / "q_forecast.parquet"
    go = pd.read_parquet(gf) if gf.exists() else pd.DataFrame(columns=["gauge_id", "product", "date"])
    products = set(gr["product"])
    gforcing = sorted(set(go["forcing"])) if "forcing" in go else []
    levels = flood_levels(domain)
    info, skill = domain_info(domain)
    cat = gpd.read_file(ROOT / S["domains"][domain]["catchments"]).set_index("gauge_id")
    gauges = sorted(set(gr.gauge_id))

    out = {}
    for gid in gauges:
        lv = levels.loc[gid, list(T_KEYS)].to_dict() if gid in levels.index else None
        g = gr[(gr.gauge_id == gid) & (gr.date >= d0)]
        series = {}
        for prod in ["wn3", "aifs"]:
            if prod in products:
                series[f"gr4j_{prod}"] = {"q50": [r(v) for v in g[g["product"] == prod].q.values]}
        e = g[g["product"] == "aifs_ens"]
        if len(e):
            series["gr4j_aifs_ens"] = {k: [r(v) for v in e[k].astype(float).values] for k in ["q05", "q25", "q50", "q75", "q95"]}
        h = go[(go.gauge_id == gid) & (go.date >= d0)] if len(go) else go
        for prod in ["google_released", "google_finetuned"]:
            x = h[h["product"] == prod] if len(h) else h
            if len(x):
                series[prod] = {k: [r(v) for v in x[k].values] for k in ["q05", "q25", "q50", "q75", "q95"]}
                series[prod]["dates"] = [d.strftime("%Y-%m-%d") for d in x.date]
        fc_dates = [d.strftime("%Y-%m-%d") for d in g[g["product"] == "aifs"].date]

        def vmax(vals):
            return max([v for v in vals if v is not None] or [0.0])

        sk = skill.get(gid, {})

        def skilful(model):          # test KGE if available, else calibration; unknown -> allowed
            v = (sk.get(model) or sk.get(f"{model}_cal") or {}).get("KGE")
            return v is None or v > KGE_BENCHMARK

        use = [k for k in ALERT_MEDIANS if k in series and skilful("gr4j" if k.startswith("gr4j") else k)]
        peak = max([vmax(series[k]["q50"]) for k in use] or [0.0])
        upper = vmax(series[ALERT_UPPER]["q95"]) if ALERT_UPPER in use else 0.0
        level = sum(peak >= lv[k] for k in T_KEYS) if lv else 0
        possible = sum(upper >= lv[k] for k in T_KEYS) if lv else 0
        c = cat.loc[gid]
        updated = bool(g.updated.any()) if "updated" in g else None
        out[gid] = dict(domain=domain, name=str(info.name.get(gid, gid)), label=str(info.label.get(gid, "")),
                        authority=str(info.authority.get(gid, "")), area_km2=r(c.area_km2, 0),
                        regulated=bool(info.regulated.get(gid, False)), updated=updated,
                        lon=r(c.gauge_lon, 5), lat=r(c.gauge_lat, 5),
                        fc_dates=fc_dates, series=series, levels={k: r(v, 2) for k, v in (lv or {}).items()},
                        has_levels=lv is not None, skilful=bool(use),
                        level=int(level), possible=int(max(possible, level)),
                        peak=r(peak, 2), skill=skill.get(gid, {}))
    wn3 = "wn3" in products or "aifs+wn3" in gforcing
    meta = {"init": init, "issued": d0.strftime("%Y-%m-%d"), "weathernext3": wn3,
            "google_forcing": "AIFS + WN3" if "aifs+wn3" in gforcing else "AIFS"}
    cat = cat.loc[gauges, ["geometry", "area_km2"]].copy()
    tol = np.where(cat.area_km2 > 5000, 0.01, 0.004)                     # degrees
    cat["geometry"] = [shapely.set_precision(gm.simplify(t), 0.001) for gm, t in zip(cat.geometry, tol)]
    return out, meta, cat[["geometry"]]


def main():
    doms = [d for d in S["domains"] if (DATA / "forecasts" / d / "gr4j").exists()]
    gauges, metas, cats = {}, {}, []
    for d in doms:
        g, m, c = build_domain(d)
        gauges |= g
        metas[d] = m
        cats.append(c)
        log(f"   {d}: {len(g)} gauges, run {m['init']}, WeatherNext 3 {'yes' if m['weathernext3'] else 'no'}")
    first = metas.get("catalonia") or next(iter(metas.values()))
    wn3 = any(m["weathernext3"] for m in metas.values())
    sources = {"aifs": "ECMWF AIFS open data (CC BY 4.0), deterministic + 50-member ensemble rain",
               "observations": "ACA; CEDEX Anuario de Aforos; SAIH Ebro, Júcar, Guadalquivir, Segura"}
    if wn3:
        sources["weathernext3"] = "Google DeepMind WeatherNext 3 (Earth Engine), ensemble mean"
    meta = {"init": first["init"], "issued": first["issued"], "domains": metas,
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "weathernext3": all(m["weathernext3"] for m in metas.values()),
            "google_forcing": first["google_forcing"], "sources": sources}

    (SITE / "data" / "archive").mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"meta": meta, "gauges": gauges}, separators=(",", ":"))
    (SITE / "data" / "latest.json").write_text(payload)
    (SITE / "data" / "archive" / f"{meta['init']}.json").write_text(payload)
    feats = [{"type": "Feature", "properties": {"id": k, "level": v["level"], "possible": v["possible"],
                                                "has_levels": v["has_levels"], "skilful": v["skilful"]},
              "geometry": {"type": "Point", "coordinates": [v["lon"], v["lat"]]}} for k, v in gauges.items()]
    (SITE / "data" / "gauges.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                             separators=(",", ":")))
    catg = pd.concat(cats)
    (SITE / "data" / "catchments.geojson").write_text(catg.reset_index().to_json(drop_id=True))
    (SITE / "index.html").write_text((ROOT / "codes" / "site_template.html").read_text())
    (SITE / ".nojekyll").write_text("")
    n = pd.Series([g["level"] for g in gauges.values()]).value_counts().sort_index().to_dict()
    ring = sum(g["possible"] > g["level"] for g in gauges.values())
    log(f"-> docs/ built: {len(gauges)} gauges, alert levels {n}, {ring} with a higher possible level; "
        f"latest.json {len(payload) / 1e6:.1f} MB, catchments {(SITE / 'data' / 'catchments.geojson').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
