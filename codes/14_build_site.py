#!/usr/bin/env python3
"""
14 — Build the public forecast website (GitHub Pages, folder docs/).

Reads today's forecasts (scripts 12, 13), the observed record (for the flood
levels) and the test-period skill of each model (script 08), and writes a
static site. Plots start on the forecast day (today): past days are not shown.
Products are published only if they exist for the run (WeatherNext 3 is
optional; the Google model's forcing is recorded as AIFS or AIFS + WN3).

  docs/index.html            map + hydrograph panel (Leaflet + Plotly from CDNs)
  docs/data/latest.json      everything the page shows for the latest run
  docs/data/gauges.geojson   gauge points and simplified catchments
  docs/data/archive/<init>.json   one file per daily run (the site's history)

ALERT LEVELS (per gauge, like Flood Hub's return-period warnings)
  Annual maxima of observed daily mean discharge (water years, >= 8 years of
  data) are fitted with a Gumbel distribution by the method of moments; the
  2-, 5- and 20-year return levels are the thresholds. A gauge's level is the
  highest threshold exceeded by the median of any forecast product in the next
  7 days. Daily means are lower than instantaneous peaks, so these are levels
  of daily flow, stated as such on the page. Gauges without enough record have
  no levels (has_levels = false) and are drawn differently, never as "safe".
"""
from __future__ import annotations

import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from hydrocat.config import DATA, P, ROOT, load_settings

S = load_settings()
SITE = ROOT / "docs"
EULER = 0.5772156649


def log(*a):
    print(*a, flush=True)


def gumbel_levels(q: pd.Series, T=(2, 5, 20)) -> dict | None:
    am = q.groupby(q.index.year + (q.index.month >= 10)).agg(["max", "count"])
    am = am[am["count"] >= 300]["max"]
    if len(am) < 8:
        return None
    beta = am.std() * np.sqrt(6) / np.pi
    mu = am.mean() - EULER * beta
    return {f"T{t}": float(mu - beta * np.log(-np.log(1 - 1 / t))) for t in T}


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main():
    fdir = DATA / "forecasts" / "catalonia"
    init = sorted(p.name for p in (fdir / "gr4j").iterdir())[-1]
    d0 = pd.Timestamp(dt.datetime.strptime(init, "%Y%m%d%H").date())
    gr = pd.read_parquet(fdir / "gr4j" / init / "q_forecast.parquet")
    gf = fdir / "google" / init / "q_forecast.parquet"
    go = pd.read_parquet(gf) if gf.exists() else pd.DataFrame(columns=["gauge_id"])
    obs = pd.read_parquet(P.q_daily)
    st = pd.read_csv(P.stations).set_index("gauge_id")
    att = pd.read_csv(P.attributes).set_index("gauge_id")
    met = pd.read_csv(P.metrics) if P.metrics.exists() else pd.DataFrame()
    cat = gpd.read_file(P.catchments).set_index("gauge_id")
    gauges = sorted(set(gr.gauge_id))

    out, feats = {}, []
    products = set(gr["product"])
    gforcing = sorted(set(go["forcing"])) if "forcing" in go else []
    for gid in gauges:
        lv = gumbel_levels(obs[gid].dropna()) if gid in obs else None
        g = gr[gr.gauge_id == gid]
        series = {}
        for prod in ["wn3", "aifs"]:
            if prod not in products:
                continue
            x = g[(g["product"] == prod) & (g.date >= d0)].set_index("date").q
            series[f"gr4j_{prod}"] = {"q50": [r(v) for v in x.values]}
        e = g[(g["product"] == "aifs_ens") & (g.date >= d0)].set_index("date")
        if len(e):
            series["gr4j_aifs_ens"] = {k: [r(v) for v in e[k].astype(float).values] for k in ["q05", "q25", "q50", "q75", "q95"]}
        h = go[go.gauge_id == gid] if len(go) else go
        for prod in ["google_released", "google_finetuned"]:
            x = h[(h["product"] == prod) & (h.date >= d0)].set_index("date") if len(h) else pd.DataFrame()
            if len(x):
                series[prod] = {k: [r(v) for v in x[k].values] for k in ["q05", "q25", "q50", "q75", "q95"]}
                series[prod]["dates"] = [d.strftime("%Y-%m-%d") for d in x.index]
        fc_dates = [d.strftime("%Y-%m-%d") for d in g[(g["product"] == "aifs") & (g.date >= d0)].date]
        peak7 = max([max([v for v in s["q50"][:7] if v is not None] or [0]) for s in series.values()] or [0])
        level = 0
        if lv:
            level = sum(peak7 >= lv[k] for k in ("T2", "T5", "T20"))
        skill = {}
        if len(met):
            mm = met[met.gauge_id == gid].set_index("model")
            skill = {k: {"KGE": r(mm.KGE.get(k), 2), "NSE": r(mm.NSE.get(k), 2)} for k in mm.index}
        out[gid] = dict(name=str(st.loc[gid, "name"]), river=str(st.loc[gid, "river"]),
                        area_km2=r(cat.area_km2.get(gid), 0), regulated=bool(att.regulated.get(gid, False)),
                        fc_dates=fc_dates, series=series, levels={k: r(v, 2) for k, v in (lv or {}).items()},
                        has_levels=lv is not None, level=int(level), peak7=r(peak7, 2), skill=skill)
        c = cat.loc[gid]
        feats.append({"type": "Feature", "properties": {"id": gid, "level": int(level), "has_levels": lv is not None},
                      "geometry": {"type": "Point", "coordinates": [c.gauge_lon, c.gauge_lat]}})

    wn3_used = "wn3" in products or "aifs+wn3" in gforcing
    sources = {"aifs": "ECMWF AIFS open data (CC BY 4.0), deterministic + 50-member ensemble rain",
               "observations": "Agència Catalana de l'Aigua (open data)"}
    if wn3_used:
        sources["weathernext3"] = "Google DeepMind WeatherNext 3 (Earth Engine), ensemble mean"
    meta = {"init": init, "issued": d0.strftime("%Y-%m-%d"),
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "weathernext3": wn3_used, "google_forcing": "AIFS + WN3" if "aifs+wn3" in gforcing else "AIFS",
            "sources": sources}
    (SITE / "data" / "archive").mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "gauges": out}
    (SITE / "data" / "latest.json").write_text(json.dumps(payload, separators=(",", ":")))
    (SITE / "data" / "archive" / f"{init}.json").write_text(json.dumps(payload, separators=(",", ":")))
    catg = cat.loc[gauges, ["geometry"]].copy()
    catg["geometry"] = catg.geometry.simplify(0.003)
    geo = {"type": "FeatureCollection", "features": feats,
           "catchments": json.loads(catg.reset_index().to_json())}
    (SITE / "data" / "gauges.geojson").write_text(json.dumps(geo, separators=(",", ":")))
    (SITE / "index.html").write_text((ROOT / "codes" / "site_template.html").read_text())
    (SITE / ".nojekyll").write_text("")
    n = pd.Series([g["level"] for g in out.values()]).value_counts().sort_index().to_dict()
    log(f"-> docs/ built for run {init}: {len(out)} gauges, alert levels {n}, WeatherNext 3 {'yes' if wn3_used else 'no'}")


if __name__ == "__main__":
    main()
