#!/usr/bin/env python3
"""
01 — Observed daily streamflow at ACA gauges (2007 → today).

WHY TWO SOURCES
  * data/raw/aca_export/  ACA "Consulta de dades del medi" daily export,
    2007-01-01 .. 2024-10-20. Long enough to calibrate GR4J, but a one-off
    file (see its README for provenance).
  * Catalan open-data portal, dataset 3yr3-vq6y ("Cabals dels rius a les conques
    internes de Catalunya"): 5-minute data, 2020 → today, updated in near real
    time, no authentication. This is what the daily operational run will use.
  Daily means from the export are used before `gauges.socrata_from`, open-data
  daily means from that date on. The two overlap for a year (2023-10 .. 2024-10),
  and that overlap is used to check they agree (table written below).

UNIT BUG IN THE OPEN-DATA FEED (found with that overlap check)
  Although every record is labelled "m³/s", the portal published values in
  LITRES per second, as integers ("16353"), until 2026-05-13, when it switched
  to decimal m³/s ("16.031") — and then went back to integer L/s for a second
  block (2026-05-28 .. 2026-06-15). Correlation with the export is 1.000 and
  the volume ratio exactly 1000. Genuine m³/s values can also be integers
  after the switch (the Ter at "7" m³/s), so the rule is, per gauge and day:
    an integer-formatted value is L/s  if the day is before
    `opendata_litres_before`, or if most of that day's values are integers
    (block publishing); isolated integers among decimals stay m³/s.
  Checks: the QC table must show volume ratios of ~1.0 (the script fails if
  not), and remaining day-to-day jumps of ~1000x are reported (spike check).

HOW A "GAUGE" IS IDENTIFIED
  By the leading code of the ACA series description, e.g. "EA010_Girona_Cabal
  riu Ter" -> EA010. The open-data `codi_estacio` field is NOT used as the key
  because it sometimes groups different rivers under one station (e.g. station
  EA047 also carries the EA035 Mogent series). Combined series
  ("EA085_C4085_EA120 ... Ter+Gurri") and the historical duplicate EA072a are
  skipped: they would double count flows that another gauge already measures.

Writes
  data/processed/stations.csv             one row per gauge: id, name, river, basin, lon/lat, record lengths
  data/processed/q_obs_daily.parquet      daily mean discharge, m3/s (index=date, columns=gauge id)
  outputs/tables/qc_export_vs_opendata.csv agreement of the two sources over their overlap
Caches
  data/raw/aca_opendata/<gauge>/<YYYY-MM>.parquet   daily aggregates of the 5-min data

Usage
  python codes/01_download_aca_gauges.py               # full build
  python codes/01_download_aca_gauges.py --recent 10   # operational: refresh last 10 days only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pyproj import Transformer

from hydrocat.config import RAW, ROOT, TABLES, P, load_settings

S = load_settings()
G = S["gauges"]
CACHE = RAW / "aca_opendata"
SKIP_SERIES = ("EA085_C4085_EA120", "EA072a")
PAGE = 50000


def log(*a):
    print(*a, flush=True)


def norm(s: str) -> str:
    """Normalise a series description for matching (accents, '_', case, '_BDX')."""
    s = re.sub(r"_BDX$", "", s.strip())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()


def code_of(variable: str) -> str:
    return variable.split("_")[0].strip()


def soql(params: dict, timeout=180, tries=6) -> list[dict]:
    url = f"{G['socrata_url']}?{urllib.parse.urlencode(params)}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # network hiccup / throttling
            if i == tries - 1:
                raise
            log(f"    retry {i + 1}: {e}")
            time.sleep(5 * (i + 1))


# ---------------------------------------------------------------------------
# 1. Station catalogue from the open-data API (names, coordinates, series)
# ---------------------------------------------------------------------------
def opendata_catalogue() -> pd.DataFrame:
    """One row per discharge series seen in the last days of the open-data feed."""
    day = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    rows = soql({
        "$select": "codi_estacio,estacio,conca,subconca,utm_x,utm_y,descripcio_variable",
        "$where": f"dia='{day}T00:00:00.000' AND tipus_variable='Cabal riu' AND hora='12:00'",
        "$limit": 5000,
    })
    cat = pd.DataFrame(rows).drop_duplicates("descripcio_variable")
    cat["variable"] = cat["descripcio_variable"].str.replace(r"_BDX$", "", regex=True).str.strip()
    cat["gauge_id"] = cat["variable"].map(code_of)
    return cat[~cat["variable"].str.startswith(SKIP_SERIES)]


# ---------------------------------------------------------------------------
# 2. Historical export
# ---------------------------------------------------------------------------
def read_export() -> pd.DataFrame:
    files = sorted((ROOT / G["export_dir"]).glob("Q_*.csv.gz"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df["gauge_id"] = df["variable"].map(code_of)
    return df[~df["variable"].str.startswith(SKIP_SERIES)]


def match_series(export: pd.DataFrame, cat: pd.DataFrame) -> dict[str, dict]:
    """gauge_id -> {'export': [variables], 'opendata': variable|None, meta...}."""
    out: dict[str, dict] = {}
    ex_vars = export.drop_duplicates("variable")[["variable", "gauge_id", "station", "basin", "X", "Y"]]
    for gid, grp in ex_vars.groupby("gauge_id"):
        out[gid] = dict(export=list(grp.variable), opendata=None, name=grp.station.iloc[0],
                        basin=grp.basin.iloc[0], X=grp.X.iloc[0], Y=grp.Y.iloc[0])
    for _, r in cat.iterrows():
        g = out.setdefault(r.gauge_id, dict(export=[], opendata=None, name=r.estacio, basin=r.conca,
                                            X=float(r.utm_x), Y=float(r.utm_y)))
        g["opendata"] = r["descripcio_variable"]
        # Station name/coordinates only when the open-data station IS this gauge:
        # station EA047 (Besos) also carries the EA035 (Mogent) series, whose
        # own location must then come from the export.
        if str(r.codi_estacio).split("_")[0] == r.gauge_id or not g["export"]:
            g["name"], g["basin"] = r.estacio, r.conca           # UTF-8, nicer names
            g["X"], g["Y"] = float(r.utm_x), float(r.utm_y)
        # keep only the export series equal to the live one (e.g. EA066 "total
        # Llobregat + CIB" vs "riu Llobregat"); if none is equal but there is
        # exactly one export series, it is the same gauge under an older name.
        same = [v for v in g["export"] if norm(v) == norm(r["variable"])]
        if same or len(g["export"]) > 1:
            g["export"] = same
    return out


# ---------------------------------------------------------------------------
# 3. Open-data 5-min -> daily, cached per gauge-month
# ---------------------------------------------------------------------------
def month_starts(start: dt.date, end: dt.date):
    m = dt.date(start.year, start.month, 1)
    while m <= end:
        yield m
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)


def fetch_month(gid: str, variable: str, m0: dt.date, force: bool) -> pd.DataFrame:
    path = CACHE / gid / f"{m0:%Y-%m}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    m1 = dt.date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1) - dt.timedelta(days=1)
    rows, off = [], 0
    while True:
        page = soql({
            "$select": "dia,hora,valor",
            "$where": (f"descripcio_variable='{variable}' AND "
                       f"dia between '{m0}T00:00:00' and '{m1}T00:00:00'"),
            "$order": "dia,hora", "$limit": PAGE, "$offset": off,
        })
        rows += page
        if len(page) < PAGE:
            break
        off += PAGE
    if rows:
        d = pd.DataFrame(rows)
        d["date"] = pd.to_datetime(d["dia"].str[:10])
        is_int = ~d["valor"].str.contains(".", regex=False)
        int_share = is_int.groupby(d["date"]).transform("mean")
        legacy = is_int & ((d["date"] < pd.Timestamp(G["opendata_litres_before"])) | (int_share > 0.5))
        d["valor"] = pd.to_numeric(d["valor"], errors="coerce").astype(float)
        d.loc[legacy, "valor"] /= 1000.0          # L/s -> m3/s (see module docstring)
        daily = d.groupby("date")["valor"].agg(q="mean", n="count").reset_index()
    else:
        daily = pd.DataFrame({"date": pd.to_datetime([]), "q": [], "n": []})
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(path, index=False)
    return daily


def download_opendata(series: dict, start: dt.date, end: dt.date, refresh_from: dt.date) -> pd.DataFrame:
    jobs = [(gid, s["opendata"], m) for gid, s in series.items() if s["opendata"]
            for m in month_starts(start, end)]
    log(f"  open data: {len(jobs)} gauge-months "
        f"({sum(1 for *_, m in jobs if not (CACHE / _[0] / f'{m:%Y-%m}.parquet').exists())} not cached)")
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_month, gid, var, m, m >= dt.date(refresh_from.year, refresh_from.month, 1)): gid
                for gid, var, m in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            d = f.result()
            if len(d):
                out.append(d.assign(gauge_id=futs[f]))
            if i % 200 == 0:
                log(f"    {i}/{len(jobs)}")
    d = pd.concat(out, ignore_index=True)
    # 5-min data -> 288 values/day; drop days with poor coverage
    d = d[d["n"] >= G["min_daily_coverage"] * 288]
    return d.pivot_table(index="date", columns="gauge_id", values="q")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=None,
                    help="operational mode: only re-download the last N days and update the parquet")
    args = ap.parse_args()
    today = dt.date.today()

    log("1. open-data station catalogue")
    cat = opendata_catalogue()
    log(f"   {len(cat)} live discharge series")

    log("2. historical export")
    export = read_export()
    series = match_series(export, cat)
    if not S["gauges"]["include_reservoir_outflows"]:
        series = {g: s for g, s in series.items() if g.startswith("EA")}
    log(f"   {len(series)} river gauges ({sum(bool(s['opendata']) for s in series.values())} live)")

    ex_q = (export[export.variable.isin([v for s in series.values() for v in s["export"]])]
            .groupby(["date", "gauge_id"])["value"].mean().unstack())

    log("3. open-data 5-min -> daily")
    start = dt.date.fromisoformat(S["gauges"]["socrata_download_start"])
    refresh_from = today - dt.timedelta(days=args.recent or 35)
    od_q = download_opendata(series, start, today, refresh_from)

    log("4. QC: export vs open data over their overlap")
    qc = []
    for gid in sorted(set(ex_q.columns) & set(od_q.columns)):
        a, b = ex_q[gid].align(od_q[gid], join="inner")
        m = a.notna() & b.notna()
        if m.sum() > 30:
            qc.append(dict(gauge_id=gid, n_days=int(m.sum()), corr=float(np.corrcoef(a[m], b[m])[0, 1]),
                           ratio_mean=float(b[m].mean() / a[m].mean()) if a[m].mean() > 0 else np.nan,
                           median_abs_diff=float((a[m] - b[m]).abs().median())))
    qc = pd.DataFrame(qc)
    qc.to_csv(TABLES / "qc_export_vs_opendata.csv", index=False)
    if len(qc):
        log(f"   {len(qc)} gauges overlap: median corr {qc['corr'].median():.3f}, "
            f"median volume ratio {qc['ratio_mean'].median():.3f}")
        bad = qc[(qc["corr"] < 0.9) | ((qc["ratio_mean"] - 1).abs() > 0.1)]
        if len(bad):
            log("   NOTE: sources disagree for:\n" + bad.round(3).to_string(index=False))
        if not 0.95 < qc["ratio_mean"].median() < 1.05:
            raise RuntimeError("open data and export differ systematically in volume — unit problem?")

    # Isolated one-day spikes: a daily mean >300x (or <1/300x) BOTH neighbouring
    # days cannot be hydrological (a real flood's next day is still high on the
    # recession). They are sensor glitches or unit residue (e.g. EA016 on
    # 2026-07-19: ~1 h of 2388 m3/s readings during 1.8 m3/s baseflow, quality
    # flag NQ_REV0 = not yet revised). Such days are removed and listed.
    log("   spike check (isolated days >300x both neighbours) — removed:")
    lr = np.log10(od_q.where(od_q > 0))
    jump = ((lr - lr.shift(1)).abs() > 2.5) & ((lr - lr.shift(-1)).abs() > 2.5)
    spk = jump.stack()[jump.stack()]
    log("   none" if spk.empty else spk.index.to_frame(index=False).to_string(index=False))
    od_q = od_q.mask(jump)

    log("5. merge and write")
    cut = pd.Timestamp(S["gauges"]["socrata_from"])
    q = pd.concat([ex_q[ex_q.index < cut], od_q[od_q.index >= cut]]).sort_index()
    q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D"))
    q.index.name = "date"
    q = q.clip(lower=0)

    # Flat runs: a daily mean of 288 five-minute readings practically never
    # repeats exactly on consecutive days, so runs of identical non-zero values
    # are infilled/placeholder data (the export has runs of up to 820 days,
    # e.g. EA099). Runs of >= `max_flat_run_days` are removed; zero runs are
    # kept (dry ephemeral rivers are real).
    n0 = int(q.notna().sum().sum())
    nmax = G["max_flat_run_days"]
    for c in q.columns:
        s = q[c]
        run_id = (s.ne(s.shift()) | s.isna()).cumsum()
        run_len = s.groupby(run_id).transform("size")
        flat = (run_len >= nmax) & (s > 0) & s.notna()
        q.loc[flat, c] = np.nan
    log(f"   flat-run QC: removed {n0 - int(q.notna().sum().sum())} gauge-days "
        f"(runs of >= {nmax} identical non-zero daily values)")
    q.to_parquet(P.q_daily)

    to_ll = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
    rows = []
    cal0, cal1 = S["periods"]["calib_start"], S["periods"]["calib_end"]
    tst0 = S["periods"]["test_start"]
    for gid, s in sorted(series.items()):
        if gid not in q.columns:
            continue
        lon, lat = to_ll.transform(s["X"], s["Y"])
        col = q[gid]
        river = (s["opendata"] or (s["export"] or [""])[0]).split("_Cabal")[-1]
        river = re.sub(r"_BDX$", "", river).replace("_", " ").strip()
        river = re.sub(r"^(total )?(riu|riera)?\s*", "", river).strip() or river
        rows.append(dict(
            gauge_id=gid, name=re.sub(r"^Aforament - ", "", str(s["name"])).strip(),
            river=river, basin=str(s["basin"]).title(), lon=round(lon, 6), lat=round(lat, 6),
            utm_x=s["X"], utm_y=s["Y"], live=bool(s["opendata"]),
            series_opendata=s["opendata"] or "", series_export="|".join(s["export"]),
            first_obs=col.first_valid_index().date() if col.notna().any() else None,
            last_obs=col.last_valid_index().date() if col.notna().any() else None,
            n_days_calib=int(col[cal0:cal1].notna().sum()),
            n_days_test=int(col[tst0:].notna().sum()),
            mean_q_m3s=round(float(col.mean()), 4)))
    st = pd.DataFrame(rows)
    # Gauges whose two sources disagree over the overlap cannot be scored
    # reliably (the test period mixes both), so they are not modelled.
    bad_src = set(qc.loc[(qc["corr"] < 0.9) | ((qc["ratio_mean"] - 1).abs() > 0.25), "gauge_id"]) if len(qc) else set()
    st["sources_consistent"] = ~st.gauge_id.isin(bad_src)
    st["modelled"] = ((st.n_days_calib >= G["min_days_calib"]) & (st.n_days_test >= G["min_days_test"])
                      & st.sources_consistent)
    st.to_csv(P.stations, index=False)
    log(f"   {len(st)} gauges written, {int(st.modelled.sum())} meet the record-length criteria")
    log(f"   -> {P.q_daily.relative_to(ROOT)} ({q.index.min().date()} .. {q.index.max().date()})")
    log(f"   -> {P.stations.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
