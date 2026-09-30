#!/usr/bin/env python3
"""
01d — One observed-discharge table for Spain (CEDEX history + SAIH recent data).

JOIN RULE (per gauge)
  * up to 2022-09-30: CEDEX Anuario (script 01b) — official, quality-checked;
  * from 2022-10-01: the basin authority's SAIH series (script 01c), used only if
      - the two sources agree over their overlap (2021-10 .. 2022-09):
        correlation >= 0.9 and volume ratio within +-25 % -> join "verified";
      - or there is no overlap to test (SAIH Júcar keeps only ~2 years)
        -> join "unverified" (kept, but reported separately);
    gauges whose sources disagree keep CEDEX only (join "rejected").
QUALITY CONTROL
  * runs of identical non-zero daily values -> removed: >= 90 days in the CEDEX
    period (values rounded to 2-3 decimals, already checked by CEDEX, so short
    repeats at low flow are genuine), >= 5 days in the SAIH period (as for ACA);
  * isolated one-day spikes > 300x both neighbours -> removed.
RECORD-LENGTH FLAGS
  calibratable: >= min_days_calib valid days in 2008-10-01 .. 2022-09-30
  testable:     >= min_days_test valid days from 2023-10-01 (SAIH gauges only)

Writes
  data/processed/spain/q_obs_daily.parquet     m3/s, date x gauge
  data/processed/spain/stations.csv            CEDEX table + join status + flags
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hydrocat.config import PROCESSED, TABLES, load_settings

S = load_settings()
G = S["gauges"]
OUT = PROCESSED / "spain"
CUT = pd.Timestamp("2022-10-01")
CAL0, CAL1 = pd.Timestamp(S["periods"]["calib_start"]), pd.Timestamp("2022-09-30")
TST0 = pd.Timestamp(S["periods"]["test_start"])


def log(*a):
    print(*a, flush=True)


def qc(q: pd.DataFrame) -> pd.DataFrame:
    """Flat runs: CEDEX period (rounded, officially checked values) only runs of
    >= max_flat_run_days_cedex; SAIH period (means of 5-15 min data) runs of
    >= max_flat_run_days, as for ACA."""
    n0 = int(q.notna().sum().sum())
    saih = q.index >= CUT
    for c in q.columns:
        s = q[c]
        run_id = (s.ne(s.shift()) | s.isna()).cumsum()
        run_len = s.groupby(run_id).transform("size")
        lim = np.where(saih, G["max_flat_run_days"], G["max_flat_run_days_cedex"])
        q.loc[(run_len >= lim) & (s > 0), c] = np.nan
    n1 = int(q.notna().sum().sum())
    lr = np.log10(q.where(q > 0))
    jump = ((lr - lr.shift(1)).abs() > 2.5) & ((lr - lr.shift(-1)).abs() > 2.5)
    q = q.mask(jump)
    log(f"   QC: flat runs removed {n0 - n1} gauge-days, one-day spikes removed {int(jump.sum().sum())}")
    return q


def main():
    st = pd.read_csv(OUT / "stations_cedex.csv", dtype={"cod_saih": str})
    cedex = pd.read_parquet(OUT / "q_obs_daily_cedex.parquet")
    q = cedex.copy()
    st["saih_join"] = "none"
    for f in sorted(OUT.glob("q_obs_daily_saih_*.parquet")):
        basin = f.stem.replace("q_obs_daily_saih_", "")
        saih = pd.read_parquet(f)
        qcf = TABLES / f"qc_cedex_vs_saih_{basin}.csv"
        qct = pd.read_csv(qcf).set_index("gauge_id") if qcf.exists() and qcf.stat().st_size > 5 else pd.DataFrame()
        for g in saih.columns:
            if g not in st.gauge_id.values:
                continue
            if g in qct.index:
                r = qct.loc[g]
                ok = (r["corr"] >= 0.9) and abs(r["ratio_mean"] - 1) <= 0.25
                status = "verified" if ok else "rejected"
            else:
                status = "unverified"
            st.loc[st.gauge_id == g, "saih_join"] = status
            if status == "rejected":
                continue
            s = saih[g][saih.index >= CUT]
            q = q.reindex(q.index.union(s.index))
            q.loc[s.index, g] = s.values
        log(f"   SAIH {basin}: {saih.shape[1]} gauges")
    q = q.sort_index()
    q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D"))
    q.index.name = "date"
    q = qc(q.clip(lower=0))
    q.to_parquet(OUT / "q_obs_daily.parquet")

    st["n_days_calib"] = st.gauge_id.map(q[CAL0:CAL1].notna().sum()).fillna(0).astype(int)
    st["n_days_test"] = st.gauge_id.map(q[TST0:].notna().sum()).fillna(0).astype(int)
    st["calibratable"] = st.n_days_calib >= G["min_days_calib"]
    st["testable"] = st.n_days_test >= G["min_days_test"]
    st.to_csv(OUT / "stations.csv", index=False)
    log(f"-> {len(st)} gauges: {int(st.calibratable.sum())} calibratable (>= {G['min_days_calib']} days "
        f"2008-10..2022-09), {int((st.calibratable & st.testable).sum())} also testable after {TST0.date()}")
    log(st.groupby("demarcacion").agg(gauges=("gauge_id", "size"), calibratable=("calibratable", "sum"),
                                      testable=("testable", "sum")).to_string())
    log(st.saih_join.value_counts().to_string())


if __name__ == "__main__":
    main()
