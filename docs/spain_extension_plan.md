# Extending the pipeline to all of Spain (decision of 2026-09-29)

Scope chosen: **every Spanish gauge with usable data, including live data from
each basin authority's SAIH**, so that GR4J and Google's model can be compared
(and later forecast) nationwide, not only in Catalonia.

## The data constraint that shapes the design

| Need | Catalonia (ACA) | Rest of Spain |
|---|---|---|
| Long daily record for calibration | ACA export 2007 → 2024 | CEDEX *Anuario de Aforos* (SAIH-ROEA), from the start of each station → **hydrological year 2021-22** (latest edition, published April 2026) |
| Data after 2023-10-01 (Google's leakage-free test window) and live data | open-data API, 5-min | only in each Confederación's SAIH, 9 different systems |

So each gauge outside Catalonia needs **two sources joined**, exactly like the
ACA export + open data, with an overlap check where they meet.

## Basin authorities and their SAIH systems — status (2026-09-29)

| Demarcación | CEDEX gauges (with SAIH code) | System | Status |
|---|---|---|---|
| Ebro | 268 (223) | SAIH Ebro, public JSON/ZIP API of its "Datos históricos" page | **done** — 207 gauges, daily, 2021-10 → today; CEDEX overlap median r = 1.000, 10 gauges flagged |
| Júcar | 60 (45) | SAIH Júcar, gauge list embedded in the map page + 5-min values endpoint used by its public charts | **done** — 34 gauges, 2024-07-26 → today (the system keeps ~2 years) |
| Guadalquivir | 104 (42) | SAIH Guadalquivir, public ASP.NET "Datos Históricos" form (daily mean) | **done** — 34 gauges, 2021-10 → today; overlap median r = 0.988, 12 flagged (regulated) |
| Miño-Sil | 66 (56) | saih.chminosil.es, 15-min tables per week (session cookie; FNMT intermediate certificate) | **live only** — public history is ~3 weeks, so no backfill; to be archived daily by the operational run |
| Segura | 55 (47) | SAIH Segura iVisor, daily means from Oct 1996 (graficaVar.php), reliable values only | **done** — terms of use accepted 2026-09-30 (cite CHS; provisional data); 42 gauges, 2021-10 → today; overlap median r = 0.972, 5 flagged |
| Cantábrico | 73 (68) | visor.saichcantabrico.es (WordPress) | **not automated** — the download script is deliberately obfuscated; request the data formally from CHC |
| Tajo | 182 (7) | saihtajo.chtajo.es (single-page app) | **pending** — data URLs are per-session encrypted tokens; fragile, low priority |
| Duero | 205 (0) | saihduero.es | **pending** — CEDEX has no SAIH codes: stations must be matched by name/location |
| Guadiana | 111 (81) | SIRA Guadiana (siraguadiana.com) | **pending** |
| Galicia-Costa | 50 | Augas de Galicia (regional) | not started |
| Internal basins | — | ACA (Catalonia) **done**; URA (Basque Country), Junta de Andalucía | not started |

Rules followed for every source: use only what the public web pages themselves
request; keep request rates low (≤ 4 parallel, cached); verify TLS (missing
intermediates are shipped in `config/certs/`, never `--insecure`); do not work
around obfuscation, logins or terms-of-use forms without the user's decision.

For every SAIH the downloader must document: URL/API, variables and units,
time step, time zone, how far back it goes, quality flags, and the overlap
agreement with CEDEX. The ACA open-data feed turned out to publish L/s under
an m³/s label for part of its record; the same overlap/unit/spike checks will
be applied to every source.

## Pipeline changes

* `01_download_aca_gauges.py` stays; new `01b_download_cedex_anuario.py`
  (all ROEA stations, daily flows, station metadata incl. catchment area) and
  `01c_download_saih_<basin>.py` (one per SAIH), all writing the same
  canonical tables (`stations.csv`, `q_obs_daily.parquet`) with a `source`
  and `authority` column.
* `02` runs over a Spain-wide flow-direction grid (tiled). Candidate: MERIT
  Hydro 3" (`MERIT/Hydro/v1_0_1`), which is better than HydroSHEDS in flat
  terrain; CEDEX publishes the drained area of every station, so the
  area-matched snapping and the ±20 % check carry over unchanged.
* `03` / `04` scale as they are (Earth Engine, cached per month); only the
  number of catchments grows (from 85 to ~1,000+), so requests are chunked by
  catchment groups.
* Regulation flags need a national reservoir inventory (e.g. the CEDEX
  *embalses* table or GRanD) instead of ACA's E-stations.
* Calibration 2008-10 → 2022-09 (end of CEDEX) for gauges outside Catalonia;
  test 2023-10 → today wherever a SAIH series exists.

## New figure 2

`02_catchments_map.png` becomes a map of **all Spanish gauges with data**,
coloured by basin authority, with modelled catchments outlined and the
Catalan (ACA) subset highlighted.
