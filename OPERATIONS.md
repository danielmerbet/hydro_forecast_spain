# Daily operation — GitHub Actions + GitHub Pages

Every day at 02:17 UTC (04:17 in Spain in summer, 03:17 in winter) the workflow `.github/workflows/daily.yml` runs
`codes/run_daily.sh` on a GitHub runner and commits the rebuilt website in
`docs/`, which GitHub Pages serves.

## What runs

| Step | Script | Source | Time |
|---|---|---|---|
| Observations | `01_download_aca_gauges.py --recent 10` | ACA open data | ~1 min (cached months) |
| Observations (Spain) | `01c_download_saih.py --basin <b> --recent 10` | SAIH Ebro, Júcar, Guadalquivir, Segura | ~1 min |
| Reanalysis | `03_extract_era5land_forcing.py --domain <d> --start 2003-01-01 --drop-grids` | ERA5-Land, Earth Engine | first run ~5 min per domain, then < 1 min (monthly means cached) |
| WeatherNext 3 (optional) | `10_fetch_weathernext3.py --domain <d>` | Earth Engine (approved access) | < 1 min |
| AIFS | `11_fetch_aifs.py --domain <d>` | ECMWF open data (CC BY 4.0) | ~5-15 min, ~1 GB download once for both domains, deleted after |
| GR4J | `12_run_gr4j_forecast.py --domain <d>` | bridge days + state updating | ~10 s Catalonia, a few min Spain |
| Google model | `13_run_google_forecast.py --domain <d>` | Catalonia released + fine-tuned, Spain released; CPU | ~3 min Catalonia, longer for Spain |
| Website | `14_build_site.py` | → `docs/` (both domains) | seconds |

A failure in any Spain step only drops Spain from that day's site (yellow
warning in the workflow); Catalonia failures stop the run.

Why 02:17 UTC (changed from 13:30 UTC on 2026-10-02): the forecast is online
before the working day. At that hour the newest complete runs are the previous
day's **12 UTC** runs (AIFS 12 UTC is published by ~19 UTC; WeatherNext 3 runs
are complete in Earth Engine ~7–13 h after initialisation), so the forecast
covers 14 full days starting today (a 00 UTC run would give 15 days but is
only ready after ~07 UTC for AIFS and ~13 UTC for WeatherNext 3). GitHub does
not guarantee the start time of scheduled runs: at 13:30 UTC they started
~5 h late; the minute :17 avoids the busiest full hour.

## One-time setup

1. **Create the GitHub repository** and push this folder (the `.gitignore`
   already excludes large and regenerable data; the commit is ~28 MB).
2. **Earth Engine service account.** In the Google Cloud project that has
   Earth Engine and WeatherNext access (`ee_project` in
   `config/settings.yaml`):
   * create (or reuse) a service account and a JSON key;
   * register the service account for Earth Engine
     (<https://signup.earthengine.google.com/#!/service_accounts> or the
     Cloud console → Earth Engine → Service accounts);
   * make sure the **WeatherNext data access** also covers the service
     account (the approval was granted to a user account; if the service
     account cannot read `projects/gcp-public-data-weathernext/assets/…`,
     request access for it or share the approval).
   * Test locally:
     `EE_SERVICE_ACCOUNT_KEY="$(cat key.json)" python codes/10_fetch_weathernext3.py`
3. **Repository secret**: Settings → Secrets and variables → Actions →
   *New repository secret* `EE_SERVICE_ACCOUNT_KEY` = the full JSON text of the key.
4. **GitHub Pages**: Settings → Pages → *Deploy from a branch* → `main` /
   `docs`.
5. **First run**: Actions → *daily-forecast* → *Run workflow*. The first run
   has no caches and takes longer (ACA open data since 2023-10 and ERA5-Land
   since 2003 for both domains are downloaded once; afterwards only their
   monthly catchment means, ~0.6 GB, are kept by `actions/cache`).

## If something fails

* The workflow page shows the log of each step; `run_daily.sh` stops at the
  first error (`set -e`), so yesterday's site stays online.
* WeatherNext 3 is **optional**: if the service account cannot read it (e.g.
  before the allowlisting is approved), no complete run exists, or Earth
  Engine fails, the run continues with AIFS only. The workflow shows a yellow
  warning, the page header says "WeatherNext 3 not available today: AIFS only",
  and the bridge days come from AIFS (past runs from ECMWF's Google Cloud
  mirror). To switch WeatherNext 3 off deliberately set `HYDROCAT_SKIP_WN3=1`.
* No AIFS run → the step fails with an explicit message (AIFS is required).
* ACA open-data unit changes are caught by script 01 (volume-ratio check).
* Scheduled workflows are paused by GitHub after 60 days without repository
  activity; the daily commit of `docs/` counts as activity.

## Costs

GitHub Actions minutes are free for public repositories. Earth Engine is free
for non-commercial use. ECMWF open data is free (CC BY 4.0, attribute ECMWF).
The WeatherNext 3 64-member Zarr in `gs://weathernext3_spatial` is
requester-pays (needs a billing account) and is **not** used.
