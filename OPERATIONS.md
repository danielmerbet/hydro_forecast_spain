# Daily operation — GitHub Actions + GitHub Pages

Every day at 13:30 UTC the workflow `.github/workflows/daily.yml` runs
`codes/run_daily.sh` on a GitHub runner and commits the rebuilt website in
`docs/`, which GitHub Pages serves.

## What runs

| Step | Script | Source | Time |
|---|---|---|---|
| Observations | `01_download_aca_gauges.py --recent 10` | ACA open data | ~1 min (cached months) |
| Reanalysis | `03_extract_era5land_forcing.py --start 2003-01-01` | ERA5-Land, Earth Engine | ~5 min (grid cache) |
| WeatherNext 3 (optional) | `10_fetch_weathernext3.py` | Earth Engine (approved access) | ~1 min |
| AIFS | `11_fetch_aifs.py` | ECMWF open data (CC BY 4.0) | ~5 min, ~1 GB download, deleted after |
| GR4J | `12_run_gr4j_forecast.py` | bridge days + state updating | ~3 min |
| Google model | `13_run_google_forecast.py` | released + fine-tuned weights, CPU | ~3 min |
| Website | `14_build_site.py` | → `docs/` | seconds |

Why 13:30 UTC: WeatherNext 3 00 UTC runs are complete in Earth Engine
~7–13 h after initialisation; AIFS 00 UTC is available from ~07 UTC.

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
   since 2003 are downloaded once, then cached by `actions/cache`).

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
