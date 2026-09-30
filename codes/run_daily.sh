#!/usr/bin/env bash
# =============================================================================
# run_daily.sh — the daily operational chain (used by .github/workflows/daily.yml,
# and runnable by hand from the repository root with the `hydrocat` env active).
#
#   01  ACA observations (open data; cached months are reused)
#   03  ERA5-Land forcing 2003 -> today-7d (grid cache reused when available)
#   10  WeatherNext 3, latest complete 00 UTC run (Earth Engine)
#   11  ECMWF AIFS 00 UTC run (open data) + 50-member ensemble rain
#   12  GR4J forecasts (bridge days, state updating, all forcings)
#   13  Google model forecasts (released + fine-tuned)
#   14  website in docs/ (+ 30 days of run archives)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

python codes/01_download_aca_gauges.py --recent 10
python codes/03_extract_era5land_forcing.py --start 2003-01-01
python codes/10_fetch_weathernext3.py
python codes/11_fetch_aifs.py
rm -rf data/raw/aifs                                  # ~1 GB of global GRIB per run
python codes/12_run_gr4j_forecast.py
python codes/13_run_google_forecast.py
python codes/14_build_site.py

# keep 30 days of run archives in the published site
ls -1 docs/data/archive/*.json 2>/dev/null | sort | head -n -30 | xargs -r rm -f
