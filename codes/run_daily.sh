#!/usr/bin/env bash
# =============================================================================
# run_daily.sh — the daily operational chain (used by .github/workflows/daily.yml,
# and runnable by hand from the repository root with the `hydrocat` env active).
#
#   01   ACA observations (open data; cached months are reused)
#   01c  SAIH Ebro, Júcar, Guadalquivir, Segura: last 10 days (GR4J state updating)
#   03   ERA5-Land forcing 2003 -> today-7d, both domains (monthly catchment
#        means cached; grids deleted once averaged)
#   10   WeatherNext 3, latest complete 00 UTC run (Earth Engine) — OPTIONAL:
#        any failure (no access yet: exit 3, no complete run, EE error) -> AIFS only
#   11   ECMWF AIFS 00 UTC run (open data) + 50-member ensemble rain
#        (global files downloaded once, cropped for each domain)
#   12   GR4J forecasts (bridge days, state updating, all forcings)
#   13   Google model forecasts (Catalonia: released + fine-tuned; Spain: released)
#   14   website in docs/ (+ 30 days of run archives)
#
# Catalonia is the reference domain: any failure there stops the run (the site
# of the day before stays online). A failure in a Spain step only drops Spain
# (or its Google forecast) from today's site, with a warning in the workflow.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

warn() { echo "::warning::$*"; }

python codes/01_download_aca_gauges.py --recent 10
for b in ebro jucar guadalquivir segura; do
  python codes/01c_download_saih.py --basin "$b" --recent 10 || warn "SAIH $b not available today"
done

python codes/03_extract_era5land_forcing.py --domain catalonia --start 2003-01-01 --drop-grids
python codes/10_fetch_weathernext3.py --domain catalonia \
  || warn "WeatherNext 3 not available today - publishing AIFS-based forecasts only"
python codes/11_fetch_aifs.py --domain catalonia
python codes/12_run_gr4j_forecast.py --domain catalonia     # may fetch AIFS day 0 of past runs (bridge)
python codes/13_run_google_forecast.py --domain catalonia

# inside a function called from `if`, set -e is off: every step checks itself
spain() {
  python codes/03_extract_era5land_forcing.py --domain spain --start 2003-01-01 --drop-grids || return 1
  python codes/10_fetch_weathernext3.py --domain spain || warn "WeatherNext 3 (Spain) not available today"
  python codes/11_fetch_aifs.py --domain spain || return 1
  python codes/12_run_gr4j_forecast.py --domain spain || return 1
  python codes/13_run_google_forecast.py --domain spain || warn "Google model (Spain) failed today"
}
if ! spain; then
  warn "Spain forecast failed today - publishing Catalonia only"
  rm -rf data/forecasts/spain/gr4j                         # keep an old Spain run off today's site
fi
rm -rf data/raw/aifs                                      # ~1 GB of global GRIB per run

python codes/14_build_site.py

# keep 30 days of run archives in the published site
ls -1 docs/data/archive/*.json 2>/dev/null | sort | head -n -30 | xargs -r rm -f
