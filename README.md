# River forecasts for Spain — GR4J vs Google's hydrology model

Daily streamflow simulation and (next phase) forecasting at the river gauges of
Catalonia's internal basins (*Conques Internes*, operated by the Agència
Catalana de l'Aigua, ACA), with two hydrological models side by side:

| | **GR4J + CemaNeige** | **Google hydrology model** |
|---|---|---|
| Type | conceptual, 6 parameters | deep learning (LSTM, "mean embedding forecast LSTM") |
| Where it learns | calibrated **per gauge** on ACA observations 2008–2023 | trained **globally** by Google on ~16,000 basins (Caravan/MultiMet), 1982–2023; used as released, not re-trained here |
| Code | `lib/hydrocat/gr4j.py` (port of INRAE's airGR, verified to 1e-6) | [google-research/flood-forecasting](https://github.com/google-research/flood-forecasting) (`googlehydrology` 1.12.0, pinned commit) |
| Output | one deterministic series | a probability distribution (CMAL); we keep median + 5–95 % |

Both models are driven by **exactly the same weather** (ERA5-Land, averaged over
each catchment), so differences in skill come from the models, not their inputs.

The result is an interactive web map (in the style of Google's
[Flood Hub](https://sites.research.google/floods/)) on GitHub Pages, updated every
day with forecasts driven by **WeatherNext 3** (Google DeepMind) and **AIFS**
(ECMWF), for the 64 Catalan (ACA) gauges and 863 gauges in the rest of Spain
(CEDEX network). Phase 1 (calibration and model comparison, Catalonia) is in
sections 2–6, the daily forecast in section 7, the Spain-wide extension in 7b.

---

## 1. Quick start (reproduce everything)

```bash
git clone <this repo> && cd hydro_forecast_spain
conda env create -f environment.yml          # ~10 min, CPU-only PyTorch
conda activate hydrocat
bash codes/00_setup.sh                       # Google's model at a pinned commit + this package
earthengine authenticate                     # once; Google account with Earth Engine access

python codes/01_download_aca_gauges.py       # ~20 min  observed discharge, Catalonia (ACA)
python codes/01b_download_cedex_anuario.py   # ~15 min  historical discharge, rest of Spain (CEDEX, to 2022-09)
python codes/01c_download_saih.py --basin ebro   # ~5 min  recent/live discharge, one SAIH at a time
python codes/02_delineate_catchments.py      # ~5 min   catchments
python codes/02b_plot_spain_gauges.py        #          Figure 2: all Spanish gauges
python codes/03_extract_era5land_forcing.py  # ~1-2 h   ERA5-Land 1981→today (Earth Engine, cached per month)
python codes/03b_extract_emo1_precip.py      # ~30 min  EMO-1 gauge-based rain (diagnostic only)
python codes/04_catchment_attributes.py      # ~5 min   hypsometry, HydroATLAS, climate indices
python codes/05_calibrate_gr4j.py            # ~20 min  GR4J calibration (14 cores)
python codes/05b_plot_gr4j_calibration.py    # ~2 min   calibration figures
python codes/06_build_google_dataset.py      # ~1 min   inputs for Google's model
python codes/07_run_google_model.py --runs baseline   # ~1 h on CPU   Google model as released
conda activate hydrocat-gpu                  # CUDA build of PyTorch (see §1b)
python codes/07b_finetune_google_model.py    # ~1.5 h on a 4 GB GPU   Google model fine-tuned on ACA
python codes/08_compare_models.py            # ~2 min   metrics and figures
pytest                                       # model/PET correctness tests
```

Every script reads `config/settings.yaml` (periods, paths, options) and writes
only to `data/` and `outputs/`. Downloads are cached, so re-running a script is
cheap; delete the cache folder to force a fresh download.

Requirements: Linux/macOS, ~15 GB free disk, `curl`, a Google Cloud project
registered for Earth Engine (non-commercial use is free) set as `ee_project` in
`config/settings.yaml`. R + airGR are optional (only for `tests/test_gr4j_vs_airgr.py`).

### 1b. GPU environment (only for fine-tuning, script 07b)

```bash
conda create --name hydrocat-gpu --clone hydrocat
conda run -n hydrocat-gpu pip install --force-reinstall --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n hydrocat-gpu pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # CUDA libraries
```
(`pip install torch` alone keeps the CPU build installed by conda — hence the
forced reinstall.) Tested on an NVIDIA RTX A500 laptop GPU (4 GB, driver 580).

---

## 2. The pipeline, step by step

| # | Script | What it does | Main output |
|---|---|---|---|
| 00 | `00_setup.sh` | Clones Google's model repo at a pinned commit, installs it and `hydrocat` | `external/flood-forecasting/` |
| 01 | `01_download_aca_gauges.py` | Daily discharge at ACA gauges: historical export (2007–Oct 2024) + open-data API (5-min, 2020→today). Checks that both agree where they overlap | `data/processed/q_obs_daily.parquet`, `stations.csv` |
| 02 | `02_delineate_catchments.py` | Catchment of every gauge from HydroSHEDS 90 m flow directions; snapping checked against ACA's official drained areas | `data/processed/catchments.gpkg` |
| 03 | `03_extract_era5land_forcing.py` | ERA5-Land daily catchment means 1981→today (P, T, radiation, pressure, humidity, wind) and FAO-56 Penman-Monteith evapotranspiration | `data/processed/forcing_era5land.parquet` |
| 04 | `04_catchment_attributes.py` | Hypsometry (for snow), the 84 static attributes Google's model needs, regulation flags | `data/processed/catchment_attributes.csv` |
| 05 | `05_calibrate_gr4j.py` | Calibrates GR4J + CemaNeige at each gauge (mean of KGE(Q) and KGE(√Q), differential evolution, 5-year spin-up) and applies the zero-flow threshold at intermittent rivers | `data/processed/gr4j/` |
| 05b | `05b_plot_gr4j_calibration.py` | Calibration figures: KGE per gauge (calibration vs test), maps, parameters, per-gauge diagnostics | `outputs/figures/05b_*`, `gr4j/` |
| 06 | `06_build_google_dataset.py` | Writes the Caravan-MultiMet-format dataset for Google's model | `data/processed/google/` |
| 07 | `07_run_google_model.py` | Runs Google's pretrained weights as released (`--gauges` for a quick preview) | `data/processed/google/q_sim_*.parquet` |
| 07b | `07b_finetune_google_model.py` | Fine-tunes the pretrained model on ACA gauges (train 2008–2021, validate 2021–2023, test untouched) | `data/processed/google/q_sim_*_finetuned.parquet` |
| 01b | `01b_download_cedex_anuario.py` | Spain: CEDEX Anuario de Aforos, all ROEA/SAIH gauges, daily to 2022-09-30 | `data/processed/spain/` |
| 01c | `01c_download_saih.py` | Spain: recent/live data from each basin authority's SAIH, joined to CEDEX via `cod_saih` and checked on the overlap (Ebro done) | `data/processed/spain/` |
| 02b | `02b_plot_spain_gauges.py` | Figure 2: all Spanish gauges with data | `outputs/figures/02_gauges_spain.png` |
| 03b | `03b_extract_emo1_precip.py` | EMO-1 (JRC, gauge-based, 1.8 km) catchment rain — diagnostic of ERA5-Land's rain bias | `data/processed/forcing_emo1_pr.parquet` |
| 08 | `08_compare_models.py` | Metrics on the common test period, per-gauge plots, summary figures | `outputs/` |

Shared code lives in `lib/hydrocat/` (config & paths, GR4J/CemaNeige, PET,
metrics, Earth Engine helpers). Tests are in `tests/`.

---

## 3. Data sources

| Data | Source | Access |
|---|---|---|
| Discharge 2007-01-01 → 2024-10-20 (daily) | ACA *Consulta de dades del medi* export — shipped in `data/raw/aca_export/` (see its README) | in repo |
| Discharge 2020 → today (5-min) | Catalan open data, dataset [`3yr3-vq6y`](https://analisi.transparenciacatalunya.cat/Medi-Ambient/Cabals-dels-rius-a-les-conques-internes-de-Catalun/3yr3-vq6y) | public API, no key |
| Gauge metadata (official drained area) | ACA SDIM2 catalogue API | public |
| Flow directions | HydroSHEDS v1 3" (`WWF/HydroSHEDS/03DIR`) | Earth Engine |
| DEM (hypsometry) | SRTM 1" (`USGS/SRTMGL1_003`) | Earth Engine |
| Basin attributes | HydroATLAS / BasinATLAS level 12 (`WWF/HydroATLAS/v1/Basins/level12`) | Earth Engine |
| Meteorology 1981 → today−7 d | ERA5-Land daily (`ECMWF/ERA5_LAND/DAILY_AGGR`) + hourly wind (`ECMWF/ERA5_LAND/HOURLY`) | Earth Engine |
| Google model + weights | github.com/google-research/flood-forecasting | public |
| Weather forecasts (phase 2) | WeatherNext 3 (`projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg`), ECMWF AIFS / IFS open data | Earth Engine (approved access) / ECMWF open data |

---

## 4. Methodological choices (and why)

**Periods.** Warm-up (spin-up) 2003-10 → 2008-09, i.e. 5 years, never scored; GR4J calibration 2008-10-01 →
2023-09-30; **test period 2023-10-01 → today** for both models. The test
period starts exactly where Google's training data end: Google's README states
the released weights saw 1982–2023-09 and that scoring them inside that window
is data leakage. Google's training set also includes 269 Spanish basins
(CAMELS-ES); some may be Catalan, which is comparable to GR4J having been
calibrated at the same gauge — the post-2023 test period keeps both honest.

**Same forcing for both models.** Google's model is fed ERA5-Land through its
`union_mapping` (all forecast/satellite products left empty, so every input
falls back to ERA5-Land at the valid date): a "perfect weather" simulation,
the same information GR4J gets.

**Evapotranspiration.** GR4J uses FAO-56 Penman-Monteith reference ET computed
from ERA5-Land net radiation, temperature, dewpoint, pressure and hourly wind
speed (`hydrocat.pet`, tested against FAO-56 Example 18) — no temperature-only
formula. ERA5-Land's own `potential_evaporation` field is strongly biased high
(Caravan deprecated it); it is used **only** to compute Google's static
`pet_mean_ERA5_LAND`, because that is how the model's training data define it.

**Units checked against the model's scaler.** Google's inputs are
precipitation mm/day, temperature °C, pressure kPa, radiation as daily-mean
W/m², streamflow mm/day; script 06 prints our inputs next to the training
means/standard deviations as a check.

**Snow.** CemaNeige (5 elevation bands) is calibrated only in catchments
reaching above 1200 m; elsewhere it runs with fixed default parameters.

**GR4J objective and bounds.** Mean of KGE(Q) and KGE(√Q). Plain KGE(Q) gave
degenerate fits at a quarter of the gauges (routing store X3 → 1 mm, baseflow
→ 0, e.g. the Onyar); X3 is also bounded at ≥ 10 mm for the same reason.

**Zero flow at intermittent rivers.** GR4J's stores only empty
asymptotically, so it never simulates Q = 0. Observed zero = Q < 0.001 m³/s
(ACA's reporting resolution), counted only in spells the river visibly receded
into (flow before the spell < 10 % of the median) — zero blocks right after a
data gap are treated as missing. A **near-natural** gauge is intermittent if it
has ≥ 1 such day per year on average (non-perennial definition of Messager et
al., 2021). For those, a per-gauge threshold q₀ is chosen so that the simulated
zero-flow frequency in the calibration period equals the observed one, and
simulated Q < q₀ is set to 0; elsewhere only the 0.001 m³/s floor applies.
Zero-flow thresholds strongly affect intermittence metrics (Yu et al., 2024),
hence a data-based rule rather than an arbitrary number.

**Regulated rivers.** Catchments containing an ACA reservoir station (Sau,
Susqueda, La Baells, Boadella, …) are flagged. HydroATLAS's `dor_pc_pva` is not
used: headwater catchments sharing a level-12 polygon with a downstream dam
inherit its value. Neither model knows dam operations; results are reported
separately, and zero-flow thresholds are not applied there.

**Quality control.** Gauges are dropped from the comparison when the two
discharge sources disagree over their overlap (EA088, EA113) and flagged when
the delineated catchment differs from ACA's official area by more than 20 %.
Runs of ≥ 5 identical non-zero daily values (infilled data; up to 820 days at
EA099) and isolated one-day spikes > 300× both neighbours are removed.

**Google model, two arms.** (1) *As released*: the global weights, no local
training — how Flood Hub runs. (2) *Fine-tuned*: all layers trained further on
ACA gauges (train 2008-10 → 2021-09, validation 2021-10 → 2023-09 to pick the
epoch, learning rate 1e-4), starting from Google's "NSE > 0.5" run as their
README recommends. The fine-tuned arm is the fair counterpart of calibrating
GR4J per gauge.

---

## 5. Data issues found (and fixed) along the way

* **Open-data units.** Every record of dataset `3yr3-vq6y` is labelled m³/s,
  but values before **2026-05-13**, and again **2026-05-28 → 2026-06-15**, are in
  **litres per second** (integers). The overlap with ACA's own export shows
  correlation 1.000 and a volume ratio of exactly 1000. Script 01 converts them
  per gauge-day and fails if the ratio is not ~1.
* **Sensor glitch.** EA016 on 2026-07-19: ~1 h of 2388 m³/s readings during
  1.8 m³/s baseflow (unrevised data); removed by the one-day spike check.
* **Water balance.** At 14 of 64 gauges ERA5-Land precipitation minus observed
  runoff exceeds FAO-56 reference ET (e.g. Fluvià at Esponellà: P 1341, Q 138,
  ET0 861 mm/yr). GR4J then pushes its exchange term X2 to the bound; the
  EMO-1 comparison (script 03b) tests whether ERA5-Land over-estimates rain.
* **CEDEX server** resets Python's TLS handshake → downloads use `curl`.
* **SAIH Ebro server** omits its intermediate certificate (FNMT "AC
  Componentes Informáticos"); it is shipped in `config/certs/` so TLS is still
  verified. One SAIH station (A312) reports in l/s — units are read from each
  file header. 10 Ebro gauges disagree with CEDEX over the overlap (SAIH
  volume 5–64 % of CEDEX) and are flagged.
* **Station vs series.** Open-data station EA047 (Besòs) also carries the
  EA035 (Mogent) series; gauges are therefore keyed by series code, and
  locations come from the export when the two differ.
* **Export encoding.** ACA's export is CP850 (DOS), not Latin-1.
* **Google model input units** (relevant to earlier work in `hydraulic_run/`):
  radiation must be daily-mean **W/m²** (not MJ/m²/day) and
  `pet_mean_ERA5_LAND` must be ERA5-Land's own PE (not FAO ET0) — otherwise
  inputs sit far outside the training distribution.

---

## 6. Results (phase 1, Catalonia, test period Oct 2023 → Sep 2026)

Medians over gauges whose catchment area is within 20 % of ACA's (script 08,
`outputs/tables/metrics_summary.csv`), all models on the same ERA5-Land weather:

| | KGE | NSE | log NSE | volume bias | top 1 % flows |
|---|---|---|---|---|---|
| **Near-natural** — GR4J | **0.53** | 0.34 | −0.03 | +16 % | −23 % |
| Google, as released | −0.21 | −0.26 | −0.54 | +98 % | **−11 %** |
| Google, fine-tuned on ACA | 0.31 | 0.30 | **0.50** | **−9 %** | −59 % |
| GR4J, EMO-1-scaled rain (sensitivity) | 0.45 | **0.38** | 0.02 | +16 % | −21 % |
| **Regulated** — GR4J | 0.23 | −0.57 | −0.58 | +37 % | −39 % |
| Google, as released | −0.76 | −3.52 | −1.10 | +111 % | −26 % |
| Google, fine-tuned on ACA | **0.48** | **0.36** | **0.42** | **+11 %** | −43 % |

Fine-tuned Google beats GR4J on KGE at 36 of 64 gauges. Google as released gets
flood timing right but doubles the volume; fine-tuned it has the best low flows
and is clearly best below dams, but under-predicts the largest peaks. ERA5-Land
rain is +29 % wetter than gauge-based EMO-1 (script 03c), which explains
GR4J's groundwater-loss parameter hitting its bound. No model catches the
January 2026 Onyar flood (258 m³/s): daily ~9 km reanalysis rain cannot
resolve that storm. Full figures: the phase-1 report page built by script 09.

---

## 7. Phase 2 — daily forecasts and web page (implemented)

| # | Script | What it does |
|---|---|---|
| 10 | `10_fetch_weathernext3.py` | Latest complete 00 UTC WeatherNext 3 run from Earth Engine (0.1°, hourly, 15 days, ensemble mean), aggregated to daily catchment means with the same weights as ERA5-Land (same grid). Net radiation and pressure by FAO-56 (eqs. 7, 37–40; tested). The 64 members are only in a requester-pays bucket and are not used. |
| 11 | `11_fetch_aifs.py` | ECMWF AIFS 00 UTC open data: deterministic (all variables, 6-hourly) + 50-member AIFS-ENS rain at daily steps. Units read from each GRIB. |
| 12 | `12_run_gr4j_forecast.py` | GR4J from ERA5-Land, ERA5-Land's ~7-day lag **bridged** with day-0 fields of each day's run (WeatherNext 3, or AIFS from ECMWF's Google Cloud/AWS mirrors when WeatherNext 3 is not readable), **routing-store updating** to the last observed flow (GRP-style, Berthet et al. 2009), then 15-day forecasts: WN3, AIFS, AIFS-ENS (quantiles of 50 runs). |
| 13 | `13_run_google_forecast.py` | Google's model, 7 days (Catalonia: released and fine-tuned; Spain: released — the fine-tuned weights were trained on Catalan gauges only): AIFS in the HRES slots, WeatherNext 3 in the GraphCast slots (AIFS in both when WeatherNext 3 is unavailable: the model accepts a missing product only where ERA5-Land can replace it, never in the future), ERA5-Land + bridge for the past year. |
| 14 | `14_build_site.py` + `site_template.html` | Static site in `docs/` for all domains: map of gauges coloured by forecast flood level (2/5/20-year return levels of daily flow, Gumbel fit to observed annual maxima), hydrograph panel with every model (forecast days only), per-gauge skill, 30 days of archives. **Colour** = highest level reached by the *medians* of GR4J (WN3, AIFS, AIFS-ENS) and Google fine-tuned over the whole forecast; a **ring** = a higher level reached by the 95 % bound of GR4J with the 50 AIFS members. Google's released model is plotted but does not set the colour (test-period volume bias +98 % in Catalonia; its 95 % bound reached 640 m³/s at Santa Coloma where the 20-year level is 140). Grey = below the 2-year level; hollow = no levels (record < 8 years). |
| — | `run_daily.sh`, `.github/workflows/daily.yml` | The daily chain on GitHub Actions at 13:30 UTC, publishing to GitHub Pages. Setup and troubleshooting: **OPERATIONS.md**. |

Design notes from building it:
* WeatherNext's hourly p10/p90 were first summed into a "rain envelope": that
  gave 90th-percentile floods ~50× the ensemble mean, so it was dropped; the
  AIFS 50-member ensemble provides the uncertainty band.
* Without state updating, GR4J started forecasts after a dry summer from an
  empty routing store (≈0 m³/s where the Ter carried 3 m³/s).
* **WeatherNext 3 licence.** Forecast data (valid < 1 h ago and future) fall under
  the GDM Real-Time Weather Forecasting Experimental Data Terms of Use
  (https://storage.googleapis.com/weathernext-public/terms-of-use.pdf). River
  discharge derived from it is a *non-retrievable Value Added Service* and may be
  published with the citation of §4(b), which the site shows. WeatherNext
  weather fields themselves — including catchment-averaged rain — count as
  unmodified data and must NOT be published: `data/forecasts/` is git-ignored
  and `docs/data/*.json` contain discharge only. Keep it that way.
* Google's framework validates every lead of the past year's forecast inputs;
  for the last 10 issue dates, leads valid after today take today's forecast.

---

## 7b. Phase 1b — all of Spain

Same pipeline, `--domain spain` (bounding box and paths in `config/settings.yaml`):

```bash
python codes/01b_download_cedex_anuario.py            # 1,174 CEDEX gauges, daily to 2022-09-30
python codes/01c_download_saih.py --basin ebro         # recent data: ebro | jucar | guadalquivir | segura
python codes/01d_merge_spain_obs.py                    # CEDEX + SAIH, overlap-checked, QC
python codes/02_delineate_catchments.py --domain spain # 1,154 catchments (HydroSHEDS 90 m)
python codes/02b_plot_spain_gauges.py                  # Figure 2
python codes/03_extract_era5land_forcing.py --domain spain
python codes/04_catchment_attributes.py --domain spain # hypsometry + CEDEX reservoirs
python codes/05_calibrate_gr4j.py --domain spain --precip none
```

| Step | Result |
|---|---|
| Gauges | 1,174 CEDEX gauges; **872 calibratable** (≥ 5 years 2008-10 → 2022-09); **242 testable** after 2023-10 (recent SAIH data: Ebro 194, Júcar 33, Guadalquivir 22) |
| SAIH joins | 211 verified on the 2021-22 overlap, 42 unverified (Júcar: no overlap), 22 rejected (sources disagree → CEDEX only) |
| Catchments | 1,154 delineated; median area error vs CEDEX 1.2 %, 79 % within 5 %, 92 % within 20 % (Guadiana weakest, 78 %: flat karst) |
| Regulation | 556 catchments contain at least one of CEDEX's 398 reservoirs |
| Forcing | ERA5-Land 1981 → today on the native 0.1° grid, exact area-weighted catchment means (same code as the forecast grids will use) |

**GR4J calibration results (863 gauges, calibration 2008-10 → 2022-09, test
2023-10 → today where SAIH data exist; raw ERA5-Land rain):**

| Basin district | gauges | median KGE calib | gauges tested | median KGE test |
|---|---|---|---|---|
| Miño-Sil | 52 | 0.90 | – | – |
| Galicia-Costa | 42 | 0.89 | – | – |
| Duero | 169 | 0.78 | – | – |
| Tajo | 122 | 0.74 | – | – |
| Cantábrico | 44 | 0.72 | – | – |
| Ebro | 217 | 0.71 | 189 | 0.54 |
| Guadiana | 73 | 0.67 | – | – |
| Júcar | 48 | 0.58 | 34 | 0.12 |
| Guadalquivir | 59 | 0.55 | 18 | 0.21 |
| Segura | 37 | 0.14 | 30 | 0.06 |
| **All** | **863** | **0.72** | **271** | **0.45** |

Near-natural catchments (474) reach a median test KGE of 0.54, regulated ones
(389) 0.45. The semi-arid, heavily managed south-east (Segura, lower Júcar,
Guadalquivir: irrigation, inter-basin transfers such as Tajo–Segura) is where
a rainfall-runoff model without water-management knowledge fails.

Status of every basin authority's live-data system, and the rules followed
when using them, are in `docs/spain_extension_plan.md`. In short: Ebro,
Júcar, Guadalquivir and Segura (terms of use accepted; cite CHS) are
connected; Miño-Sil offers only ~3 weeks of history (to be archived by the
daily run);
Cantábrico's download page is deliberately obfuscated (formal data request
instead); Tajo, Duero and Guadiana are pending.

CEDEX-specific QC: its values are rounded to 2–3 decimals and already checked
by CEDEX, so only runs of ≥ 90 identical days are removed there (the 5-day rule
used for 5-minute-derived series would have deleted 2 million genuine
low-flow days).

### Spain in the daily forecast

Built once (locally; the outputs needed daily are committed):

```bash
python codes/04_catchment_attributes.py --domain spain --google  # + 84 Google statics (HydroATLAS, climate indices)
python codes/06_build_google_dataset.py --domain spain            # Google inputs (statics committed)
python codes/14_build_site.py                                    # also writes data/processed/spain/flood_levels.csv
```

Every day (`run_daily.sh`): SAIH Ebro, Júcar, Guadalquivir and Segura last 10
days (`01c --recent 10`) → ERA5-Land (monthly catchment means cached, grids
dropped) → WeatherNext 3 and AIFS cropped to Spain → GR4J for the 863
calibrated gauges (state updating at the ~270 gauges with a verified or
unverified SAIH join; the others run without it and the page says so) →
Google's released model → one site with Catalonia and Spain. A failure in a
Spain step drops Spain from that day's site but never blocks Catalonia.

---

## 8. Repository layout

```
config/settings.yaml          all settings (periods, paths, options)
codes/NN_*.py                 the pipeline, run in numeric order
lib/hydrocat/                 shared library (GR4J, PET, metrics, paths, EE helpers)
tests/                        correctness tests (GR4J vs airGR, FAO-56 PET)
data/raw/                     downloaded / shipped raw data (caches are git-ignored)
data/interim/                 intermediate files and logs
data/processed/               model-ready data and model outputs
outputs/figures, tables       results
external/flood-forecasting/   Google's model (cloned by 00_setup.sh, git-ignored)
```

## 9. References

* Perrin, Michel & Andréassian (2003) GR4J. *J. Hydrol.* 279, 275–289.
* Valéry, Andréassian & Perrin (2014) CemaNeige. *J. Hydrol.* 517, 1166–1175.
* Coron et al. (2017) The suite of lumped GR hydrological models in an R package (airGR). *EMS* 94, 166–171.
* Allen et al. (1998) FAO Irrigation and Drainage Paper 56.
* Nearing et al. (2024) Global prediction of extreme floods in ungauged watersheds. *Nature* 627, 559–563.
* Kratzert et al. (2023) Caravan — a global community dataset for large-sample hydrology. *Sci. Data* 10, 61.
* Gupta et al. (2009) Decomposition of the MSE and NSE (KGE). *J. Hydrol.* 377, 80–91.
* Lehner, Verdin & Jarvis (2008) HydroSHEDS; Linke et al. (2019) HydroATLAS. *Sci. Data* 6, 283.
* Muñoz-Sabater et al. (2021) ERA5-Land. *ESSD* 13, 4349–4383.
