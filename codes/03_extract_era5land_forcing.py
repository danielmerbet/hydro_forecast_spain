#!/usr/bin/env python3
"""
03 — Daily ERA5-Land forcing, averaged over every catchment (1981 → ~today-7d).

WHY ERA5-LAND, AND WHY THE SAME FOR BOTH MODELS
  Google's model was trained on Caravan, whose meteorological inputs and
  climate indices all come from ERA5-Land. Feeding GR4J the *same* catchment
  means lets the comparison isolate the hydrological model: if one model does
  better, it is not because it saw better rain. ERA5-Land (0.1 deg, ~9 km) is
  coarse for the smallest catchments and under-represents convective
  extremes; that limitation is shared by both models (see also 03b-03d for its
  precipitation bias against gauge-based EMO-1).

WHY FROM 1981
  Caravan's static climate indices (p_mean, aridity, ...) are defined over
  1981-2020 (script 04). GR4J only needs 2003 onwards (5-year spin-up).

HOW — grid first, catchment means locally
  1. For each month, the ERA5-Land grid over the domain (native 0.1 deg cells)
     is downloaded from Earth Engine (ECMWF/ERA5_LAND/DAILY_AGGR, UTC days) in
     one request: ~3 s per month for all of Spain. The daily mean 10 m wind
     SPEED is computed server-side from ECMWF/ERA5_LAND/HOURLY
     (mean of sqrt(u^2+v^2); averaging u and v first would underestimate it).
  2. Catchment means use exact area fractions of each grid cell
     (hydrocat.gridweights, the same code later used for the forecast grids).
  An earlier version averaged over polygons inside Earth Engine
  (reduceRegions); it was ~100x slower for Spain's ~1,150 overlapping
  catchments and gives the same numbers to within sub-cell sampling.

UNITS (identical to Caravan's conventions)
  total_precipitation    mm/day          (m * 1000)
  temperature_2m(_min/_max) degC         (K - 273.15)
  surface_net_solar_radiation   W/m2     daily mean (J m-2 day-1 / 86400)
  surface_net_thermal_radiation W/m2     daily mean (negative = net loss)
  surface_pressure       kPa             (Pa / 1000)
  dewpoint_temperature_2m degC
  wind_speed_10m         m/s             daily mean of hourly speeds
  potential_evaporation  mm/day, positive — ERA5-Land's own PE (sign flipped).
                         Known to be far too high; kept ONLY because Google's
                         static attribute pet_mean_ERA5_LAND is defined with it.
  pet_fao56              mm/day          FAO-56 Penman-Monteith reference ET from the
                                         fields above (hydrocat.pet): GR4J's evaporation.

Usage
  python codes/03_extract_era5land_forcing.py                      # Catalonia, 1981 ->
  python codes/03_extract_era5land_forcing.py --domain spain --start 2003-01-01

Writes   <domain forcing path in settings.yaml>   long table: date, gauge_id, variables
Caches   data/raw/era5land_grid_<domain>/<YYYY-MM>.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.io import MemoryFile

from hydrocat.config import RAW, ROOT, load_settings
from hydrocat.eeutils import init_ee, retry
from hydrocat.gridweights import catchment_means, catchment_weights
from hydrocat.pet import fao56_penman_monteith

S = load_settings()
BANDS = ["total_precipitation_sum", "temperature_2m", "temperature_2m_min", "temperature_2m_max",
         "dewpoint_temperature_2m", "surface_net_solar_radiation_sum", "surface_net_thermal_radiation_sum",
         "surface_pressure", "potential_evaporation_sum"]
ALL = BANDS + ["wind_speed_10m"]
HOURLY = "ECMWF/ERA5_LAND/HOURLY"
RES = 0.1                      # ERA5-Land native grid: cell centres on whole tenths of a degree


def log(*a):
    print(*a, flush=True)


def domain_grid(bbox):
    """Native-aligned grid (edges at x.x5) covering the bbox."""
    x0, y0, x1, y1 = bbox
    gx0 = np.floor((x0 - 0.05) / RES) * RES + 0.05
    gy1 = np.ceil((y1 - 0.05) / RES) * RES + 0.05
    nx = int(np.ceil((x1 - gx0) / RES))
    ny = int(np.ceil((gy1 - y0) / RES))
    return Affine(RES, 0, round(gx0, 4), 0, -RES, round(gy1, 4)), (ny, nx)


def fetch_month(m0: dt.date, tr: Affine, shape, cache) -> dict:
    """One month of daily ERA5-Land fields on the domain grid -> {band: (days, ny, nx)}."""
    f = cache / f"{m0:%Y-%m}.npz"
    m1 = (pd.Timestamp(m0) + pd.offsets.MonthBegin(1)).date()
    hourly = ee.ImageCollection(HOURLY).select(["u_component_of_wind_10m", "v_component_of_wind_10m"])

    def add_wind_speed(img):
        d0 = ee.Date(img.get("system:time_start"))
        ws = (hourly.filterDate(d0, d0.advance(1, "day"))
              .map(lambda h: h.pow(2).reduce("sum").sqrt()).mean().rename("wind_speed_10m"))
        return img.select(BANDS).addBands(ws)

    col = ee.ImageCollection(S["forcing"]["era5land_asset"]).filterDate(str(m0), str(m1)).map(add_wind_speed)
    dates = retry(col.aggregate_array("system:index").getInfo)
    if not dates:
        return {}
    img = col.toBands().toFloat()
    ny, nx = shape
    raw = retry(ee.data.computePixels, {
        "expression": img, "fileFormat": "GEO_TIFF",
        "grid": {"dimensions": {"width": nx, "height": ny},
                 "affineTransform": {"scaleX": tr.a, "shearX": 0, "translateX": tr.c,
                                     "shearY": 0, "scaleY": tr.e, "translateY": tr.f},
                 "crsCode": "EPSG:4326"}})
    with MemoryFile(raw) as mem, mem.open() as ds:
        a = ds.read().astype(np.float32)                   # (days * bands, ny, nx), day-major
    a[~np.isfinite(a)] = np.nan                            # sea / masked cells come back as -inf
    a = a.reshape(len(dates), len(ALL), ny, nx)
    out = {b: a[:, i] for i, b in enumerate(ALL)} | {"dates": np.array(sorted(dates))}
    np.savez_compressed(f, **out)
    return out


def to_units(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.to_datetime(df["date"]), "gauge_id": df["gauge_id"]})
    d["total_precipitation"] = (df["total_precipitation_sum"] * 1000).clip(lower=0)
    d["temperature_2m"] = df["temperature_2m"] - 273.15
    d["temperature_2m_min"] = df["temperature_2m_min"] - 273.15
    d["temperature_2m_max"] = df["temperature_2m_max"] - 273.15
    d["surface_net_solar_radiation"] = df["surface_net_solar_radiation_sum"] / 86400
    d["surface_net_thermal_radiation"] = df["surface_net_thermal_radiation_sum"] / 86400
    d["surface_pressure"] = df["surface_pressure"] / 1000
    d["potential_evaporation"] = (-df["potential_evaporation_sum"] * 1000).clip(lower=0)
    d["dewpoint_temperature_2m"] = df["dewpoint_temperature_2m"] - 273.15
    d["wind_speed_10m"] = df["wind_speed_10m"]
    d["pet_fao56"] = fao56_penman_monteith(
        d["temperature_2m_min"], d["temperature_2m_max"], d["dewpoint_temperature_2m"],
        (df["surface_net_solar_radiation_sum"] + df["surface_net_thermal_radiation_sum"]) / 1e6,
        d["surface_pressure"], d["wind_speed_10m"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--start", default="1981-01-01")
    ap.add_argument("--refresh-days", type=int, default=45,
                    help="months overlapping the last N days are always re-downloaded")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    D = S["domains"][args.domain]
    cache = RAW / f"era5land_grid_{args.domain}"
    cache.mkdir(parents=True, exist_ok=True)
    out = ROOT / D["forcing"]

    init_ee()
    gdf = gpd.read_file(ROOT / D["catchments"])
    tr, shape = domain_grid(D["bbox"])
    log(f"{args.domain}: grid {shape[0]} x {shape[1]} cells of {RES} deg, {len(gdf)} catchments")

    months = pd.date_range(args.start, dt.date.today(), freq="MS").date
    refresh_from = pd.Timestamp(dt.date.today() - dt.timedelta(days=args.refresh_days)).replace(day=1).date()
    todo = [m for m in months if m >= refresh_from or not (cache / f"{m:%Y-%m}.npz").exists()]
    log(f"   {len(months)} months, {len(todo)} to download")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_month, m, tr, shape, cache) for m in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 48 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)}")

    log("   catchment weights + means")
    W = catchment_weights(gdf, tr, shape)
    frames = []
    for m in months:
        f = cache / f"{m:%Y-%m}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        dates = pd.to_datetime(z["dates"], format="%Y%m%d")
        cols = {b: catchment_means(W, z[b]) for b in ALL}                  # each (days, n_catch)
        n_d, n_c = len(dates), len(gdf)
        frames.append(pd.DataFrame({"date": np.repeat(dates, n_c), "gauge_id": np.tile(gdf.gauge_id.values, n_d),
                                    **{b: cols[b].ravel() for b in ALL}}))
    raw = pd.concat(frames, ignore_index=True).dropna(subset=["total_precipitation_sum"])
    forcing = to_units(raw).sort_values(["gauge_id", "date"]).reset_index(drop=True)
    missing = set(gdf.gauge_id) - set(forcing.gauge_id)
    if missing:
        raise RuntimeError(f"no forcing for {sorted(missing)} (outside the ERA5-Land land mask?)")
    out.parent.mkdir(parents=True, exist_ok=True)
    forcing.to_parquet(out, index=False)
    log(f"-> {out.relative_to(ROOT)}: {forcing.date.min().date()} .. {forcing.date.max().date()}, "
        f"{forcing.gauge_id.nunique()} catchments")
    s = forcing.groupby("gauge_id")[["total_precipitation", "potential_evaporation", "pet_fao56", "temperature_2m"]].mean()
    log("mean annual P / PE(ERA5-Land raw) / ET0(FAO-56 PM) [mm/yr], T [degC] — range over catchments:")
    log((s * [365.25, 365.25, 365.25, 1]).describe().loc[["min", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
