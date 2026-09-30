#!/usr/bin/env python3
"""
10 — WeatherNext 3 forecast (Google DeepMind), daily catchment means for 15 days.

SOURCE
  Earth Engine ImageCollection
    projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg
  (requires the WeatherNext data-access approval on the Cloud project in
  settings.yaml). One image per (init time, forecast hour 1..360), 0.1 deg,
  ensemble MEAN and p10/p25/p50/p75/p90 of each hourly variable. The 64
  members themselves are only in the requester-pays GCS bucket
  gs://weathernext3_spatial (needs a billing account; not used).

WHAT IS COMPUTED
  The latest fully published 00 UTC run (all 360 hours present) is taken.
  Hours (24k, 24k+24] after the init form forecast day k = 0..14 (UTC days),
  aggregated server-side, downloaded as one small grid and turned into
  catchment means with the same area weights as ERA5-Land (the two products
  share the same 0.1 deg grid):
    total_precipitation        mm/day   sum of hourly ensemble-mean rain
    precip_p10 / precip_p90    mm/day   sums of the hourly p10 / p90. NOT daily
                                        ensemble percentiles (unknowable without
                                        members): an envelope, labelled as such.
    temperature_2m(_min/_max)  degC     daily mean / min / max of hourly mean
    dewpoint_temperature_2m    degC     daily mean
    wind_speed_10m             m/s      daily mean
    surface_net_solar/thermal_radiation  W/m2  FAO-56 eqs. 37-40 from downward
                                        solar radiation, temperature, humidity
    surface_pressure           kPa      FAO-56 eq. 7 from catchment median elevation
    pet_fao56                  mm/day   FAO-56 Penman-Monteith (same as the reanalysis)
  Units and names match the ERA5-Land forcing of script 03, so the models can
  switch from reanalysis to forecast without conversion.

Writes  data/forecasts/<domain>/weathernext3/<init YYYYMMDDHH>/forcing.parquet (+ meta.json)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile

from hydrocat.config import DATA, PROCESSED, ROOT, load_settings
from hydrocat.eeutils import init_ee, retry
from hydrocat.gridweights import catchment_means, catchment_weights
from hydrocat.pet import fao56_net_radiation, fao56_penman_monteith, pressure_from_elevation

S = load_settings()
ASSET = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_0p1deg"
DAYS = 15
AGG = [  # (output band, source band, reducer)
    ("tp", "total_precipitation_1hr_mean", "sum"),
    ("tp_p10", "total_precipitation_1hr_p10", "sum"),
    ("tp_p90", "total_precipitation_1hr_p90", "sum"),
    ("t2m", "temperature_2m_mean", "mean"),
    ("t2m_min", "temperature_2m_mean", "min"),
    ("t2m_max", "temperature_2m_mean", "max"),
    ("d2m", "dewpoint_temperature_2m_mean", "mean"),
    ("ws10", "wind_speed_10m_mean", "mean"),
    ("ssrd", "surface_solar_radiation_downwards_1hr_mean", "sum"),
]


def log(*a):
    print(*a, flush=True)


def latest_init(col) -> dt.datetime:
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    for back in range(0, 24 * 5, 24):
        t = (now - dt.timedelta(hours=back)).replace(hour=0)
        n = retry(col.filter(ee.Filter.eq("start_time", t.strftime("%Y-%m-%dT%H:%M:%SZ")))
                  .filter(ee.Filter.eq("forecast_hour", 360)).size().getInfo)
        if n:
            return t
    raise RuntimeError("no complete WeatherNext 3 00 UTC run in the last 5 days")


def daily_stack(col, init: dt.datetime) -> ee.Image:
    run = col.filter(ee.Filter.eq("start_time", init.strftime("%Y-%m-%dT%H:%M:%SZ")))
    imgs = []
    for k in range(DAYS):
        day = run.filter(ee.Filter.rangeContains("forecast_hour", 24 * k + 1, 24 * k + 24))
        bands = []
        for name, src, red in AGG:
            c = day.select(src)
            b = {"sum": c.sum(), "mean": c.mean(), "min": c.min(), "max": c.max()}[red]
            bands.append(b.rename(f"d{k:02d}_{name}"))
        imgs.append(ee.Image.cat(bands))
    return ee.Image.cat(imgs).toFloat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--init", default=None, help="YYYY-MM-DD (00 UTC run); default: latest complete")
    args = ap.parse_args()
    D = S["domains"][args.domain]
    init_ee()
    col = ee.ImageCollection(ASSET)
    init = dt.datetime.fromisoformat(args.init) if args.init else latest_init(col)
    out = DATA / "forecasts" / args.domain / "weathernext3" / init.strftime("%Y%m%d%H")
    out.mkdir(parents=True, exist_ok=True)
    log(f"WeatherNext 3 run {init:%Y-%m-%d %H} UTC, {DAYS} days, domain {args.domain}")

    # same grid as ERA5-Land (0.1 deg, cell centres on whole tenths): reuse script 03's grid
    import importlib.util
    spec = importlib.util.spec_from_file_location("s03", ROOT / "codes" / "03_extract_era5land_forcing.py")
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)
    tr, (ny, nx) = s03.domain_grid(D["bbox"])
    raw = retry(ee.data.computePixels, {
        "expression": daily_stack(col, init), "fileFormat": "GEO_TIFF",
        "grid": {"dimensions": {"width": nx, "height": ny},
                 "affineTransform": {"scaleX": tr.a, "shearX": 0, "translateX": tr.c,
                                     "shearY": 0, "scaleY": tr.e, "translateY": tr.f},
                 "crsCode": "EPSG:4326"}})
    with MemoryFile(raw) as mem, mem.open() as ds:
        a = ds.read().astype(np.float64)
    a[~np.isfinite(a)] = np.nan
    a = a.reshape(DAYS, len(AGG), ny, nx)

    gdf = gpd.read_file(ROOT / D["catchments"])
    W = catchment_weights(gdf, tr, (ny, nx))
    m = {name: catchment_means(W, a[:, i]) for i, (name, _, _) in enumerate(AGG)}   # (days, n_catch)
    dates = pd.date_range(init.date(), periods=DAYS, freq="D")
    base = PROCESSED if args.domain == "catalonia" else PROCESSED / args.domain
    hy = pd.read_csv(base / "hypsometry.csv").set_index("gauge_id")
    elev = gdf.gauge_id.map(hy["p50"]).values
    lat = gdf.geometry.representative_point().y.values
    n_c = len(gdf)
    f = pd.DataFrame({"date": np.repeat(dates, n_c), "gauge_id": np.tile(gdf.gauge_id.values, DAYS)})
    rep = lambda x: np.tile(x, DAYS)  # noqa: E731
    f["total_precipitation"] = np.clip(m["tp"].ravel() * 1000, 0, None)
    f["precip_p10"] = np.clip(m["tp_p10"].ravel() * 1000, 0, None)
    f["precip_p90"] = np.clip(m["tp_p90"].ravel() * 1000, 0, None)
    f["temperature_2m"] = m["t2m"].ravel() - 273.15
    f["temperature_2m_min"] = m["t2m_min"].ravel() - 273.15
    f["temperature_2m_max"] = m["t2m_max"].ravel() - 273.15
    f["dewpoint_temperature_2m"] = m["d2m"].ravel() - 273.15
    f["wind_speed_10m"] = m["ws10"].ravel()
    rs = m["ssrd"].ravel() / 1e6                                            # J m-2 day-1 -> MJ
    rns, rnl = fao56_net_radiation(rs, f.temperature_2m_min, f.temperature_2m_max,
                                   f.dewpoint_temperature_2m, rep(lat), f.date.dt.dayofyear.values, rep(elev))
    f["surface_net_solar_radiation"] = rns * 1e6 / 86400                     # W/m2, as ERA5-Land
    f["surface_net_thermal_radiation"] = rnl * 1e6 / 86400
    f["surface_pressure"] = pressure_from_elevation(rep(elev))
    f["pet_fao56"] = fao56_penman_monteith(f.temperature_2m_min, f.temperature_2m_max, f.dewpoint_temperature_2m,
                                           rns + rnl, f.surface_pressure, f.wind_speed_10m)
    f.to_parquet(out / "forcing.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({
        "source": ASSET, "init_utc": init.isoformat(), "days": DAYS,
        "precip_envelope": "precip_p10/p90 are sums of hourly ensemble percentiles, not daily percentiles",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=1))
    log(f"-> {out.relative_to(ROOT)}/forcing.parquet  ({n_c} catchments x {DAYS} days)")
    s = f.groupby("date")[["total_precipitation", "precip_p90", "temperature_2m", "pet_fao56"]].median()
    log("median over catchments:\n" + s.round(2).to_string())


if __name__ == "__main__":
    main()
