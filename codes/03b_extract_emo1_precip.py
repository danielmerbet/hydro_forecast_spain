#!/usr/bin/env python3
"""
03b — EMO-1 gauge-based precipitation per catchment (diagnostic of ERA5-Land's rain).

WHY
  At 14 of 64 ACA gauges, ERA5-Land precipitation minus observed runoff is
  larger than FAO-56 reference evapotranspiration — water "must be lost",
  and GR4J's exchange term X2 goes to its bound. Either ERA5-Land
  over-estimates rain (plausible in the Pre-Pyrenees) or water is abstracted
  upstream of the gauges. EMO-1 is built from ~57,000 rain gauges (including
  AEMET/Meteocat stations), so comparing the two tells which explanation holds.

DATA
  EMO-1arcmin v3.0.4 (JRC / Copernicus EMS, EFAS), daily precipitation, 1 arcmin
  (~1.8 km), 1990-2024, CC BY 4.0:
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/meteorological_forcings/EMO-1arcmin/pr/
  Each 350 MB European yearly file is downloaded, cut to the domain window
  and deleted (only the window is kept in data/raw/emo1/). Updated yearly -> usable for calibration /
  diagnosis, NOT for the daily operational run.
  Time convention: the value stamped day d+1 06:00 UTC is the accumulation
  06 UTC (d) -> 06 UTC (d+1) and is assigned to day d.

Catchment mean = mean of the EMO-1 cells whose centre lies inside the
catchment polygon (at least the nearest cell for very small catchments).

Writes
  data/processed/forcing_emo1_pr.parquet   long: date, gauge_id, total_precipitation (mm/day)
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from affine import Affine

from hydrocat.config import PROCESSED, RAW, ROOT, P

URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/meteorological_forcings/"
       "EMO-1arcmin/pr/EMO-1arcmin-pr_{year}.nc")
CACHE = RAW / "emo1"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED / "forcing_emo1_pr.parquet"


def log(*a):
    print(*a, flush=True)


def read_year(year: int, bbox) -> xr.DataArray:
    """Download one yearly file (~350 MB), keep only the domain window, delete it.

    (Remote HTTP range reads of the HDF5 file were tried first: they stall when
    run in parallel, and one-at-a-time they are no faster than a plain download.)
    """
    f = CACHE / f"pr_{year}_{'_'.join(f'{b:.2f}' for b in bbox)}.nc"
    if f.exists():
        return xr.open_dataarray(f)
    x0, y0, x1, y1 = bbox
    tmp = CACHE / f"_full_{year}.nc"
    subprocess.run(["curl", "-sS", "--fail", "--retry", "4", "-o", str(tmp), URL.format(year=year)], check=True)
    with xr.open_dataset(tmp, engine="h5netcdf") as ds:
        da = ds["pr"].sel(lat=slice(y1, y0), lon=slice(x0, x1)).load().astype("float32")
    tmp.unlink()
    # value stamped d+1 06:00 UTC = accumulation 06 UTC d -> 06 UTC d+1 -> day d
    da["time"] = (pd.DatetimeIndex(da.time.values) - pd.Timedelta(hours=30)).normalize()
    da.to_netcdf(f)
    log(f"   {year} done")
    return da


def cell_weights(gdf, lat, lon):
    """For each catchment, a boolean mask of EMO-1 cells whose centre is inside it."""
    dx, dy = float(lon[1] - lon[0]), float(lat[1] - lat[0])     # dy < 0 (north -> south)
    tr = Affine(dx, 0, float(lon[0]) - dx / 2, 0, dy, float(lat[0]) - dy / 2)
    masks = {}
    for gid, geom in zip(gdf.gauge_id, gdf.geometry):
        m = rasterio.features.rasterize([(geom, 1)], out_shape=(len(lat), len(lon)), transform=tr,
                                        all_touched=False).astype(bool)
        if not m.any():                                          # tiny catchment: nearest cell
            c = geom.representative_point()
            m[np.abs(lat - c.y).argmin(), np.abs(lon - c.x).argmin()] = True
        masks[gid] = m
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2003)
    ap.add_argument("--end", type=int, default=2024)
    args = ap.parse_args()

    gdf = gpd.read_file(P.catchments)
    x0, y0, x1, y1 = gdf.total_bounds
    bbox = (np.floor(x0 * 10) / 10 - 0.1, np.floor(y0 * 10) / 10 - 0.1,
            np.ceil(x1 * 10) / 10 + 0.1, np.ceil(y1 * 10) / 10 + 0.1)
    years = list(range(args.start, args.end + 1))
    log(f"EMO-1 pr {years[0]}-{years[-1]}, window {bbox}, {len(gdf)} catchments")
    with ThreadPoolExecutor(max_workers=3) as ex:        # 3 parallel downloads
        das = list(ex.map(lambda y: read_year(y, bbox), years))
    da = xr.concat(das, dim="time").sortby("time")
    masks = cell_weights(gdf, da.lat.values, da.lon.values)
    arr = da.values                                               # time, lat, lon
    rows = {gid: np.nanmean(arr[:, m], axis=1) for gid, m in masks.items()}
    w = pd.DataFrame(rows, index=pd.DatetimeIndex(da.time.values, name="date"))
    long = w.stack().rename("total_precipitation").reset_index().rename(columns={"level_1": "gauge_id"})
    long.to_parquet(OUT, index=False)
    log(f"-> {OUT.relative_to(ROOT)}  ({w.index.min().date()} .. {w.index.max().date()})")


if __name__ == "__main__":
    main()
