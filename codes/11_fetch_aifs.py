#!/usr/bin/env python3
"""
11 — ECMWF AIFS forecast (open data), daily catchment means for 15 days.

SOURCE — ECMWF open data (CC BY 4.0, attribute ECMWF), no key needed:
  aifs-single  deterministic AIFS, 0.25 deg, 6-hourly steps 0..360 h
               params tp, 2t, 2d, 10u, 10v, sp, ssrd
  aifs-ens     AIFS ensemble, 50 perturbed members: ONLY total precipitation,
               at daily steps (0, 24, ..., 360 h). Open data are global fields
               with no spatial subsetting; the full ensemble with all variables
               at 6-hourly steps would be ~10-18 GB per run, too much for a daily
               job, while rain-only daily steps are ~0.6 GB and carry the
               uncertainty that matters most for floods.

DAILY AGGREGATION (UTC days k = 0..14 after the 00 UTC init)
  tp, ssrd   accumulated from init -> field(24k+24) - field(24k)
  2t, 2d, sp, wind speed  mean of the four 6-hourly instants 24k+6 .. 24k+24;
             Tmin/Tmax = min/max of those four instants (coarser than hourly
             data: an approximation, used only for FAO-56 PET and snow).
  Precipitation units are read from each GRIB file (AIFS: kg m-2 = mm; IFS:
  m) and never assumed — a wrong guess is a factor 1000.
  Net radiation, pressure and PET: FAO-56, exactly as for WeatherNext 3
  (script 10), so the two forecast products are processed identically.

Writes  data/forecasts/<domain>/aifs/<init YYYYMMDDHH>/forcing.parquet          deterministic
        data/forecasts/<domain>/aifs/<init>/precip_members.parquet              date, gauge_id, member, tp
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from hydrocat.config import DATA, PROCESSED, RAW, ROOT, load_settings
from hydrocat.gridweights import catchment_means, catchment_weights
from hydrocat.pet import fao56_net_radiation, fao56_penman_monteith

S = load_settings()
DAYS = 15
RES = 0.25
PARAMS = ["tp", "2t", "2d", "10u", "10v", "sp", "ssrd"]


def log(*a):
    print(*a, flush=True)


def to_mm(da):
    u = (da.attrs.get("units") or da.attrs.get("GRIB_units") or "").strip()
    if u == "m":
        return da * 1000.0
    if u in ("kg m**-2", "kg m-2", "mm"):
        return da
    raise ValueError(f"unrecognised precipitation units {u!r} — refusing to guess")


def crop(da, bbox):
    x0, y0, x1, y1 = bbox
    lon = da.longitude
    if float(lon.max()) > 180:                                  # 0..360 -> -180..180
        da = da.assign_coords(longitude=((lon + 180) % 360) - 180).sortby("longitude")
    da = da.sortby("latitude", ascending=False)
    return da.sel(longitude=slice(x0 - RES, x1 + RES), latitude=slice(y1 + RES, y0 - RES))


def grid_of(da):
    lon, lat = da.longitude.values, da.latitude.values
    return Affine(RES, 0, float(lon[0]) - RES / 2, 0, -RES, float(lat[0]) + RES / 2), (len(lat), len(lon))


def open_param(path, name):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    var = list(ds.data_vars)[0]
    return ds[var]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--no-ens", action="store_true", help="skip the AIFS ensemble precipitation")
    args = ap.parse_args()
    from ecmwf.opendata import Client
    D = S["domains"][args.domain]
    cli = Client(source="ecmwf", model="aifs-single")
    init = cli.latest(param="tp", step=360)
    init = init.replace(hour=0) if init.hour else init                      # the day's 00 UTC run
    cache = RAW / "aifs" / init.strftime("%Y%m%d%H")
    cache.mkdir(parents=True, exist_ok=True)
    out = DATA / "forecasts" / args.domain / "aifs" / init.strftime("%Y%m%d%H")
    out.mkdir(parents=True, exist_ok=True)
    log(f"AIFS run {init:%Y-%m-%d %H} UTC, domain {args.domain}")

    steps = list(range(0, 24 * DAYS + 1, 6))
    fields = {}
    for p in PARAMS:
        f = cache / f"single_{p}.grib2"
        if not f.exists():
            cli.retrieve(date=init.date(), time=0, step=steps, param=p, target=str(f))
        fields[p] = crop(open_param(f, p), D["bbox"])
    tr, shape = grid_of(fields["tp"])
    gdf = gpd.read_file(ROOT / D["catchments"])
    W = catchment_weights(gdf, tr, shape)

    def cm(a):  # (days, ny, nx) -> (days, n_catch)
        return catchment_means(W, np.asarray(a, float))

    def at(da, h):
        return da.sel(step=pd.Timedelta(hours=h)).values

    tp = to_mm(fields["tp"])
    daily = {k: [] for k in ["tp", "ssrd", "t", "tmin", "tmax", "td", "ws", "sp"]}
    ws = np.hypot(fields["10u"], fields["10v"])
    for k in range(DAYS):
        h0, h1 = 24 * k, 24 * k + 24
        inst = [24 * k + 6, 24 * k + 12, 24 * k + 18, 24 * k + 24]
        daily["tp"].append(np.clip(at(tp, h1) - at(tp, h0), 0, None))
        daily["ssrd"].append(np.clip(at(fields["ssrd"], h1) - at(fields["ssrd"], h0), 0, None))
        t = np.stack([at(fields["2t"], h) for h in inst])
        daily["t"].append(t.mean(0)); daily["tmin"].append(t.min(0)); daily["tmax"].append(t.max(0))
        daily["td"].append(np.mean([at(fields["2d"], h) for h in inst], 0))
        daily["ws"].append(np.mean([ws.sel(step=pd.Timedelta(hours=h)).values for h in inst], 0))
        daily["sp"].append(np.mean([at(fields["sp"], h) for h in inst], 0))
    m = {k: cm(np.stack(v)) for k, v in daily.items()}

    dates = pd.date_range(init.date(), periods=DAYS, freq="D")
    n_c = len(gdf)
    base = PROCESSED if args.domain == "catalonia" else PROCESSED / args.domain
    elev = gdf.gauge_id.map(pd.read_csv(base / "hypsometry.csv").set_index("gauge_id")["p50"]).values
    lat = gdf.geometry.representative_point().y.values
    rep = lambda x: np.tile(x, DAYS)  # noqa: E731
    f = pd.DataFrame({"date": np.repeat(dates, n_c), "gauge_id": np.tile(gdf.gauge_id.values, DAYS)})
    f["total_precipitation"] = m["tp"].ravel()
    f["temperature_2m"] = m["t"].ravel() - 273.15
    f["temperature_2m_min"] = m["tmin"].ravel() - 273.15
    f["temperature_2m_max"] = m["tmax"].ravel() - 273.15
    f["dewpoint_temperature_2m"] = m["td"].ravel() - 273.15
    f["wind_speed_10m"] = m["ws"].ravel()
    f["surface_pressure"] = m["sp"].ravel() / 1000
    rns, rnl = fao56_net_radiation(m["ssrd"].ravel() / 1e6, f.temperature_2m_min, f.temperature_2m_max,
                                   f.dewpoint_temperature_2m, rep(lat), f.date.dt.dayofyear.values, rep(elev))
    f["surface_net_solar_radiation"] = rns * 1e6 / 86400
    f["surface_net_thermal_radiation"] = rnl * 1e6 / 86400
    f["pet_fao56"] = fao56_penman_monteith(f.temperature_2m_min, f.temperature_2m_max, f.dewpoint_temperature_2m,
                                           rns + rnl, f.surface_pressure, f.wind_speed_10m)
    f.to_parquet(out / "forcing.parquet", index=False)
    log(f"-> {out.relative_to(ROOT)}/forcing.parquet")

    if not args.no_ens:
        ens = Client(source="ecmwf", model="aifs-ens")
        fe = cache / "ens_tp.grib2"
        if not fe.exists():
            ens.retrieve(date=init.date(), time=0, type="pf", step=list(range(0, 24 * DAYS + 1, 24)),
                         number=list(range(1, 51)), param="tp", target=str(fe))
        da = crop(to_mm(open_param(fe, "tp")), D["bbox"])
        tr_e, shape_e = grid_of(da)
        We = W if shape_e == shape else catchment_weights(gdf, tr_e, shape_e)
        rows = []
        for num in da.number.values:
            acc = da.sel(number=num)
            dd = np.stack([np.clip(acc.sel(step=pd.Timedelta(hours=24 * k + 24)).values
                                   - acc.sel(step=pd.Timedelta(hours=24 * k)).values, 0, None) for k in range(DAYS)])
            mm = catchment_means(We, dd)
            rows.append(pd.DataFrame({"date": np.repeat(dates, n_c), "gauge_id": np.tile(gdf.gauge_id.values, DAYS),
                                      "member": int(num), "total_precipitation": mm.ravel()}))
        pd.concat(rows).to_parquet(out / "precip_members.parquet", index=False)
        log(f"-> {out.relative_to(ROOT)}/precip_members.parquet ({da.number.size} members)")

    (out / "meta.json").write_text(json.dumps({
        "source": "ECMWF open data aifs-single (+ aifs-ens tp)", "licence": "CC BY 4.0, (c) ECMWF",
        "init_utc": init.isoformat(), "days": DAYS,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=1))
    s = f.groupby("date")[["total_precipitation", "temperature_2m", "pet_fao56"]].median()
    log("median over catchments:\n" + s.round(2).to_string())


if __name__ == "__main__":
    main()
