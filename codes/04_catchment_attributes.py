#!/usr/bin/env python3
"""
04 — Static catchment attributes.

Three groups of attributes, for three different uses:

1. HYPSOMETRY (for GR4J's CemaNeige snow module)
   101 elevation quantiles (0, 1, ..., 100 %) of each catchment, from the
   SRTM 30 m DEM (USGS/SRTMGL1_003) sampled at 90 m in Earth Engine. CemaNeige
   splits the catchment into equal-area elevation bands from these.

2. GOOGLE MODEL STATICS (exactly the 84 attributes in the pretrained config)
   * 74 HydroATLAS attributes (Linke et al., 2019), from the level-12
     BasinATLAS polygons intersecting each catchment (Earth Engine
     WWF/HydroATLAS/v1/Basins/level12). Aggregated like the earlier Onyar
     build: area-weighted mean, except `*_smn` (min) and `*_smx` (max).
   * 10 climate indices computed from the ERA5-Land forcing of script 03
     over 1981-2020, with Caravan's own definitions (Kratzert et al., 2023,
     caravan_utils.calculate_climate_indices): p_mean, pet_mean_ERA5_LAND,
     aridity_ERA5_LAND, moisture_index_ERA5_LAND, seasonality_ERA5_LAND
     (Knoben et al., 2018, from monthly climatologies), frac_snow,
     high/low_prec_freq/dur. NOTE: pet here is ERA5-Land's own
     `potential_evaporation`, which is known to be far too high; that is
     deliberate, because it is what the model was trained with (the scaler's
     global mean of pet_mean_ERA5_LAND is 10.5 mm/day).

3. REGULATION FLAGS (for interpreting results)
   A gauge is flagged `regulated` when an ACA reservoir outflow station (E01
   Boadella, E06 Sau, E07 Susqueda, E14 La Baells, ...) lies inside its
   catchment; every large reservoir of the internal basins has one.
   HydroATLAS's degree of regulation (dor_pc_pva) is kept for information but
   NOT used: it is a reach attribute, and a headwater catchment sharing a
   level-12 polygon with a downstream dam inherits the dam's value (EA050,
   EA094 show >600 % although both lie upstream of their reservoir). Neither
   GR4J nor Google's model knows about dam operations, so both are expected to
   do worse there; results are reported separately for natural and regulated
   gauges.

Writes
  data/processed/catchment_attributes.csv   one row per gauge
  data/processed/hypsometry.csv             101 elevation quantiles per gauge
"""
from __future__ import annotations

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

from hydrocat.config import PROCESSED, ROOT, P, load_settings
from hydrocat.eeutils import init_ee, retry

S = load_settings()
HYPSO = PROCESSED / "hypsometry.csv"


def log(*a):
    print(*a, flush=True)


def google_static_names() -> list[str]:
    cfg = ROOT / S["google"]["repo_dir"] / S["google"]["pretrained_runs"]["baseline"] / "config.yml"
    return yaml.safe_load(cfg.read_text())["static_attributes"]


# ---------------------------------------------------------------------------
def hypsometry(gdf) -> pd.DataFrame:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    pct = list(range(101))
    rows = []
    for _, g in gdf.iterrows():
        geom = ee.Geometry(g.geometry.simplify(0.0005).__geo_interface__)
        r = retry(dem.reduceRegion(ee.Reducer.percentile(pct), geom, 90, maxPixels=1e9).getInfo)
        rows.append([g.gauge_id] + [r[f"elevation_p{p}"] for p in pct])
    return pd.DataFrame(rows, columns=["gauge_id"] + [f"p{p}" for p in pct])


def hydroatlas(gdf, names, tile_deg: float | None = None) -> pd.DataFrame:
    """tile_deg: download the level-12 polygons in tiles (Spain: ~15,000 polygons,
    too many for one request); polygons on tile edges are de-duplicated by HYBAS_ID."""
    x0, y0, x1, y1 = gdf.total_bounds
    props = sorted(set(n for n in names if not n.endswith("_ERA5_LAND")) | {"dor_pc_pva", "HYBAS_ID"})
    boxes = [(x0, y0, x1, y1)] if not tile_deg else [
        (x, y, min(x + tile_deg, x1), min(y + tile_deg, y1))
        for x in np.arange(x0, x1, tile_deg) for y in np.arange(y0, y1, tile_deg)]
    parts = []
    for b in boxes:
        fc = ee.FeatureCollection("WWF/HydroATLAS/v1/Basins/level12").filterBounds(ee.Geometry.Rectangle(list(b)))
        parts.append(retry(ee.data.computeFeatures, {"expression": fc.select(props),
                                                     "fileFormat": "GEOPANDAS_GEODATAFRAME"}))
    lev = pd.concat(parts, ignore_index=True).drop_duplicates("HYBAS_ID").set_crs("EPSG:4326")
    log(f"   {len(lev)} level-12 BasinATLAS polygons")
    eq = "EPSG:3035"   # equal-area for weights
    inter = gpd.overlay(gdf[["gauge_id", "geometry"]].to_crs(eq), lev.to_crs(eq), how="intersection")
    inter["w"] = inter.area
    out = []
    for gid, grp in inter.groupby("gauge_id"):
        w = grp["w"] / grp["w"].sum()
        row = {"gauge_id": gid}
        for n in props:
            if n == "HYBAS_ID" or n not in grp:
                continue
            v = grp[n].astype(float)
            row[n] = v.min() if n.endswith("_smn") else v.max() if n.endswith("_smx") else float((v * w).sum())
        out.append(row)
    return pd.DataFrame(out)


def climate_indices(forcing: pd.DataFrame) -> pd.DataFrame:
    """Caravan definitions (caravan_utils.calculate_climate_indices), 1981-2020."""
    f = forcing[(forcing.date >= "1981-01-01") & (forcing.date <= "2020-12-31")]
    out = []
    for gid, d in f.groupby("gauge_id"):
        d = d.set_index("date")
        p, pet, t = d["total_precipitation"], d["potential_evaporation"], d["temperature_2m"]
        p_mean, pet_mean = p.mean(), pet.mean()
        mp, mpet, mt = (x.groupby(x.index.month).mean() for x in (p, pet, t))
        mi = np.where(mp > mpet, 1 - mpet / mp, np.where(mp < mpet, mp / mpet - 1, 0.0))

        def mean_run(mask):
            idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
            return float((idx[1::2] - idx[::2]).mean()) if len(idx) else 0.0

        hi, lo = (p >= 5 * p_mean).values, (p < 1).values
        out.append(dict(gauge_id=gid, p_mean=p_mean, pet_mean_ERA5_LAND=pet_mean,
                        aridity_ERA5_LAND=pet_mean / p_mean, frac_snow=mp[mt < 0].sum() / mp.sum(),
                        moisture_index_ERA5_LAND=float(mi.mean()), seasonality_ERA5_LAND=float(mi.max() - mi.min()),
                        high_prec_freq=hi.mean(), high_prec_dur=mean_run(hi),
                        low_prec_freq=lo.mean(), low_prec_dur=mean_run(lo)))
    return pd.DataFrame(out)


def reservoirs_in(gdf) -> pd.Series:
    """ACA reservoir outflow stations (E-codes) inside each catchment."""
    ex = pd.concat([pd.read_csv(f) for f in sorted((ROOT / S["gauges"]["export_dir"]).glob("Q_*.csv.gz"))])
    res = ex[ex.variable.str.match(r"^E\d")].drop_duplicates("variable")
    res = res.assign(code=res.variable.str.split("_").str[0]).drop_duplicates("code")
    lon, lat = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True).transform(res.X.values, res.Y.values)
    pts = gpd.GeoDataFrame(res[["code", "station"]], geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    # 300 m buffer: dam stations sit on the river right at the catchment edge
    j = gpd.sjoin(pts.to_crs("EPSG:25831").assign(geometry=lambda x: x.buffer(300)),
                  gdf[["gauge_id", "geometry"]].to_crs("EPSG:25831"), predicate="intersects")
    return j.groupby("gauge_id")["code"].apply(lambda s: "|".join(sorted(set(s))))


def hypsometry_batched(gdf, batch=100) -> pd.DataFrame:
    """Same SRTM percentiles as hypsometry(), in batches (Spain: ~1,150 catchments).
    Catchments < 500 km2 at 90 m, larger ones at 500 m (hypsometry of a large
    catchment does not need 90 m, and the request stays small)."""
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    pct = list(range(101))
    red = ee.Reducer.percentile(pct)
    out = []
    for small in (True, False):
        sub = gdf[(gdf.area_km2 < 500) == small]
        scale = 90 if small else 500
        for i in range(0, len(sub), batch):
            part = sub.iloc[i:i + batch][["gauge_id", "geometry"]].copy()
            part["geometry"] = part.geometry.simplify(0.002 if small else 0.005)
            fc = ee.FeatureCollection(part.__geo_interface__)
            df = retry(ee.data.computeFeatures, {
                "expression": dem.reduceRegions(fc, red, scale, tileScale=4).map(lambda f: f.setGeometry(None)),
                "fileFormat": "PANDAS_DATAFRAME"})
            out.append(df)
            log(f"   hypsometry {'small' if small else 'large'} {i + len(part)}/{len(sub)}")
    d = pd.concat(out, ignore_index=True)
    cols = {f"p{p}": d[f"p{p}"] if f"p{p}" in d else d[str(p)] for p in pct}
    return pd.DataFrame({"gauge_id": d.gauge_id, **cols})


def cedex_reservoirs() -> gpd.GeoDataFrame:
    """All CEDEX reservoirs (embalse.csv of every basin district), as points."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s01b", ROOT / "codes" / "01b_download_cedex_anuario.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = []
    for dem in m.DEMARCACIONES:
        e = m.fetch(dem, "embalse.csv")
        e = e.assign(lon=e.longwgs84.map(m.dms), lat=e.latwgs84.map(m.dms), demarcacion=dem)
        rows.append(e[["ref_ceh", "nom_embalse", "lon", "lat", "demarcacion"]])
    r = pd.concat(rows).dropna(subset=["lon", "lat"])
    return gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.lon, r.lat), crs="EPSG:4326")


def spain_main(google: bool = False):
    """Spain: hypsometry (GR4J snow bands) and regulation flags; with google=True
    also the 84 Google statics (HydroATLAS + Caravan climate indices, exactly
    as for Catalonia) for the catchments with calibrated GR4J parameters, the
    ones that are forecast."""
    D = S["domains"]["spain"]
    out_dir = ROOT / "data" / "processed" / "spain"
    gdf = gpd.read_file(ROOT / D["catchments"])
    hy_f = out_dir / "hypsometry.csv"
    if hy_f.exists() and set(pd.read_csv(hy_f).gauge_id) == set(gdf.gauge_id):
        hy = pd.read_csv(hy_f)
        log("   hypsometry cached")
    else:
        hy = hypsometry_batched(gdf)
        hy.to_csv(hy_f, index=False)
    res = cedex_reservoirs()
    j = gpd.sjoin(res.to_crs("EPSG:3035").assign(geometry=lambda x: x.buffer(300)),
                  gdf[["gauge_id", "geometry"]].to_crs("EPSG:3035"), predicate="intersects")
    up = j.groupby("gauge_id")["nom_embalse"].apply(lambda s: "|".join(sorted(set(map(str.strip, s)))))
    att = gdf.drop(columns="geometry")
    att["elev_median_m"] = att.gauge_id.map(hy.set_index("gauge_id")["p50"])
    att["elev_max_m"] = att.gauge_id.map(hy.set_index("gauge_id")["p100"])
    att["reservoirs_upstream"] = att.gauge_id.map(up).fillna("")
    att["n_reservoirs_upstream"] = att.gauge_id.map(j.groupby("gauge_id").size()).fillna(0).astype(int)
    att["regulated"] = att.n_reservoirs_upstream > 0
    if google:
        names = google_static_names()
        fc_ids = set(pd.read_csv(out_dir / "gr4j" / "parameters.csv").gauge_id)
        sub = gdf[gdf.gauge_id.isin(fc_ids)]
        log(f"   Google statics for {len(sub)} catchments: HydroATLAS level-12 (tiled)")
        ha = hydroatlas(sub, names, tile_deg=2.0)
        log("   Caravan climate indices (ERA5-Land 1981-2020)")
        cols = ["date", "gauge_id", "total_precipitation", "potential_evaporation", "temperature_2m"]
        f = pd.read_parquet(ROOT / D["forcing"], columns=cols)
        ci = climate_indices(f[f.gauge_id.isin(fc_ids)])
        att = att.merge(ha, on="gauge_id", how="left").merge(ci, on="gauge_id", how="left")
        miss = [n for n in names if n not in att or att.loc[att.gauge_id.isin(fc_ids), n].isna().any()]
        if miss:
            raise RuntimeError(f"missing Google static attributes: {miss}")
    att.to_csv(out_dir / "catchment_attributes.csv", index=False)
    log(f"-> spain/catchment_attributes.csv: {len(att)} catchments, {int(att.regulated.sum())} regulated "
        f"({len(res)} CEDEX reservoirs)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    ap.add_argument("--hypsometry-only", action="store_true",
                    help="only the elevation quantiles (all GR4J needs)")
    ap.add_argument("--google", action="store_true", help="Spain: also the 84 Google model statics")
    args = ap.parse_args()
    init_ee()
    if args.domain == "spain":
        return spain_main(args.google)
    gdf = gpd.read_file(P.catchments)
    names = google_static_names()
    log(f"{len(gdf)} catchments, {len(names)} Google static attributes")

    log("1. hypsometry (SRTM)")
    if HYPSO.exists() and set(pd.read_csv(HYPSO).gauge_id) == set(gdf.gauge_id):
        hy = pd.read_csv(HYPSO)
        log("   cached")
    else:
        hy = hypsometry(gdf)
        hy.to_csv(HYPSO, index=False)
    if args.hypsometry_only:
        return

    log("2. HydroATLAS level-12")
    ha = hydroatlas(gdf, names)

    log("3. Caravan climate indices (ERA5-Land 1981-2020)")
    ci = climate_indices(pd.read_parquet(P.forcing))

    log("4. regulation")
    res = reservoirs_in(gdf)

    att = (gdf.drop(columns="geometry").merge(ha, on="gauge_id", how="left").merge(ci, on="gauge_id", how="left"))
    att["elev_median_m"] = att.gauge_id.map(hy.set_index("gauge_id")["p50"])
    att["elev_max_m"] = att.gauge_id.map(hy.set_index("gauge_id")["p100"])
    att["reservoirs_upstream"] = att.gauge_id.map(res).fillna("")
    att["regulated"] = att.reservoirs_upstream != ""       # see docstring: dor_pc_pva not used
    missing = [n for n in names if n not in att or att[n].isna().any()]
    if missing:
        raise RuntimeError(f"missing Google static attributes: {missing}")
    att.to_csv(P.attributes, index=False)
    log(f"-> {P.attributes.relative_to(ROOT)}  ({int(att.regulated.sum())} regulated, "
        f"{int((~att.regulated).sum())} near-natural)")
    log(att[["gauge_id", "area_km2", "elev_median_m", "p_mean", "pet_mean_ERA5_LAND", "aridity_ERA5_LAND",
             "frac_snow", "dor_pc_pva", "reservoirs_upstream"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
