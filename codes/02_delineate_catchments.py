#!/usr/bin/env python3
"""
02 — Delineate the catchment upstream of every ACA gauge.

DATA
  * Flow directions: HydroSHEDS v1, 3 arc-second (~90 m) void-filled D8
    directions (Lehner et al., 2008), read from Earth Engine (WWF/HydroSHEDS/03DIR)
    so no manual download is needed. ESRI D8 encoding (1=E, 2=SE, 4=S ... 128=NE).
  * Official drained areas: ACA's SDIM2 catalogue
    (aplicacions.aca.gencat.cat/sdim2/apirest/catalog?componentType=aforament),
    field "Superfície conca drenada". Matched to our gauges by UTM coordinates.

HOW GAUGES ARE SNAPPED TO THE RIVER NETWORK
  A gauge's reported coordinates rarely fall exactly on the 90 m model river,
  and snapping to "the biggest river nearby" is wrong at confluences (the
  Onyar gauge EA020 in Girona is 1.7 km from the Ter gauge EA010). So:
    1. candidate cells = all cells within `snap_radius_m` of the gauge
       (`snap_radius_official_m` when an official area is available, because
       area matching then protects against jumping to the wrong river);
    2. if ACA publishes the drained area A, pick the candidate whose upstream
       area best matches A (in log space), with a small distance penalty;
    3. otherwise pick the nearest candidate with upstream area >= min_area_km2.
  The ratio (delineated / official area) is written for every gauge, and a
  gauge is flagged `area_ok = False` if it is off by more than 20 %. Manual
  corrections can be given in config/gauge_overrides.csv (columns gauge_id,
  lon, lat), which replaces the gauge location before snapping.

Writes
  (paths per domain in settings.yaml `domains`; --domain catalonia | spain)
  data/interim/hydrosheds_03dir*.tif       flow directions for the domain
  data/processed[/spain]/catchments.gpkg   one polygon per gauge (EPSG:4326) with area,
                                           official area, snapping diagnostics
  outputs/figures/02_catchments_map.png    overview map (Catalonia)

  Spain: gauges and official areas (suprest) come from the CEDEX table of
  script 01b. The river network index is built once, so ~1,200 catchments on
  the ~1.5e8-cell Spanish grid take minutes, not hours.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from pyproj import Transformer
from rasterio.merge import merge
from shapely.geometry import shape
from shapely.ops import unary_union

from hydrocat.config import CONFIG, FIGURES, INTERIM, RAW, ROOT, P, load_settings
from hydrocat.eeutils import init_ee, retry

S = load_settings()
C = S["catchments"]
RES = 1.0 / 1200.0                     # 3 arc-seconds in degrees
SDIM_URL = "https://aplicacions.aca.gencat.cat/sdim2/apirest/catalog?componentType=aforament"
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)  # ESRI D8: N, NE, E, SE, S, SW, W, NW


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. Flow directions from Earth Engine (tiled download, mosaicked locally)
# ---------------------------------------------------------------------------
def download_flowdir(bbox, out, tdir):
    if out.exists():
        log(f"  cached: {out.relative_to(ROOT)}")
        return
    init_ee()
    # NB: .unmask() would drop the native 3" projection; masked cells are cleaned locally instead
    img = ee.Image(C["flowdir_asset"]).select(0).toUint8()
    x0, y0, x1, y1 = bbox
    tiles, step = [], 0.5
    tdir.mkdir(parents=True, exist_ok=True)
    for lx in np.arange(x0, x1, step):
        for ly in np.arange(y0, y1, step):
            bx1, by1 = min(lx + step, x1), min(ly + step, y1)
            f = tdir / f"dir_{lx:.2f}_{ly:.2f}.tif"
            if not f.exists():
                # The tile is defined by its pixel grid alone (origin + dimensions).
                # A region rectangle west of Greenwich was read as wrapping the
                # globe (Earth Engine error "Pixel grid dimensions (227520x600)").
                c0, r0 = round(float(lx) / RES), round(float(by1) / RES)
                wpx, hpx = round((float(bx1) - float(lx)) / RES), round((float(by1) - float(ly)) / RES)
                url = retry(img.getDownloadURL, {
                    "crs": "EPSG:4326",
                    "crs_transform": [RES, 0, c0 * RES, 0, -RES, r0 * RES],
                    "dimensions": f"{wpx}x{hpx}",
                    "format": "GEO_TIFF"})
                with urllib.request.urlopen(url, timeout=300) as r:
                    data = r.read()
                if data[:2] == b"PK":            # some EE versions zip single files
                    data = zipfile.ZipFile(io.BytesIO(data)).read(
                        [n for n in zipfile.ZipFile(io.BytesIO(data)).namelist() if n.endswith(".tif")][0])
                f.write_bytes(data)
            tiles.append(f)
    log(f"  mosaicking {len(tiles)} tiles")
    srcs = [rasterio.open(t) for t in tiles]
    arr, tr = merge(srcs, nodata=255)
    prof = srcs[0].profile | dict(height=arr.shape[1], width=arr.shape[2], transform=tr,
                                  nodata=255, compress="deflate", BIGTIFF="IF_SAFER")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr)
    for s_ in srcs:
        s_.close()


# ---------------------------------------------------------------------------
# 2. D8 accumulation in km2 (numba, no external GIS dependency)
# ---------------------------------------------------------------------------
from numba import njit  # noqa: E402

_DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
_DC = np.array([0, 1, 1, 1, 0, -1, -1, -1])
_CODES = np.array(DIRMAP)


@njit(cache=True)
def _downstream(fdir):
    nr, nc = fdir.shape
    down = -np.ones(nr * nc, dtype=np.int32)          # int32: Spain's grid is ~1.5e8 cells
    for r in range(nr):
        for c in range(nc):
            d = fdir[r, c]
            for k in range(8):
                if d == _CODES[k]:
                    rr, cc = r + _DR[k], c + _DC[k]
                    if 0 <= rr < nr and 0 <= cc < nc and fdir[rr, cc] != 255:
                        down[r * nc + c] = rr * nc + cc
                    break
    return down


@njit(cache=True)
def _accumulate(down, w):
    n = down.shape[0]
    indeg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if down[i] >= 0:
            indeg[down[i]] += 1
    acc = w.copy()
    stack = np.empty(n, dtype=np.int32)
    top = 0
    for i in range(n):
        if indeg[i] == 0:
            stack[top] = i
            top += 1
    while top > 0:
        top -= 1
        i = stack[top]
        j = down[i]
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                stack[top] = j
                top += 1
    return acc


@njit(cache=True)
def _upstream_index(down):
    """CSR list of upstream neighbours of every cell, built ONCE for all gauges."""
    n = down.shape[0]
    cnt = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        if down[i] >= 0:
            cnt[down[i] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    ups = np.empty(cnt[n], dtype=np.int32)
    fill = cnt[:-1].copy()
    for i in range(n):
        j = down[i]
        if j >= 0:
            ups[fill[j]] = i
            fill[j] += 1
    return cnt, ups


@njit(cache=True)
def _upstream_cells(cnt, ups, outlet, n_expected):
    """Indices of all cells draining to `outlet` (the outlet included)."""
    out = np.empty(n_expected + 16, dtype=np.int64)
    out[0] = outlet
    n_out, head = 1, 0
    while head < n_out:
        i = out[head]
        head += 1
        for k in range(cnt[i], cnt[i + 1]):
            if n_out >= out.shape[0]:           # safety: grow
                new = np.empty(out.shape[0] * 2, dtype=np.int64)
                new[:n_out] = out[:n_out]
                out = new
            out[n_out] = ups[k]
            n_out += 1
    return out[:n_out]


# ---------------------------------------------------------------------------
# 3. Official drained areas (ACA SDIM2 catalogue)
# ---------------------------------------------------------------------------
def sdim_catalogue() -> pd.DataFrame:
    f = RAW / "aca_sdim" / "catalog_aforament.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SDIM_URL, timeout=120) as r:
            f.write_bytes(r.read())
    d = json.loads(f.read_text())
    rows = []
    for p in d["providers"]:
        for s in p["sensors"]:
            a = s.get("componentAdditionalInfo", {})
            area = a.get("Superfície conca drenada")
            if not a.get("Coordenada X (UTM ETRS89)"):
                continue
            rows.append(dict(
                sdim_component=s["component"], sdim_name=s["componentDesc"],
                x=float(a["Coordenada X (UTM ETRS89)"]), y=float(a["Coordenada Y (UTM ETRS89)"]),
                area_official_km2=float(re.sub(r"[^\d,\.]", "", area).replace(".", "").replace(",", "."))
                if area and re.search(r"\d", area) else np.nan))
    return pd.DataFrame(rows).drop_duplicates("sdim_component")


def attach_official_area(st: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in st.iterrows():
        d = np.hypot(cat.x - g.utm_x, cat.y - g.utm_y)
        i = d.idxmin()
        ok = d[i] < 250
        out.append(dict(gauge_id=g.gauge_id,
                        sdim_component=cat.sdim_component[i] if ok else None,
                        area_official_km2=cat.area_official_km2[i] if ok else np.nan,
                        sdim_match_dist_m=round(float(d[i]), 1)))
    return st.merge(pd.DataFrame(out), on="gauge_id")


def _other_river(used, g) -> bool:
    """Is an already-used outlet cell on a DIFFERENT river than gauge g?
    Same river if the names match (ACA), else if the official drained areas
    agree within 20 % (CEDEX has no river names; the same site is sometimes
    listed twice for different periods, e.g. Teruel 8015 / 8027)."""
    if used is None:
        return False
    river, area = used
    if river and g.river:
        return river != g.river
    if np.isfinite(area) and np.isfinite(g.area_official_km2):
        return abs(np.log(area / g.area_official_km2)) > np.log(1.2)
    return True


# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="catalonia", choices=list(S["domains"]))
    args = ap.parse_args()
    D = S["domains"][args.domain]
    flowdir = ROOT / D["flowdir"]
    out_gpkg = ROOT / D["catchments"]

    log(f"1. flow directions (HydroSHEDS 03DIR via Earth Engine), domain {args.domain}")
    download_flowdir(D["bbox"], flowdir, INTERIM / f"flowdir_tiles_{args.domain}")
    with rasterio.open(flowdir) as src:
        fdir = src.read(1)
        tr = src.transform
    fdir = np.where(np.isin(fdir, DIRMAP), fdir, 255).astype(np.uint8)   # sea / no data -> 255
    nr, nc = fdir.shape
    log(f"   grid {nr} x {nc}")

    log("2. accumulation (km2) + upstream index")
    lat_c = tr.f + tr.e * (np.arange(nr) + 0.5)
    cell_km2 = (6371.0088 * np.deg2rad(RES)) ** 2 * np.cos(np.deg2rad(lat_c))
    w = np.repeat(cell_km2.astype(np.float32)[:, None], nc, axis=1).ravel()
    w[fdir.ravel() == 255] = 0.0
    down = _downstream(fdir)
    acc = _accumulate(down, w).reshape(nr, nc)
    del w
    cnt, ups = _upstream_index(down)

    log("3. gauges + official areas")
    if args.domain == "catalonia":
        st = pd.read_csv(P.stations)
        st = attach_official_area(st, sdim_catalogue())
    else:  # CEDEX table already carries lon/lat and the official drained area (suprest)
        st = pd.read_csv(ROOT / D["stations"])
        st["river"] = ""                        # CEDEX has no river names -> area-based outlet rule
        st["basin"] = st["demarcacion"]
        st["sdim_component"] = None
    ov_file = CONFIG / "gauge_overrides.csv"
    if ov_file.exists():
        ov = pd.read_csv(ov_file, comment="#").set_index("gauge_id")
        for gid, r in ov.iterrows():
            m = st.gauge_id == gid
            if m.any():
                st.loc[m, ["lon", "lat"]] = r.lon, r.lat
                log(f"   override location for {gid}")

    log("4. snap + delineate")
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)   # equal-area metres, all Europe
    geoms, rows, used = [], [], {}
    # gauges with an official area first: they are the most reliable anchors
    st = st.sort_values("area_official_km2", na_position="last")
    for _, g in st.iterrows():
        rad =C["snap_radius_official_m"] if np.isfinite(g.area_official_km2) else C["snap_radius_m"]
        r0 = int((g.lat - tr.f) / tr.e)
        c0 = int((g.lon - tr.c) / tr.a)
        k = int(np.ceil(rad / 60)) + 1
        rr, cc = np.mgrid[max(r0 - k, 0):min(r0 + k + 1, nr), max(c0 - k, 0):min(c0 + k + 1, nc)]
        lon_c = tr.c + (cc + 0.5) * tr.a
        lat_c2 = tr.f + (rr + 0.5) * tr.e
        xm, ym = to_m.transform(lon_c, lat_c2)
        gx, gy = to_m.transform(g.lon, g.lat)
        dist = np.hypot(xm - gx, ym - gy)
        a = acc[rr, cc]
        valid = (dist <= rad) & (a >= C["min_area_km2"])
        if not valid.any():
            log(f"   {g.gauge_id}: no river cell >= {C['min_area_km2']} km2 within {rad} m — skipped")
            continue
        A = g.area_official_km2
        if np.isfinite(A):
            score = np.abs(np.log(np.where(valid, a, np.nan) / A)) + 0.1 * dist / rad
        else:
            score = np.where(valid, dist, np.nan)
        # An outlet cell already used by a gauge on ANOTHER river is excluded
        # (EA094 / EA095 are two different streams 1.3 km apart at Vilada).
        taken = np.array([_other_river(used.get(int(x)), g) for x in (rr * nc + cc).ravel()])
        score = np.where(taken.reshape(score.shape), np.nan, score)
        if np.all(np.isnan(score)):
            log(f"   {g.gauge_id}: every candidate cell is used by another river — skipped")
            continue
        i = np.nanargmin(score)
        r, c = rr.ravel()[i], cc.ravel()[i]
        used[int(r * nc + c)] = (g.river, g.area_official_km2)
        cells = _upstream_cells(cnt, ups, r * nc + c, int(acc[r, c] / cell_km2[r] * 1.2) + 10)
        rows_i, cols_i = cells // nc, cells % nc
        r_lo, r_hi, c_lo, c_hi = rows_i.min(), rows_i.max() + 1, cols_i.min(), cols_i.max() + 1
        sub = np.zeros((r_hi - r_lo, c_hi - c_lo), dtype=np.uint8)
        sub[rows_i - r_lo, cols_i - c_lo] = 1
        sub_tr = rasterio.Affine(tr.a, 0, tr.c + c_lo * tr.a, 0, tr.e, tr.f + r_lo * tr.e)
        polys = [shape(p) for p, v in rasterio.features.shapes(sub, mask=sub == 1, transform=sub_tr) if v == 1]
        geom = unary_union(polys)
        area = float(acc[r, c])
        ratio = area / A if np.isfinite(A) else np.nan
        geoms.append(geom)
        rows.append(dict(gauge_id=g.gauge_id, name=g["name"], river=g.river, basin=g.basin,
                         gauge_lon=g.lon, gauge_lat=g.lat,
                         outlet_lon=round(tr.c + (c + 0.5) * tr.a, 6),
                         outlet_lat=round(tr.f + (r + 0.5) * tr.e, 6),
                         snap_dist_m=round(float(dist.ravel()[i]), 1),
                         area_km2=round(area, 2), area_official_km2=A,
                         area_ratio=round(ratio, 3) if np.isfinite(ratio) else np.nan,
                         area_ok=bool(abs(ratio - 1) <= 0.2) if np.isfinite(ratio) else True,
                         sdim_component=g.sdim_component))
        flag = "" if rows[-1]["area_ok"] else "   <-- CHECK"
        log(f"   {g.gauge_id:7s} {area:9.1f} km2  official {A:9.1f}  snap {dist.ravel()[i]:5.0f} m{flag}")

    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs="EPSG:4326")
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    n_bad = int((~gdf.area_ok).sum())
    log(f"   {len(gdf)} catchments -> {out_gpkg.relative_to(ROOT)}  ({n_bad} flagged area mismatch > 20 %)")
    if args.domain != "catalonia":
        return                                   # Spain-wide map: script 02b

    fig, ax = plt.subplots(figsize=(9, 8))
    gdf.sort_values("area_km2", ascending=False).plot(ax=ax, column="area_km2", cmap="viridis",
                                                      alpha=0.35, edgecolor="k", linewidth=0.4)
    ax.scatter(gdf.gauge_lon, gdf.gauge_lat, s=12, c=np.where(gdf.area_ok, "tab:blue", "tab:red"), zorder=3)
    for _, r in gdf.iterrows():
        ax.annotate(r.gauge_id, (r.gauge_lon, r.gauge_lat), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_title("ACA gauges and delineated catchments (red = area mismatch > 20 %)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_catchments_map.png", dpi=180)


if __name__ == "__main__":
    main()
