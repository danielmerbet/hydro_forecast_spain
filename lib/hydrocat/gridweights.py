"""Area-weighted catchment means of gridded fields (reanalysis or forecasts).

Every gridded product used in the pipeline (ERA5-Land 0.1 deg, WeatherNext 3
0.1 deg, AIFS/IFS 0.25 deg) is turned into catchment means the same way:

1. `catchment_weights` builds, once per grid, a sparse matrix W
   (n_catchments x n_cells) whose entries are the fraction of each catchment's
   area that falls in each grid cell. Fractions are computed by rasterising the
   polygons on a grid `oversample` times finer than the product (default 10:
   ~1 km for a 0.1 deg grid) and weighting sub-cells by cos(latitude), so they
   are area fractions, not cell counts.
2. `catchment_means` applies W to a stack of fields (time, lat, lon). Cells
   with missing values (sea, product mask) are dropped and the remaining
   weights renormalised, day by day.

A catchment smaller than one sub-cell still gets the cell containing its
representative point.
"""
from __future__ import annotations

import numpy as np
import rasterio.features
from affine import Affine
from scipy import sparse


def catchment_weights(gdf, transform: Affine, shape: tuple[int, int], oversample: int = 10,
                      id_col: str = "gauge_id"):
    """Sparse (n_catchments x ny*nx) area-fraction matrix, rows summing to 1."""
    ny, nx = shape
    fine_tr = transform * Affine.scale(1 / oversample)
    fy, fx = ny * oversample, nx * oversample
    lat_f = fine_tr.f + fine_tr.e * (np.arange(fy) + 0.5)
    coslat = np.cos(np.deg2rad(lat_f)).astype(np.float64)
    rows, cols, vals = [], [], []
    for i, geom in enumerate(gdf.geometry):
        x0, y0, x1, y1 = geom.bounds
        # rasterise only the window around the polygon
        c0 = max(int(np.floor((x0 - fine_tr.c) / fine_tr.a)) - 1, 0)
        c1 = min(int(np.ceil((x1 - fine_tr.c) / fine_tr.a)) + 1, fx)
        r0 = max(int(np.floor((y1 - fine_tr.f) / fine_tr.e)) - 1, 0)
        r1 = min(int(np.ceil((y0 - fine_tr.f) / fine_tr.e)) + 1, fy)
        win_tr = fine_tr * Affine.translation(c0, r0)
        m = rasterio.features.rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0), transform=win_tr,
                                        all_touched=False, dtype="uint8").astype(bool)
        if not m.any():
            p = geom.representative_point()
            cc = int((p.x - transform.c) / transform.a)
            rr = int((p.y - transform.f) / transform.e)
            rows.append(i); cols.append(rr * nx + cc); vals.append(1.0)
            continue
        rr_f, cc_f = np.nonzero(m)
        w = coslat[rr_f + r0]
        coarse = ((rr_f + r0) // oversample) * nx + (cc_f + c0) // oversample
        uniq, inv = np.unique(coarse, return_inverse=True)
        acc = np.bincount(inv, weights=w)
        rows += [i] * len(uniq); cols += list(uniq); vals += list(acc / acc.sum())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(gdf), ny * nx))


def catchment_means(W, fields: np.ndarray) -> np.ndarray:
    """fields: (time, ny, nx) -> (time, n_catchments); NaN/inf cells excluded, weights renormalised."""
    t = fields.shape[0]
    x = fields.reshape(t, -1).astype(np.float64)
    valid = np.isfinite(x)
    x = np.where(valid, x, 0.0)
    num = (W @ x.T).T
    den = (W @ valid.T.astype(np.float64)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
