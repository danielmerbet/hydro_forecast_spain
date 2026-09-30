#!/usr/bin/env python3
"""
02b — Figure 2: every Spanish river gauge with daily discharge data.

  * CEDEX Anuario de Aforos (script 01b): all ROEA/SAIH gauges of the
    inter-regional basin districts + Galicia-Costa, split into
      usable  = at least 5 years of daily data since 2003-10 (enough to
                calibrate after the 5-year spin-up), and
      other   = shorter / older records;
  * ACA (script 01): Catalonia's internal basins, modelled gauges.
  * If Spain-wide catchments exist (data/processed/spain/catchments.gpkg), they
    are drawn underneath.

Country outlines: US Department of State LSIB (Earth Engine USDOS/LSIB_SIMPLE/2017),
cached in data/raw/boundaries/.

Writes outputs/figures/02_gauges_spain.png
"""
from __future__ import annotations

import ee
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from hydrocat.config import FIGURES, PROCESSED, RAW, P
from hydrocat.eeutils import init_ee, retry

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_CEDEX, C_ACA = "#2a78d6", "#eb6834"
mpl.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def countries():
    f = RAW / "boundaries" / "iberia_lsib.gpkg"
    if not f.exists():
        init_ee()
        fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
            ee.Filter.inList("country_na", ["Spain", "Portugal", "France", "Andorra", "Morocco"]))
        g = retry(ee.data.computeFeatures, {"expression": fc.select(["country_na"]),
                                            "fileFormat": "GEOPANDAS_GEODATAFRAME"})
        f.parent.mkdir(parents=True, exist_ok=True)
        g.set_crs("EPSG:4326").to_file(f)
    return gpd.read_file(f)


def main():
    cedex = pd.read_csv(PROCESSED / "spain" / "stations_cedex.csv")
    aca = pd.read_csv(P.stations)
    aca = aca[aca.modelled]
    usable = cedex.n_days_2003_2022 >= 1825

    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_facecolor("#f4f7fa")
    c = countries()
    c.plot(ax=ax, color="white", edgecolor=GRID, lw=0.7)
    c[c.country_na == "Spain"].plot(ax=ax, color="white", edgecolor=INK2, lw=0.8)
    cat_f = PROCESSED / "spain" / "catchments.gpkg"
    if cat_f.exists():
        gpd.read_file(cat_f).sort_values("area_km2", ascending=False).plot(
            ax=ax, facecolor="none", edgecolor=C_CEDEX, lw=0.25, alpha=0.5)
    gpd.read_file(P.catchments).plot(ax=ax, facecolor="none", edgecolor=C_ACA, lw=0.3, alpha=0.6)

    ax.scatter(cedex.lon[~usable], cedex.lat[~usable], s=9, facecolor="white", edgecolor="#9a9994", lw=0.6,
               label=f"CEDEX gauge, < 5 yr of data since 2003 ({int((~usable).sum())})", zorder=3)
    ax.scatter(cedex.lon[usable], cedex.lat[usable], s=10, color=C_CEDEX, edgecolor="white", lw=0.3,
               label=f"CEDEX gauge, ≥ 5 yr since 2003 — calibratable ({int(usable.sum())})", zorder=4)
    ax.scatter(aca.lon, aca.lat, s=12, color=C_ACA, edgecolor="white", lw=0.3,
               label=f"ACA gauge (Catalan internal basins), modelled ({len(aca)})", zorder=5)
    for dem, d in cedex.groupby("demarcacion"):
        ax.annotate(dem, (d.lon.median(), d.lat.median()), fontsize=8.5, color=INK, ha="center",
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.annotate("Conques Internes\n(ACA)", (aca.lon.median(), aca.lat.median() - 0.35), fontsize=8.5,
                color=INK, ha="center", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(-9.6, 3.6)
    ax.set_ylim(35.8, 44.0)
    ax.set_aspect(1 / 0.77)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.set_title(f"Figure 2 — River gauges with daily discharge in mainland Spain "
                 f"({len(cedex) + len(aca)} gauges)", loc="left", color=INK, fontsize=11)
    ax.text(0.005, -0.06, "Sources: CEDEX Anuario de Aforos 2021-22 (history to 2022-09-30); ACA open data "
            "and export (to present). Recent/live data for CEDEX gauges come from each SAIH (script 01c).",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.savefig(FIGURES / "02_gauges_spain.png", dpi=170, bbox_inches="tight")
    print("-> outputs/figures/02_gauges_spain.png")


if __name__ == "__main__":
    main()
