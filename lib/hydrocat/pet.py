"""Potential / reference evapotranspiration.

``fao56_penman_monteith`` is what the pipeline uses (GR4J's PET input): the
FAO-56 reference evapotranspiration (Allen et al., 1998, eq. 6) computed
entirely from ERA5-Land variables — net radiation, temperature, dewpoint,
surface pressure and 10 m wind speed — i.e. physically based, with no
temperature-only empiricism. It is the same quantity Caravan (v1.4+) provides
as ``potential_evaporation_sum_FAO_PENMAN_MONTEITH``, which replaced ERA5-Land's
own ``potential_evaporation`` because the latter is strongly biased high.

``pe_oudin`` (Oudin et al., 2005; temperature + extraterrestrial radiation)
is kept only for comparison / sensitivity tests.
"""
from __future__ import annotations

import numpy as np


def sat_vapour_pressure(t_c):
    """FAO-56 eq. 11, kPa."""
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def fao56_penman_monteith(tmin_c, tmax_c, tdew_c, rn_mj, pres_kpa, u10_ms):
    """Daily FAO-56 reference ET0 [mm/day].

    tmin_c, tmax_c : daily min / max 2 m air temperature [degC]
    tdew_c         : daily mean 2 m dewpoint [degC]  -> actual vapour pressure
    rn_mj          : daily net radiation at the surface [MJ m-2 day-1]
                     (ERA5-Land net shortwave + net longwave)
    pres_kpa       : surface pressure [kPa]
    u10_ms         : daily mean 10 m wind SPEED [m/s] (mean of hourly speeds)
    Soil heat flux G is 0 at the daily step (FAO-56 eq. 42).
    """
    tmin_c, tmax_c = np.asarray(tmin_c, float), np.asarray(tmax_c, float)
    t = 0.5 * (tmin_c + tmax_c)                                    # FAO-56 eq. 9
    es = 0.5 * (sat_vapour_pressure(tmax_c) + sat_vapour_pressure(tmin_c))   # eq. 12
    ea = sat_vapour_pressure(np.asarray(tdew_c, float))            # eq. 14
    delta = 4098.0 * sat_vapour_pressure(t) / (t + 237.3) ** 2      # eq. 13
    gamma = 0.000665 * np.asarray(pres_kpa, float)                  # eq. 8
    u2 = np.asarray(u10_ms, float) * 4.87 / np.log(67.8 * 10.0 - 5.42)   # eq. 47, 10 m -> 2 m
    num = 0.408 * delta * np.asarray(rn_mj, float) + gamma * 900.0 / (t + 273.0) * u2 * np.clip(es - ea, 0, None)
    den = delta + gamma * (1.0 + 0.34 * u2)
    return np.clip(num / den, 0.0, None)


def fao56_net_radiation(rs_mj, tmin_c, tmax_c, tdew_c, lat_deg, doy, elev_m, albedo=0.23):
    """Net radiation Rn = Rns - Rnl [MJ m-2 day-1] from incoming solar radiation,
    FAO-56 eqs. 37-40 (used when a product gives downward shortwave only, e.g.
    WeatherNext 3). Rso from eq. 37, Rnl from eq. 39 with actual vapour pressure
    from the dewpoint. Returns (Rns, -Rnl) so the second term has ERA5's sign
    convention (net thermal negative = loss)."""
    rs = np.asarray(rs_mj, float)
    ra = extraterrestrial_radiation(lat_deg, doy)
    rso = (0.75 + 2e-5 * np.asarray(elev_m, float)) * ra                       # eq. 37
    rns = (1 - albedo) * rs                                                     # eq. 38
    sigma = 4.903e-9
    ea = sat_vapour_pressure(np.asarray(tdew_c, float))
    ratio = np.clip(np.where(rso > 0, rs / rso, 0.5), 0.25, 1.0)
    rnl = (sigma * ((np.asarray(tmax_c) + 273.16) ** 4 + (np.asarray(tmin_c) + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * ratio - 0.35))              # eq. 39
    return rns, -rnl


def pressure_from_elevation(elev_m):
    """Atmospheric pressure [kPa] from elevation, FAO-56 eq. 7."""
    return 101.3 * ((293.0 - 0.0065 * np.asarray(elev_m, float)) / 293.0) ** 5.26


def extraterrestrial_radiation(lat_deg, doy):
    """FAO-56 eq. 21, MJ m-2 day-1."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))
    doy = np.asarray(doy)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                           + np.cos(phi) * np.cos(delta) * np.sin(ws))


def pe_oudin(tmean_c, lat_deg, doy):
    """Oudin et al. (2005), mm/day. For sensitivity tests only."""
    re = extraterrestrial_radiation(lat_deg, doy)
    t = np.asarray(tmean_c, float)
    return np.where(t + 5.0 > 0, re / (2.45 * 1000.0) * (t + 5.0) / 100.0 * 1000.0, 0.0)
