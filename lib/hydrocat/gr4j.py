"""GR4J rainfall-runoff model with the CemaNeige snow module (daily).

This is a line-by-line port of the reference implementation in the R package
airGR (INRAE; Coron et al., 2017), Fortran routines ``frun_GR4J`` and
``frun_CEMANEIGE``, compiled with numba. ``tests/test_gr4j_vs_airgr.py``
checks that it reproduces airGR's discharge to within 1e-6 mm/day.

References
----------
Perrin, C., Michel, C., Andreassian, V. (2003). Improvement of a parsimonious
    model for streamflow simulation. J. Hydrol. 279, 275-289.   (GR4J)
Valery, A., Andreassian, V., Perrin, C. (2014). "As simple as possible but not
    simpler": what is useful in a temperature-based snow-accounting routine?
    J. Hydrol. 517, 1166-1175.                                   (CemaNeige)

Parameters (units)
------------------
X1  production store capacity              [mm]
X2  groundwater exchange coefficient       [mm/day]
X3  routing store capacity                 [mm]
X4  unit-hydrograph time base              [day]
CTG snowpack thermal-state weighting coef. [-]
Kf  degree-day melt factor                 [mm/degC/day]

Elevation layers
----------------
As in airGR, the catchment is split into ``n_layers`` equal-area elevation
bands from its hypsometric curve (101 quantiles). Precipitation is
extrapolated with Valery's gradient (exp(0.00041 * dz), rescaled so the layer
mean equals the catchment input) and temperature with a lapse rate. airGR uses
a day-of-year lapse-rate table (Valery, 2010); here a constant
``LAPSE_TMEAN`` (the mean of that table) is used instead, a simplification
that only affects the few Pyrenean catchments where snow matters at all.
"""
from __future__ import annotations

import numpy as np
from numba import njit

NH = 20                  # airGR: max UH1 length (days); UH2 is 2*NH
GRADP = 0.00041          # Valery (2010) precipitation gradient [1/m]
LAPSE_TMEAN = 0.0055     # degC per m (mean of Valery 2010 daily table, ~0.43-0.66 degC/100 m)
LAPSE_TMIN = 0.0050
LAPSE_TMAX = 0.0060


# -----------------------------------------------------------------------------
# Unit hydrographs
# -----------------------------------------------------------------------------
@njit(cache=True)
def _ss1(i, c, d=2.5):
    if i <= 0:
        return 0.0
    if i < c:
        return (i / c) ** d
    return 1.0


@njit(cache=True)
def _ss2(i, c, d=2.5):
    if i <= 0:
        return 0.0
    if i < c:
        return 0.5 * (i / c) ** d
    if i < 2 * c:
        return 1.0 - 0.5 * (2.0 - i / c) ** d
    return 1.0


@njit(cache=True)
def _uh_ordinates(x4):
    o1 = np.zeros(NH)
    o2 = np.zeros(2 * NH)
    for i in range(1, NH + 1):
        o1[i - 1] = _ss1(i, x4) - _ss1(i - 1, x4)
    for i in range(1, 2 * NH + 1):
        o2[i - 1] = _ss2(i, x4) - _ss2(i - 1, x4)
    return o1, o2


# -----------------------------------------------------------------------------
# GR4J
# -----------------------------------------------------------------------------
@njit(cache=True)
def _qr_of(R, x3):
    rr = R / x3
    rr = rr * rr
    rr = rr * rr
    return R * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + rr)))


@njit(cache=True)
def gr4j(P, E, x1, x2, x3, x4, s_ini=0.3, r_ini=0.5, update_idx=-1, q_obs_upd=0.0):
    """Run GR4J. P, E: daily precipitation (liquid water input) and PET [mm].

    Returns Q [mm/day]. Initial stores follow airGR's defaults
    (production store 30 % full, routing store 50 % full, empty UHs).

    Forecast mode (update_idx >= 0): on day `update_idx` the routing store is
    reset so that simulated discharge equals the observation q_obs_upd [mm/day]
    ("routing store updating", as in the GRP forecasting model; Berthet et al.,
    2009, HESS 13, 819-831). The level R* is found by bisection such that the
    routing outflow QR(R*) = max(q_obs - QD, 0); the production store and the
    unit hydrographs are left untouched.
    """
    n = P.shape[0]
    Q = np.zeros(n)
    S = s_ini * x1
    R = r_ini * x3
    o1, o2 = _uh_ordinates(x4)
    uh1 = np.zeros(NH)
    uh2 = np.zeros(2 * NH)
    n1 = max(1, min(NH - 1, int(x4 + 1.0)))
    n2 = max(1, min(2 * NH - 1, 2 * int(x4 + 1.0)))

    for t in range(n):
        p1 = P[t]
        e = E[t]
        if p1 <= e:
            en = e - p1
            ws = min(en / x1, 13.0)
            tws = np.tanh(ws)
            sr = S / x1
            er = S * (2.0 - sr) * tws / (1.0 + (1.0 - sr) * tws)
            S = S - er
            pr = 0.0
        else:
            pn = p1 - e
            ws = min(pn / x1, 13.0)
            tws = np.tanh(ws)
            sr = S / x1
            ps = x1 * (1.0 - sr * sr) * tws / (1.0 + sr * tws)
            pr = pn - ps
            S = S + ps
        if S < 0.0:
            S = 0.0

        # percolation
        sr = S / x1
        sr = sr * sr
        sr = sr * sr
        perc = S * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + sr / 25.62891)))
        S = S - perc
        pr = pr + perc

        # split and convolution
        prhu1 = pr * 0.9
        prhu2 = pr * 0.1
        for k in range(n1):
            uh1[k] = uh1[k + 1] + o1[k] * prhu1
        uh1[NH - 1] = o1[NH - 1] * prhu1
        for k in range(n2):
            uh2[k] = uh2[k + 1] + o2[k] * prhu2
        uh2[2 * NH - 1] = o2[2 * NH - 1] * prhu2

        # exchange
        rr = R / x3
        exch = x2 * rr * rr * rr * np.sqrt(rr)

        # routing store
        R = R + uh1[0] + exch
        if R < 0.0:
            R = 0.0
        # direct flow
        qd = max(0.0, uh2[0] + exch)
        if t == update_idx:
            target = max(q_obs_upd - qd, 0.0)
            lo, hi = 0.0, 10.0 * x3
            while _qr_of(hi, x3) < target and hi < 1e6:
                hi *= 2.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _qr_of(mid, x3) < target:
                    lo = mid
                else:
                    hi = mid
            R = 0.5 * (lo + hi)
        qr = _qr_of(R, x3)
        R = R - qr
        Q[t] = qr + qd
    return Q


# -----------------------------------------------------------------------------
# CemaNeige (one call handles all layers; returns catchment-mean liquid input)
# -----------------------------------------------------------------------------
@njit(cache=True)
def cemaneige(Pl, Fsol, Tl, mean_an_solid, ctg, kf):
    """Pl, Fsol, Tl: (n_layers, n_days). Returns layer-mean (rain + melt) [mm/day]."""
    nl, n = Pl.shape
    out = np.zeros(n)
    gthr = 0.9 * mean_an_solid
    for l in range(nl):
        G = 0.0
        eTG = 0.0
        for t in range(n):
            pliq = (1.0 - Fsol[l, t]) * Pl[l, t]
            psol = Fsol[l, t] * Pl[l, t]
            G = G + psol
            eTG = ctg * eTG + (1.0 - ctg) * Tl[l, t]
            if eTG > 0.0:
                eTG = 0.0
            if eTG == 0.0 and Tl[l, t] > 0.0:
                potmelt = min(G, kf * Tl[l, t])
            else:
                potmelt = 0.0
            if G < gthr:
                gratio = G / gthr
            else:
                gratio = 1.0
            melt = (0.9 * gratio + 0.1) * potmelt
            G = G - melt
            out[t] += (pliq + melt) / nl
    return out


# -----------------------------------------------------------------------------
# Elevation layers (airGR DataAltiExtrapolation_Valery, constant lapse rates)
# -----------------------------------------------------------------------------
def layer_elevations(hypso101: np.ndarray, n_layers: int) -> np.ndarray:
    """Mean elevation of equal-area bands, exactly as airGR computes ZLayers."""
    h = np.asarray(hypso101, float)
    nmoy, nreste = 100 // n_layers, 100 % n_layers
    z, ncont = [], 0
    for _ in range(n_layers):
        nn = nmoy + 1 if nreste > 0 else nmoy
        nreste = max(nreste - 1, 0)
        if nn == 1:
            z.append(h[ncont])
        elif nn == 2:
            z.append(0.5 * (h[ncont] + h[ncont + 1]))
        else:
            z.append(h[ncont + nn // 2])
        ncont += nn
    return np.array(z)


def extrapolate_layers(P, T, Tmin, Tmax, hypso101, n_layers):
    """Return (P_layers, Fsol_layers, T_layers), each (n_layers, n_days)."""
    z_in = float(hypso101[50])          # forcing is a catchment mean -> median elevation
    zl = layer_elevations(hypso101, n_layers)
    P = np.asarray(P, float)
    zc = np.minimum(zl, 4000.0)
    pl = P[None, :] * np.exp(GRADP * (zc[:, None] - z_in))
    mean_l = pl.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pl = np.where(mean_l > 0, pl / mean_l * P, 0.0)
    dz = (z_in - zl)[:, None]
    tl = np.asarray(T, float)[None, :] + dz * LAPSE_TMEAN
    if z_in < 1500 and Tmin is not None and Tmax is not None:      # Hydrotel
        tmin = np.asarray(Tmin, float)[None, :] + dz * LAPSE_TMIN
        tmax = np.asarray(Tmax, float)[None, :] + dz * LAPSE_TMAX
        with np.errstate(invalid="ignore", divide="ignore"):
            fs = 1.0 - tmax / (tmax - tmin)
        fs = np.where(tmin >= 0, 0.0, fs)
        fs = np.where(tmax <= 0, 1.0, fs)
    else:                                                           # USACE
        fs = 1.0 - (tl - (-1.0)) / (3.0 - (-1.0))
        fs = np.where(tl > 3.0, 0.0, fs)
        fs = np.where(tl < -1.0, 1.0, fs)
    fs = np.clip(np.nan_to_num(fs, nan=0.0), 0.0, 1.0)
    return pl, fs, tl


class CatchmentModel:
    """GR4J + CemaNeige for one catchment, with inputs prepared once.

    >>> m = CatchmentModel(P, T, Tmin, Tmax, E, hypso101, n_layers=5)
    >>> q_mm = m.run([X1, X2, X3, X4, CTG, Kf])
    """

    def __init__(self, P, T, Tmin, Tmax, E, hypso101, n_layers=5):
        self.pl, self.fs, self.tl = extrapolate_layers(P, T, Tmin, Tmax, hypso101, n_layers)
        self.E = np.ascontiguousarray(E, dtype=float)
        # airGR: mean annual solid precipitation over the whole input series
        self.mean_an_solid = float(np.mean((self.fs * self.pl).mean(axis=0)) * 365.25)
        self.mean_an_solid = max(self.mean_an_solid, 1e-6)

    def liquid_input(self, ctg, kf):
        return cemaneige(self.pl, self.fs, self.tl, self.mean_an_solid, ctg, kf)

    def run(self, params, update_idx=-1, q_obs_upd=0.0):
        x1, x2, x3, x4, ctg, kf = params
        return gr4j(self.liquid_input(ctg, kf), self.E, x1, x2, x3, x4, 0.3, 0.5, update_idx, q_obs_upd)
