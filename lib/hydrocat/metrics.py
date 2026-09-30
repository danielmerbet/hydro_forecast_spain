"""Skill metrics for daily streamflow. All ignore days where obs or sim is NaN."""
from __future__ import annotations

import numpy as np


def _clean(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    if len(o) < 2 or np.var(o) == 0:
        return np.nan
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def kge_components(obs, sim):
    """Kling-Gupta efficiency (Gupta et al., 2009) and its components r, alpha, beta."""
    o, s = _clean(obs, sim)
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


def kge(obs, sim) -> float:
    return kge_components(obs, sim)[0]


def pbias(obs, sim) -> float:
    """Percent bias: + means the model over-estimates volume."""
    o, s = _clean(obs, sim)
    return 100.0 * (s.sum() - o.sum()) / o.sum() if len(o) and o.sum() > 0 else np.nan


def nse_log(obs, sim) -> float:
    """NSE on log flows (low-flow skill). A small offset avoids log(0)."""
    o, s = _clean(obs, sim)
    if len(o) < 2:
        return np.nan
    eps = max(0.01 * o.mean(), 1e-6)
    return nse(np.log(o + eps), np.log(np.clip(s, 0, None) + eps))


def peak_error(obs, sim, q: float = 0.99) -> float:
    """Relative error (%) in the mean of flows above the obs q-quantile (on obs-peak days)."""
    o, s = _clean(obs, sim)
    if len(o) < 50:
        return np.nan
    m = o >= np.quantile(o, q)
    return 100.0 * (s[m].mean() - o[m].mean()) / o[m].mean()


def all_metrics(obs, sim) -> dict:
    k, r, a, b = kge_components(obs, sim)
    o, _ = _clean(obs, sim)
    return dict(n_days=len(o), NSE=nse(obs, sim), KGE=k, r=r, alpha=a, beta=b,
                PBIAS=pbias(obs, sim), logNSE=nse_log(obs, sim), peak_err_pct=peak_error(obs, sim))
