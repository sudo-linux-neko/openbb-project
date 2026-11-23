# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Calibrate GBM / Heston / Merton / Regime params from strategy log-returns.

Moment-matched (fast, disclosed approximations — not options-implied).
All annualized unless noted. dt = 1/252.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _logret(simple: pd.Series) -> pd.Series:
    return np.log1p(simple.clip(lower=-0.999))


def calibrate_all(strategy_simple_ret: pd.Series, spy_simple_ret: pd.Series | None = None) -> dict:
    r = strategy_simple_ret.dropna()
    lr = _logret(r)
    n = len(lr)
    mu_ann = float(lr.mean() * 252)
    var_ann = float(lr.var(ddof=1) * 252)
    sigma_ann = float(np.sqrt(var_ann))
    skew = float(lr.skew())
    kurt = float(lr.kurtosis())  # Fisher excess

    # --- Heston: theta/v0 from variance; xi from vol-of-vol; rho from leverage ---
    sigma_21d = r.rolling(21, min_periods=21).std() * np.sqrt(252)
    dsigma = sigma_21d.diff().dropna()
    xi = float(np.clip(2 * dsigma.std() * np.sqrt(252), 0.2, 0.8)) if len(dsigma) > 50 else 0.35
    rho = float(r.loc[dsigma.index].corr(dsigma)) if len(dsigma) > 50 else -0.6
    rho = float(np.clip(rho if np.isfinite(rho) else -0.6, -0.85, 0.2))
    heston = {"mu": mu_ann, "v0": var_ann, "theta": var_ann,
              "kappa": 2.0, "xi": xi, "rho": rho,
              "feller": float(2 * 2.0 * var_ann - xi ** 2)}

    # --- Merton: de-jumped vol + jump triple to match skew/kurt ---
    sigma_dj = float(r.clip(-0.03, 0.03).std() * np.sqrt(252))
    lam = float(np.clip(252 * (r.abs() > 0.04).mean(), 0.3, 3.0))
    left_tail = r[r < r.quantile(0.02)].mean() if len(r) > 100 else -0.04
    muJ = float(np.clip(np.log1p(max(left_tail, -0.15)) * 0.6, -0.08, -0.01))
    sigJ = 0.045
    merton = {"mu": mu_ann, "sigma": max(sigma_dj, 0.05), "lambda": lam,
              "muJ": muJ, "sigJ": sigJ}

    # --- Regime: label bear if 60d DD < -8% or 21d vol > 75th pct ---
    eq = (1 + r).cumprod()
    dd60 = eq / eq.rolling(60, min_periods=60).max() - 1.0
    vol21 = r.rolling(21, min_periods=21).std() * np.sqrt(252)
    bear = ((dd60 < -0.08) | (vol21 > vol21.quantile(0.75))).fillna(False)
    mus, sigs = {}, {}
    for name, mask in (("bull", ~bear), ("bear", bear)):
        sub = lr[mask.loc[lr.index]]
        mus[name] = float(sub.mean() * 252) if len(sub) > 50 else (0.10 if name == "bull" else -0.15)
        sigs[name] = float(sub.std(ddof=1) * np.sqrt(252)) if len(sub) > 50 else (0.14 if name == "bull" else 0.35)
    s = bear.astype(int)
    p11 = float(((s.shift(1) == 0) & (s == 0)).sum() / max((s.shift(1) == 0).sum(), 1))
    p22 = float(((s.shift(1) == 1) & (s == 1)).sum() / max((s.shift(1) == 1).sum(), 1))
    regime = {"mu1": mus["bull"], "sig1": sigs["bull"], "mu2": mus["bear"], "sig2": sigs["bear"],
              "p11": float(np.clip(p11, 0.9, 0.999)), "p22": float(np.clip(p22, 0.85, 0.995)),
              "bear_share": float(bear.mean())}
    gbm = {"mu": mu_ann, "sigma": sigma_ann}
    return {"gbm": gbm, "heston": heston, "merton": merton, "regime": regime,
            "diag": {"n": n, "skew_log": skew, "exkurt_log": kurt,
                     "sigma_total": sigma_ann, "sigma_dejumped": merton["sigma"]}}
