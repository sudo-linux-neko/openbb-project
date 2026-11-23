# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Institution-grade bootstrap engine.

- Stationary bootstrap (Politis-Romano 1994): geometric block lengths, mean b,
  circular wrap -> stationary pseudo-series.
- Circular block bootstrap (Politis-Romano 1992): fixed length l, wrap.
- Default b=21 (monthly vol clustering, T^{1/3} rule: T~7500 -> T^{1/3}~19.5).
  Sensitivity {10, 21, 63} always reported.
- Reproducible: numpy SeedSequence + PCG64, per-replicate spawn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


def _draw_stationary(r: np.ndarray, b: float, rng: np.random.Generator) -> np.ndarray:
    T = len(r)
    p = 1.0 / b
    out = np.empty(T, dtype=float)
    t = 0
    while t < T:
        start = int(rng.integers(0, T))
        L = int(rng.geometric(p))
        for k in range(L):
            if t >= T:
                break
            out[t] = r[(start + k) % T]
            t += 1
    return out


def _draw_circular(r: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    T = len(r)
    out = np.empty(T, dtype=float)
    t = 0
    while t < T:
        start = int(rng.integers(0, T))
        for k in range(min(block, T - t)):
            out[t] = r[(start + k) % T]
            t += 1
    return out


def _stats_from_path(path: np.ndarray) -> dict:
    r = path
    n = len(r)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mu / sd * np.sqrt(252) if sd > 0 else 0.0
    eq = np.cumprod(1 + r)
    cagr = float(eq[-1] ** (252 / n) - 1) if eq[-1] > 0 else -1.0
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq / peak - 1.0))
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {"Sharpe": sharpe, "CAGR": cagr, "MaxDD": mdd, "Calmar": calmar,
            "TotalRet": float(eq[-1] - 1)}


def run_bootstrap(returns: pd.Series, B: int = 10000, b: float = 21.0,
                  method: str = "stationary", seed: int = 42,
                  bench_sharpe: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (replicates DataFrame, summary dict with CI + P(SR<0))."""
    r = returns.dropna().to_numpy(dtype=float)
    T = len(r)
    if T < 500:
        raise ValueError(f"need >=500 obs, got {T}")
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(B)
    rows = []
    for i in tqdm(range(B), desc=f"Bootstrap {method} b={b}", unit="rep"):
        rng = np.random.default_rng(child_seeds[i])
        path = _draw_stationary(r, b, rng) if method == "stationary" else _draw_circular(r, int(b), rng)
        rows.append(_stats_from_path(path))
    df = pd.DataFrame(rows)
    q = df.quantile([0.025, 0.5, 0.975])
    summary = {
        "method": method, "B": B, "b": b, "T": T, "seed": seed,
        "Sharpe_CI95": [float(q.loc[0.025, "Sharpe"]), float(q.loc[0.975, "Sharpe"])],
        "Sharpe_med": float(q.loc[0.5, "Sharpe"]),
        "P_Sharpe_below_0": float((df["Sharpe"] < 0).mean()),
        "MaxDD_med": float(q.loc[0.5, "MaxDD"]),
        "MaxDD_CI95": [float(q.loc[0.025, "MaxDD"]), float(q.loc[0.975, "MaxDD"])],
        "CAGR_med": float(q.loc[0.5, "CAGR"]),
        "CAGR_CI95": [float(q.loc[0.025, "CAGR"]), float(q.loc[0.975, "CAGR"])],
    }
    if bench_sharpe is not None:
        summary["P_Sharpe_below_bench"] = float((df["Sharpe"] < bench_sharpe).mean())
        summary["bench_sharpe"] = float(bench_sharpe)
    return df, summary


def autocorr_block_check(returns: pd.Series) -> dict:
    """Data-driven block check: first lag where |acf| of r and |r| dies."""
    r = returns.dropna().to_numpy(dtype=float)
    T = len(r)
    se = 1.96 / np.sqrt(T)
    out = {}
    for name, x in (("ret", r), ("absret", np.abs(r - r.mean()))):
        xc = x - x.mean()
        acf = np.correlate(xc, xc, mode="full")[T - 1:] / (np.dot(xc, xc) + 1e-12)
        Lstar = None
        for L in range(1, min(120, T // 4)):
            if all(abs(acf[L + j]) < se for j in range(3)):
                Lstar = L
                break
        out[name] = {"Lstar": Lstar, "suggested_b": int(np.clip(2 * (Lstar or 10), 10, 63))}
    out["T_cuberoot_rule"] = float(round(T ** (1 / 3) * 1.5, 1))
    return out
