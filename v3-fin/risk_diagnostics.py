# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Institution-grade risk diagnostics: beta, turnover, regime split, drawdowns.

All trailing-only (no lookahead). Benchmark for beta = SPY total-return proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(strategy_ret: pd.Series, bench_ret: pd.Series,
                 window: int = 60) -> pd.Series:
    """OLS beta of strategy on benchmark over trailing window."""
    df = pd.DataFrame({"s": strategy_ret, "m": bench_ret}).dropna()
    cov = df["s"].rolling(window, min_periods=window).cov(df["m"])
    var = df["m"].rolling(window, min_periods=window).var()
    return (cov / var.replace(0, np.nan)).rename(f"beta_{window}d")


def rolling_vol(ret: pd.Series, window: int = 60) -> pd.Series:
    return ret.rolling(window, min_periods=window).std() * np.sqrt(252)


def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    mu = ret.rolling(window, min_periods=window).mean() * 252
    sd = ret.rolling(window, min_periods=window).std() * np.sqrt(252)
    return (mu / sd.replace(0, np.nan)).rename(f"sharpe_{window}d")


def turnover_stats(weights: pd.DataFrame, returns: pd.Series,
                   cost_bps: float) -> dict:
    """Monthly + annual turnover and cost drag attribution."""
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    monthly = turnover.resample("ME").sum()
    annual = turnover.resample("YE").sum()
    cost_drag_ann = float((turnover * cost_bps / 1e4).mean() * 252)
    return {
        "turnover_daily_mean": float(turnover.mean()),
        "turnover_monthly_mean": float(monthly.mean()),
        "turnover_annual_mean": float(annual.mean()),
        "turnover_annual_max": float(annual.max()),
        "cost_drag_ann": cost_drag_ann,
        "monthly_turnover": monthly,
        "annual_turnover": annual,
        "daily_turnover": turnover,
    }


def regime_split(returns: pd.Series, regime_on: pd.Series) -> pd.DataFrame:
    """Performance conditioned on risk-on vs risk-off days."""
    from metrics import perf_stats
    idx = returns.index.intersection(regime_on.index)
    r, rg = returns.loc[idx], regime_on.loc[idx].fillna(False).astype(bool)
    rows = {}
    for name, mask in (("risk_on", rg), ("risk_off", ~rg)):
        sub = r[mask]
        s = perf_stats(sub) if len(sub) > 20 else {}
        s["days"] = int(mask.sum())
        s["share"] = float(mask.mean())
        rows[name] = s
    return pd.DataFrame(rows).T


def crisis_windows(returns: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Pre-specified stress windows (no cherry-picking after the fact)."""
    from metrics import perf_stats
    windows = {
        "COVID_crash_2020-02/03": ("2020-02-19", "2020-03-23"),
        "Inflation_2022": ("2022-01-01", "2022-12-31"),
        "Q4_2018": ("2018-10-01", "2018-12-24"),
        "Taper_2013": ("2013-05-01", "2013-08-31"),
    }
    rows = {}
    for name, (a, b) in windows.items():
        s = perf_stats(returns.loc[a:b]) if len(returns.loc[a:b]) > 5 else {}
        bb = perf_stats(bench.loc[a:b]) if len(bench.loc[a:b]) > 5 else {}
        rows[name] = {"strat_CAGR": s.get("CAGR"), "strat_MaxDD": s.get("MaxDD"),
                      "bench_CAGR": bb.get("CAGR"), "bench_MaxDD": bb.get("MaxDD"),
                      "excess": (1 + (returns.loc[a:b] - bench.loc[a:b])).prod() - 1
                      if len(returns.loc[a:b]) else None}
    return pd.DataFrame(rows).T


def top_drawdowns(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """Top-N drawdown episodes with peak/trough/recovery dates."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    episodes = []
    i = 0
    vals = dd.to_numpy()
    idx = dd.index
    while i < len(vals):
        if vals[i] < -1e-9:
            start = idx[i - 1] if i > 0 else idx[i]
            j = i
            while j < len(vals) and vals[j] < -1e-9:
                j += 1
            trough = idx[i + int(np.argmin(vals[i:j]))]
            end = idx[j] if j < len(vals) else pd.NaT
            episodes.append({"peak": start, "trough": trough, "recovery": end,
                             "depth": float(vals[i:j].min()),
                             "length_d": int(j - i)})
            i = j
        else:
            i += 1
    df = pd.DataFrame(episodes).sort_values("depth").head(top).reset_index(drop=True)
    return df
