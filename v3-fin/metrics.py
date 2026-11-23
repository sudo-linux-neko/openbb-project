# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Performance analytics: Sharpe/Sortino/MaxDD/Calmar + Deflated Sharpe.

Deflated Sharpe (Bailey & Lopez de Prado 2014) implemented in closed form:
penalizes Sharpe for number of trials K and skew/kurtosis of returns.
This is the honest answer to 'is it overfit': report DSR, not just Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_dt = equity.loc[:trough].idxmax()
    return float(dd.min()), peak_dt, trough


def perf_stats(returns: pd.Series, periods: int = 252) -> dict:
    r = returns.dropna()
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    years = len(r) / periods if len(r) else np.nan
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 else 0.0
    vol = float(r.std() * np.sqrt(periods)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std() * np.sqrt(periods)) if r.std() else 0.0
    downside = r[r < 0].std() * np.sqrt(periods)
    sortino = float(r.mean() / r[r < 0].std() * np.sqrt(periods)) if r[r < 0].std() else 0.0
    mdd, peak_dt, trough_dt = max_drawdown(equity) if len(equity) else (0.0, None, None)
    calmar = float(cagr / abs(mdd)) if mdd else 0.0
    skew = float(r.skew()) if len(r) > 2 else 0.0
    kurt = float(r.kurtosis()) if len(r) > 3 else 0.0
    best = float(r.max()); worst = float(r.min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": mdd, "Peak": peak_dt, "Trough": trough_dt, "Calmar": calmar,
            "Skew": skew, "Kurt": kurt, "TotalRet": total, "Years": years,
            "BestDay": best, "WorstDay": worst, "N": len(r)}


def deflated_sharpe(returns: pd.Series, n_trials: int, benchmark_sr: float = 0.0) -> dict:
    """Return DSR estimate + expected Sharpe under H0.

    Reference: Bailey & Lopez de Prado, 'The Deflated Sharpe Ratio' (2014).
    """
    r = returns.dropna()
    n = len(r)
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else 0.0
    skew = float(r.skew()); kurt = float(r.kurtosis())
    # Expected Sharpe under null given K trials (Eq. in paper, Euler-Mascheroni)
    gamma = 0.5772156649
    e_sr0 = np.sqrt(1 / max(n - 1, 1)) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
                                          + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(n - 1, 1)
    dsr = float(stats.norm.cdf((sr - e_sr0) / np.sqrt(max(var_sr, 1e-12))))
    return {"Sharpe": sr, "E[SR0]": float(e_sr0), "DSR": dsr,
            "trials": n_trials, "benchmark_SR": benchmark_sr}


def compare_table(strat: pd.Series, bench: pd.Series) -> pd.DataFrame:
    idx = strat.index.intersection(bench.index)
    s, b = strat.loc[idx], bench.loc[idx]
    active = (s - b)
    rows = {}
    for name, series in (("Strategy", s), ("60/40", b), ("Active", active)):
        rows[name] = perf_stats(series)
    df = pd.DataFrame(rows).T
    # excess / tracking
    df.loc["Active", "CAGR"] = float(((1 + active).cumprod().iloc[-1]) ** (252 / len(active)) - 1)
    return df


def per_ticker_breadth(prices: pd.DataFrame, equities: list[str],
                       mom_lb: int = 252, skip: int = 21) -> pd.DataFrame:
    """Sanity: what fraction of names ever qualify / contribute? Guards vs
    '2 stocks drove everything' concentration illusion."""
    mom = prices[equities].shift(skip) / prices[equities].shift(skip + mom_lb) - 1
    trend = prices[equities].gt(prices[equities].rolling(200, min_periods=200).mean())
    monthly_raw = prices.resample("ME").last().index
    monthly = prices.index[prices.index.get_indexer(monthly_raw, method="ffill")].drop_duplicates()
    monthly = monthly[monthly.isin(mom.index) & monthly.isin(trend.index)]
    hit = ((mom.loc[monthly].rank(axis=1, ascending=False) <= 20) & trend.loc[monthly].fillna(False))
    return pd.DataFrame({"months_selected": hit.sum(),
                         "sel_rate": hit.mean()}).sort_values("months_selected", ascending=False)
