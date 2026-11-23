# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Backtester: monthly-rebalanced, cost-aware, vol-targeted.

r_p(t) = sum_i w_i(t-1) * r_i(t) - turnover(t) * cost
Gross exposure scaled by trailing vol: scale = vol_target / realized_vol,
clipped to [min_gross, max_gross], computed on trailing data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def apply_vol_target(strategy_ret: pd.Series, vol_target_ann: float,
                     vol_window: int = 60,
                     min_gross: float = 0.0,
                     max_gross: float = 1.0) -> tuple[pd.Series, pd.Series]:
    trailing_vol = strategy_ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    scale = (vol_target_ann / trailing_vol.replace(0, np.nan)).clip(min_gross, max_gross)
    scale = scale.fillna(0.0)  # warmup -> stay in cash, not a backfilled guess
    return (strategy_ret * scale.shift(1).fillna(0.0)), scale.shift(1).fillna(0.0)


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame,
                 cost_bps: float, vol_target_ann: float, vol_window: int,
                 min_gross: float, max_gross: float,
                 max_name_weight: float = 0.05) -> dict:
    rets = to_returns(prices).reindex(weights.index).fillna(0.0)
    weights = weights.reindex(prices.index).fillna(0.0)
    # hard cap per-name weight (institutional concentration limit)
    w = weights.copy()
    over = w.gt(max_name_weight)
    # only cap equity sleeve; hedge ETFs (TLT/SHY) may legally exceed 5%
    # -> identify hedge cols heuristically: caller passes full frame, we cap
    # columns that are NOT the two largest-weight columns on risk-off dates.
    # Simpler & safe: cap everything above max_name_weight EXCEPT TLT/SHY.
    for col in ("TLT", "SHY"):
        if col in w.columns:
            over[col] = False
    if over.any().any():
        w = w.mask(over, max_name_weight)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    gross = w.abs().sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    cost_rate = cost_bps / 1e4
    strat_gross_ret = (w.shift(0) * rets).sum(axis=1)  # w already lagged by strategy
    strat_net_unscaled = strat_gross_ret - turnover * cost_rate

    strat_ret, scale = apply_vol_target(strat_net_unscaled, vol_target_ann,
                                        vol_window, min_gross, max_gross)
    equity = (1 + strat_ret).cumprod()
    equity_unscaled = (1 + strat_net_unscaled).cumprod()
    return {
        "returns": strat_ret, "equity": equity,
        "returns_unscaled": strat_net_unscaled, "equity_unscaled": equity_unscaled,
        "weights": w, "scale": scale, "turnover": turnover, "gross": gross * scale,
    }


def benchmark_6040(prices: pd.DataFrame, w_spy: float = 0.6, w_agg: float = 0.4,
                   cost_bps: float = 2.0) -> pd.Series:
    """Monthly-rebalanced 60/40 SPY/AGG total-return proxy."""
    cols = [c for c in ("SPY", "AGG") for _ in [0]]
    if not all(c in prices.columns for c in ("SPY", "AGG")):
        raise KeyError("SPY and AGG required for 60/40 benchmark")
    rets = to_returns(prices[["SPY", "AGG"]])
    month_ends = prices.resample("ME").last().index
    month_ends = prices.index[prices.index.get_indexer(month_ends, method="ffill")].drop_duplicates()
    w = pd.DataFrame(0.0, index=prices.index, columns=["SPY", "AGG"])
    w.loc[month_ends] = [w_spy, w_agg]
    w = w.replace(0.0, np.nan).ffill().fillna(0.0).shift(1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return ((w * rets).sum(axis=1) - turnover * cost_bps / 1e4).rename("bench_6040")
