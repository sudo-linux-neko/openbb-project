# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Strategy: regime-filtered cross-sectional momentum with bond hedge.

Rules (applied at month-end close, effective next trading day — no lookahead):
  1. REGIME: risk-on iff SPY_close > SMA200(SPY). Else risk-off.
  2. RISK-ON: among equities with price > SMA200(own) and valid 12-1 momentum,
     hold equal-weight TOP-N by momentum (12m return skipping last 21d).
  3. RISK-OFF: hold hedge mix (default 50% TLT / 50% SHY).
  4. VOL TARGET: scale gross exposure to vol_target using 60d realized vol
     of the *unscaled* strategy return stream (trailing only).

Only 3 research degrees of freedom (lookback, trend_window, top_n) — this is
the anti-overfit feature, not a limitation. Complexity is in validation, not rules.
"""
from __future__ import annotations

import pandas as pd


def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()


def momentum_12_1(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Trailing total return over `lookback` days ending `skip` days ago."""
    return prices.shift(skip) / prices.shift(skip + lookback) - 1.0


def build_signals(prices: pd.DataFrame, equities: list[str], regime_symbol: str,
                  hedge_symbols: list[str], momentum_lookback: int = 252,
                  momentum_skip: int = 21, trend_window: int = 200,
                  top_n: int = 20,
                  rebalance_freq: str = "ME",
                  hedge_weights: tuple[float, ...] | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    equities = [e for e in equities if e in prices.columns]
    if regime_symbol not in prices.columns:
        raise KeyError(f"regime symbol {regime_symbol} not in price columns")
    for h in hedge_symbols:
        if h not in prices.columns:
            raise KeyError(f"hedge symbol {h} not in price columns")
    if hedge_weights is None:
        hedge_weights = tuple([1.0 / len(hedge_symbols)] * len(hedge_symbols))
    if len(hedge_weights) != len(hedge_symbols) or abs(sum(hedge_weights) - 1.0) > 1e-9:
        raise ValueError(f"hedge_weights {hedge_weights} must sum to 1 over {hedge_symbols}")
    hedge_w = dict(zip(hedge_symbols, hedge_weights))

    mom = momentum_12_1(prices[equities], momentum_lookback, momentum_skip)
    above_trend = prices[equities].gt(prices[equities].rolling(trend_window,
                                                               min_periods=trend_window).mean())
    regime_on = (prices[regime_symbol] > sma(prices[regime_symbol], trend_window)).rename("regime_on")

    month_ends = prices.resample(rebalance_freq).last().index
    # snap month-ends to actual trading days (resample dates often fall on weekends)
    month_ends = prices.index[prices.index.get_indexer(month_ends, method="ffill")]
    month_ends = month_ends.drop_duplicates().sort_values()
    # drop warmup dates where indicators are not yet defined
    month_ends = month_ends[month_ends >= prices.index[min(260, len(prices) - 1)]]

    weights = pd.DataFrame(float("nan"), index=prices.index,
                           columns=equities + hedge_symbols)
    regime_state = pd.Series(pd.NA, index=prices.index, dtype="boolean")

    for dt in month_ends:
        is_on = bool(regime_on.loc[dt]) if pd.notna(regime_on.loc[dt]) else False
        regime_state.loc[dt] = is_on
        w = pd.Series(0.0, index=weights.columns)
        if is_on:
            m = mom.loc[dt].dropna()
            m = m[above_trend.loc[dt].reindex(m.index).fillna(False)]
            picks = m.nlargest(top_n).index.tolist()
            if picks:
                w[picks] = 1.0 / len(picks)
            else:  # no qualifiers -> stay hedged (fail-safe to low risk)
                for h, hw in hedge_w.items():
                    w[h] = hw
                regime_state.loc[dt] = False
        else:
            for h, hw in hedge_w.items():
                w[h] = hw
        weights.loc[dt] = w

    weights = weights.ffill().fillna(0.0)
    # enforce 1-day implementation lag everywhere (signal t -> exposure t+1)
    weights = weights.shift(1).fillna(0.0)
    regime_state = regime_state.ffill().fillna(False).shift(1).fillna(False)
    return weights, regime_state.astype(bool), month_ends
