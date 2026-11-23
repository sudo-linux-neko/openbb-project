# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Robustness: walk-forward + bootstrap. The anti-overfit core.

1. Walk-forward (expanding train, 1y test, 21d purge gap): for a tiny grid of
   (trend_window, top_n) pick best TRAIN Sharpe, evaluate on TEST only.
   Report mean/median OOS Sharpe and IS-vs-OOS decay. Big decay = overfit.
2. Bootstrap: resample test-period daily returns (block=21d) 1000x to get
   CI for Sharpe and P(Sharpe<0). Fragile strategies break here.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from strategy import build_signals
from backtester import run_backtest, to_returns
from metrics import perf_stats


def _window_return(prices: pd.DataFrame, weights: pd.DataFrame, **bt) -> pd.Series:
    out = run_backtest(prices, weights, cost_bps=bt["cost_bps"],
                       vol_target_ann=bt["vol_target_ann"], vol_window=bt["vol_window"],
                       min_gross=bt["min_gross"], max_gross=bt["max_gross"])
    return out["returns"]


def walk_forward(prices: pd.DataFrame, equities: list[str], cfg,
                 grid_trend=(150, 200, 250), grid_topn=(15, 20, 30)) -> pd.DataFrame:
    years = sorted(set(prices.index.year))
    rows = []
    combos = list(itertools.product(grid_trend, grid_topn))
    bt = dict(cost_bps=cfg.total_cost_bps, vol_target_ann=cfg.vol_target_ann,
              vol_window=cfg.vol_window, min_gross=cfg.min_gross, max_gross=cfg.max_gross)
    first_test = min(years) + cfg.train_years
    for test_y in tqdm(range(first_test, max(years) + 1), desc="Walk-forward", unit="yr"):
        train = prices[prices.index.year < test_y]
        train = train[train.index >= train.index.min() + pd.DateOffset(years=0)]
        # keep last train_years of history
        cutoff = pd.Timestamp(test_y - cfg.train_years, 1, 1)
        train = train[train.index >= cutoff]
        purge = pd.Timestamp(test_y, 1, 1) - pd.Timedelta(days=cfg.purge_days)
        train = train[train.index < purge]
        test = prices[(prices.index.year == test_y)]
        if len(train) < 750 or len(test) < 100:
            continue
        best, best_key = -np.inf, None
        for tw, tn in combos:
            w, _, _ = build_signals(train, equities, cfg.regime_symbol,
                                    list(cfg.hedge_symbols),
                                    momentum_lookback=cfg.momentum_lookback,
                                    momentum_skip=cfg.momentum_skip,
                                    trend_window=tw, top_n=tn)
            r = _window_return(train, w, **bt)
            s = perf_stats(r.loc[str(test_y - cfg.train_years):])["Sharpe"]
            if s > best:
                best, best_key = s, (tw, tn)
        tw, tn = best_key
        wfull_test, _, _ = build_signals(
            prices[prices.index <= test.index.max()], equities, cfg.regime_symbol,
            list(cfg.hedge_symbols), momentum_lookback=cfg.momentum_lookback,
            momentum_skip=cfg.momentum_skip, trend_window=tw, top_n=tn)
        r_oos = _window_return(prices, wfull_test, **bt).loc[test.index]
        rows.append({"test_year": test_y, "picked_trend": tw, "picked_topn": tn,
                     "IS_Sharpe": best, "OOS_Sharpe": perf_stats(r_oos)["Sharpe"],
                     "OOS_CAGR": perf_stats(r_oos)["CAGR"],
                     "OOS_MaxDD": perf_stats(r_oos)["MaxDD"]})
    return pd.DataFrame(rows)


def block_bootstrap_ci(returns: pd.Series, n: int = 1000, block: int = 21,
                       seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    nb = int(np.ceil(len(r) / block))
    sharpes = []
    for _ in tqdm(range(n), desc="Bootstrap", unit="iter"):
        idx = np.concatenate([np.arange(s, min(s + block, len(r)))
                              for s in rng.integers(0, len(r) - block + 1, nb)])[:len(r)]
        sample = r[idx]
        sd = sample.std()
        sharpes.append(float(sample.mean() / sd * np.sqrt(252)) if sd else 0.0)
    a = np.array(sharpes)
    return {"mean": float(a.mean()), "p2.5": float(np.percentile(a, 2.5)),
            "p50": float(np.percentile(a, 50)), "p97.5": float(np.percentile(a, 97.5)),
            "P(Sharpe<0)": float((a < 0).mean())}
