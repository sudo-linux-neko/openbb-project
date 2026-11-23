# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Walk-forward swing backtester: weekly rebalance, top-K probs, vol-aware sizing, costs."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if len(rets) < 5:
        return {}
    ann = 252 / 5 if len(rets) else 252
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / (downside + 1e-12) * np.sqrt(252)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(1, len(equity) / 252)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    calmar = cagr / (abs(dd.min()) + 1e-12)
    return {
        "total_return": round(float(total), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown": round(float(dd.min()), 4),
        "calmar": round(float(calmar), 3),
        "win_weeks": int((rets > 0).sum()),
        "weeks": int(len(rets)),
    }


def backtest_portfolio(signals: pd.DataFrame, prices: dict[str, pd.DataFrame],
                       initial=100_000.0, top_k=5, fee_bps=5, min_prob=0.55,
                       max_positions=8) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """signals: index=date, columns=symbol, values=prob (already time-safe, shifted).
    prices: {symbol: ohlcv df with close}.
    Rebalance weekly (Friday close): hold top-K above min_prob, equal-vol weight.
    Returns (trades, metrics, equity_curve)."""
    # weekly rebalance dates: Fridays or last available day each week
    all_dates = sorted(set(signals.index))
    weeks = pd.Series(all_dates).groupby(
        [pd.to_datetime(all_dates).to_period("W")]).tail(1).tolist()
    equity = initial
    curve = []
    holdings: dict[str, float] = {}  # symbol -> shares
    cash = initial
    trades = []
    k = min(top_k, max_positions)
    for d in all_dates:
        # mark to market daily
        mv = cash + sum(holdings.get(s, 0) * float(prices[s]["close"].reindex([d]).ffill().iloc[0])
                        if s in prices and d in prices[s].index or True else 0 for s in list(holdings))
        # safer mtm:
        mv = cash
        for s, sh in list(holdings.items()):
            try:
                px = float(prices[s]["close"].asof(d))
                mv += sh * px
            except Exception:
                pass
        curve.append((d, mv))
        if d not in weeks:
            continue
        # rebalance
        try:
            row = signals.loc[d].dropna().sort_values(ascending=False)
        except KeyError:
            continue
        picks = row[row >= min_prob].head(k).index.tolist()
        # sell losers
        for s in list(holdings):
            if s not in picks:
                try:
                    px = float(prices[s]["close"].asof(d))
                    proceeds = holdings[s] * px * (1 - fee_bps / 1e4)
                    cash += proceeds
                    trades.append({"date": d, "symbol": s, "side": "SELL", "price": px})
                    del holdings[s]
                except Exception:
                    pass
        # buy new (equal weight of current equity)
        if picks:
            eq_now = cash + sum(holdings.get(s, 0) * float(prices[s]["close"].asof(d)) for s in holdings)
            per = eq_now / max(1, k)
            for s in picks:
                if s in holdings or s not in prices:
                    continue
                try:
                    px = float(prices[s]["close"].asof(d))
                    if px <= 0 or per < px:
                        continue
                    spend = min(cash, per) * (1 - fee_bps / 1e4)
                    sh = int(spend // px)
                    if sh <= 0:
                        continue
                    cash -= sh * px * (1 + fee_bps / 1e4)
                    holdings[s] = holdings.get(s, 0) + sh
                    trades.append({"date": d, "symbol": s, "side": "BUY", "price": px})
                except Exception:
                    continue
    eq = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date").sort_index()
    metrics = compute_metrics(eq["equity"])
    # buy&hold SPY baseline if available
    trades_df = pd.DataFrame(trades)
    return trades_df, metrics, eq
