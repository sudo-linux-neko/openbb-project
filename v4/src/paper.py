# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Alpaca paper trading plug (safe by default: dry-run prints orders)."""
from __future__ import annotations
import os


def get_client(paper: bool = True):
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        return None
    from alpaca.trading.client import TradingClient
    return TradingClient(key, secret, paper=paper)


def submit_orders(signals: dict[str, float], notional_per_trade: float = 2000.0,
                  dry_run: bool = True, min_prob: float = 0.6):
    """signals: {symbol: prob}. Submits market buys for prob>=min_prob (strips '-USD' unsupported)."""
    picks = {s: p for s, p in signals.items() if p >= min_prob and "-USD" not in s}
    if dry_run:
        print(f"[paper DRY-RUN] would BUY {picks} @ ${notional_per_trade} each")
        return picks
    client = get_client()
    if client is None:
        print("[paper] No APCA keys in env; falling back to dry-run.")
        return picks
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    results = {}
    for s, p in picks.items():
        try:
            req = MarketOrderRequest(symbol=s, notional=notional_per_trade,
                                     side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            o = client.submit_order(req)
            results[s] = str(o.id)
            print(f"[paper] BUY {s} prob={p:.3f} id={o.id}")
        except Exception as e:
            print(f"[paper] {s} FAILED: {e}")
    return results
