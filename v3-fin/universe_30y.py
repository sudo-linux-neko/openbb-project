# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""30-year universe: pre-1995 continuous tickers only (>=40 names).

Agent-1 shortlist. Excludes post-1995 births (META, GOOGL, NVDA, AMZN, V, MA,
ABBV, CRM, GS, BKNG, ...), spin/merger breaks (PM, RTX, LIN, VZ, ABBV, ZTS...),
borderline late starts (BRK-B 1996-05, LMT 1995-03).
All names listed well before 1995. Coverage filter in run script enforces
>=70% of SPY calendar since 1995 + >=5000 bars + first bar <=1996-06-30.
"""
from __future__ import annotations

EQUITIES_30Y: list[str] = [
    "AAPL", "MSFT", "JPM", "XOM", "LLY", "JNJ", "WMT", "PG", "ORCL", "HD",
    "BAC", "MRK", "CVX", "KO", "AMD", "PEP", "TMO", "CSCO", "MCD", "ABT",
    "DHR", "TXN", "WFC", "AMGN", "IBM", "QCOM", "AMAT", "GE", "CAT", "SPGI",
    "UNP", "HON", "ADBE", "NEE", "LOW", "AXP", "TJX", "SYK", "C", "ADI",
    "MMC", "MDT", "T", "D", "SO", "DUK", "NKE", "SBUX", "SCHW", "MO",
    "USB", "PGR", "BDX", "AON", "ITW", "SHW", "NSC", "APD", "FDX", "HUM",
    "UNH", "CB", "COST", "GILD", "INTU", "MS",
]

ETFS_30Y: list[str] = ["SPY", "AGG", "TLT", "SHY", "QQQ", "GLD"]
# Note: TLT (2002), SHY (2002), QQQ (1999), GLD (2004), AGG (2003) start after
# 1995. Equity sleeve runs 30y; hedge/bench sleeves activate when available
# (pre-inception hedge days hold SHY-proxy = cash at 0% — disclosed drag).

ALL_30Y: list[str] = sorted(set(EQUITIES_30Y + ETFS_30Y))

SURVIVORSHIP_NOTE = (
    "30y results use current large-cap survivors with continuous data since 1995; "
    "delisted, bankrupt, and acquired names are absent, and early small-cap histories "
    "were illiquid. This overstates absolute return/Sharpe and understates drawdowns; "
    "treat long-horizon levels as upper bounds and judge the strategy on "
    "relative/breadth-robust metrics, not headline CAGR."
)
