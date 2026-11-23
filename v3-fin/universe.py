# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Static investable universe (~100 names + ETFs).

Why static? $0 budget + no survivorship-bias-free free source.
We use a fixed large-cap list and DISCLOSE the bias instead of hiding it:
current S&P-100-style constituents backfilled 15y overstates quality
(delisted/bankrupt names missing). Robustness section quantifies per-ticker
breadth so a few mega-caps can't silently drive the result.

All symbols are liquid US listings that yfinance covers for 15y+.
"""
from __future__ import annotations

# ~100 large-cap US equities (S&P-100 proxy, liquid, 15y history).
EQUITIES_100: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "AVGO", "JPM",
    "XOM", "LLY", "UNH", "V", "MA", "JNJ", "WMT", "PG", "ORCL", "HD",
    "COST", "BAC", "ABBV", "MRK", "CVX", "CRM", "KO", "AMD", "PEP", "TMO",
    "ACN", "LIN", "CSCO", "MCD", "ABT", "DHR", "TXN", "WFC", "VZ", "AMGN",
    "IBM", "INTU", "QCOM", "AMAT", "GE", "CAT", "NOW", "SPGI", "UNP", "HON",
    "PM", "RTX", "GS", "MS", "BKNG", "ADBE", "NEE", "LOW", "ELV", "AXP",
    "TJX", "SYK", "GILD", "VRTX", "C", "LMT", "ADI", "MMC", "PLD", "CB",
    "MDT", "AMT", "T", "D", "SO", "DUK", "NKE", "SBUX", "BLK", "SCHW",
    "MO", "ISRG", "ZTS", "CI", "REGN", "TMUS", "FI", "CME", "USB", "PGR",
    "BDX", "EOG", "AON", "ITW", "SHW", "NSC", "MCO", "APD", "FDX", "HUM",
]

# Regime / hedge / benchmark ETFs. All free on yfinance with 15y+ history.
REGIME_ETF = "SPY"
HEDGE_ETFS = ["TLT", "SHY"]
BENCH_ETFS = ["SPY", "AGG"]
EXTRA_ETFS = ["QQQ", "GLD"]  # diagnostics only, not traded by default

assert len(EQUITIES_100) >= 100, f"need >=100 names, got {len(EQUITIES_100)}"

ALL_SYMBOLS: list[str] = sorted(set(EQUITIES_100 + [REGIME_ETF] + HEDGE_ETFS + BENCH_ETFS + EXTRA_ETFS))


def describe() -> str:
    return (
        f"{len(EQUITIES_100)} equities + "
        f"{len(set([REGIME_ETF] + HEDGE_ETFS + BENCH_ETFS + EXTRA_ETFS))} ETFs = "
        f"{len(ALL_SYMBOLS)} symbols total"
    )
