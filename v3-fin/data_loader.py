# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""OpenBB data layer with disk cache, retries, and progress bars.

$0-budget provider: yfinance via OpenBB (obb.equity.price.historical).
Falls back to yfinance directly if the OpenBB shape changes.

Cache layout: data_cache/<SYMBOL>.parquet  (date-indexed OHLCV+close)
A manifest (data_cache/manifest.json) records coverage so re-runs are instant.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def _fetch_one_openbb(symbol: str, start: str, end: str, provider: str,
                      interval: str, adjustment: str) -> pd.DataFrame:
    from openbb import obb  # local import: keeps module importable without openbb
    out = obb.equity.price.historical(
        symbol, start_date=start, end_date=end,
        provider=provider, interval=interval, adjustment=adjustment,
    )
    df = out.to_dataframe()
    # Normalize: index -> DatetimeIndex named 'date', columns lowercased
    if isinstance(df.index, pd.MultiIndex):
        # some providers return (symbol, date) multiindex
        df = df.reset_index()
        date_col = "date" if "date" in df.columns else df.columns[0]
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol] if df["symbol"].nunique() > 1 else df
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    df.columns = [str(c).lower() for c in df.columns]
    for drop in ("symbol",):
        if drop in df.columns:
            df = df.drop(columns=[drop])
    if "close" not in df.columns:
        raise ValueError(f"{symbol}: no 'close' in {list(df.columns)}")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["close"]].dropna()


def _fetch_one_yf_direct(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"{symbol}: yfinance returned empty")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    close = "close" if "close" in df.columns else "adj close"
    out = pd.DataFrame({"close": pd.to_numeric(df[close], errors="coerce")})
    out.index = pd.to_datetime(df.index)
    return out.dropna()


def fetch_symbol(symbol: str, start: str, end: str, provider: str = "yfinance",
                 interval: str = "1d", adjustment: str = "splits_and_dividends",
                 max_retries: int = 3, pause: float = 0.5) -> pd.DataFrame:
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            return _fetch_one_openbb(symbol, start, end, provider, interval, adjustment)
        except Exception as e1:  # noqa: BLE001 — fall back, then retry
            last = e1
            log.warning("%s openbb attempt %d/%d failed: %s", symbol, attempt, max_retries, e1)
            try:
                return _fetch_one_yf_direct(symbol, start, end)
            except Exception as e2:  # noqa: BLE001
                last = e2
                log.warning("%s yfinance-direct attempt %d failed: %s", symbol, attempt, e2)
        time.sleep(pause * attempt)
    raise RuntimeError(f"{symbol}: all fetch attempts failed ({last})")


def load_universe(symbols: list[str], start: str, end: str, cache_dir: Path,
                  provider: str = "yfinance", interval: str = "1d",
                  adjustment: str = "splits_and_dividends",
                  batch_size: int = 20, pause: float = 0.5,
                  max_retries: int = 3) -> pd.DataFrame:
    """Download (or load from cache) daily closes -> wide DataFrame.

    Returns: DatetimeIndex x symbols, values = close. Forward-filled max 5d,
    symbols with <60% coverage or <3y data are dropped and reported.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    closes: dict[str, pd.Series] = {}
    missing: list[str] = []

    for sym in tqdm(symbols, desc="Universe download", unit="sym"):
        fp = cache_dir / f"{sym.replace('-', '_')}.parquet"
        df = None
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                df.index = pd.to_datetime(df.index)
                # reuse cache only if it covers requested window reasonably
                if (df.index.min() > pd.Timestamp(start) + pd.Timedelta(days=370)
                        or df.index.max() < pd.Timestamp(end) - pd.Timedelta(days=60)):
                    df = None  # stale -> refetch
            except Exception as e:  # noqa: BLE001
                log.warning("%s cache unreadable, refetching: %s", sym, e)
                df = None
        if df is None:
            try:
                df = fetch_symbol(sym, start, end, provider, interval,
                                  adjustment, max_retries, pause)
                df.to_parquet(fp)
            except Exception as e:  # noqa: BLE001
                log.error("%s skipped: %s", sym, e)
                missing.append(sym)
                continue
            time.sleep(pause)
        closes[sym] = df["close"].rename(sym)

    if not closes:
        raise RuntimeError("No symbols downloaded — check network/provider.")
    prices = pd.DataFrame(closes).sort_index()
    prices = prices.loc[start:end]
    # Coverage report
    n_days = len(prices)
    cov = prices.notna().mean().sort_values()
    bad = cov[cov < 0.60].index.tolist()
    short = [c for c in prices.columns
             if prices[c].dropna().shape[0] < 750]  # <~3y daily
    drop = sorted(set(bad) | set(short))
    if drop:
        log.warning("Dropping %d low-coverage symbols: %s", len(drop), drop[:15])
        prices = prices.drop(columns=drop)
    prices = prices.ffill(limit=5)  # holidays / 1-2d halts only; no long fill
    manifest = {
        "start": str(prices.index.min()), "end": str(prices.index.max()),
        "n_days": len(prices), "n_symbols": int(prices.shape[1]),
        "dropped": drop, "missing": missing,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Prices: %s rows x %s cols (%s -> %s)",
             len(prices), prices.shape[1], prices.index.min().date(),
             prices.index.max().date())
    return prices
