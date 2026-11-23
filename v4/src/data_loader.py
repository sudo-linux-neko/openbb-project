# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Max-data OpenBB loader — uses every FREE source (no API keys needed).

Sources used:
 1. obb.equity.price.historical (yfinance) — OHLCV stocks/ETF
 2. obb.crypto.price.historical (yfinance) — OHLCV crypto
 3. obb.equity.profile (yfinance) — sector/industry context
 4. obb.equity.fundamental.metrics/ratios/income/balance/cash (yfinance when available)
 5. obb.news.company (yfinance) — headlines for sentiment proxy (count + headline polarity)
 6. obb.index.price.historical (yfinance) — SP500 context (fallback to SPY)
Local TA (pandas-ta-openbb + TA-Lib) used for indicators to avoid keyed providers.
"""
from __future__ import annotations
import re
import time
from pathlib import Path
from datetime import date
import pandas as pd
from openbb import obb

from config import DATA_DIR


def _to_df(result) -> pd.DataFrame:
    try:
        return result.to_df()
    except Exception:
        return result


def fetch_ohlcv(symbol: str, start: str = "2015-01-01", end: str | None = None,
                provider: str = "yfinance", use_cache: bool = True) -> pd.DataFrame:
    """Fetch OHLCV for stock OR crypto with disk cache."""
    safe = symbol.replace("-", "").replace("=", "")
    cache = DATA_DIR / f"ohlcv_{safe}_{provider}.parquet"
    if use_cache and cache.exists():
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df
        except Exception:
            pass
    is_crypto = "-USD" in symbol or symbol.upper() in ("BTCUSD", "ETHUSD", "SOLUSD")
    for attempt in range(3):
        try:
            if is_crypto:
                r = obb.crypto.price.historical(symbol, start_date=start, end_date=end, provider=provider)
            else:
                r = obb.equity.price.historical(symbol, start_date=start, end_date=end, provider=provider)
            df = _to_df(r)
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"OHLCV fetch failed for {symbol}: {e}") from e
            time.sleep(2 * (attempt + 1))
    if not isinstance(df.index, pd.DatetimeIndex):
        # openbb sometimes returns 'date' column
        for c in ("date", "Date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_index()
    # OpenBB yfinance returns Index of datetime.date objects -> coerce to DatetimeIndex
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df = df.sort_index()
    df.columns = [c.lower() for c in df.columns]
    # normalize close naming
    rename = {"adj close": "adj_close", "adjclose": "adj_close"}
    df = df.rename(columns=rename)
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    df.to_parquet(cache)
    return df


def fetch_profile(symbol: str) -> dict:
    """Sector/industry — used as categorical feature. Cached."""
    cache = DATA_DIR / f"profile_{symbol}.parquet"
    try:
        if cache.exists():
            return pd.read_parquet(cache).iloc[0].to_dict()
        r = obb.equity.profile(symbol, provider="yfinance")
        df = _to_df(r)
        df.to_parquet(cache)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


def fetch_fundamentals(symbol: str) -> pd.DataFrame:
    """Best-effort fundamentals (yfinance). Returns empty DF if unavailable (e.g. crypto)."""
    if "-USD" in symbol:
        return pd.DataFrame()
    cache = DATA_DIR / f"fund_{symbol}.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    frames = []
    for fn in ("metrics", "ratios"):
        try:
            r = getattr(obb.equity.fundamental, fn)(symbol, provider="yfinance", limit=40)
            df = _to_df(r)
            frames.append(df)
        except Exception:
            continue
    if frames:
        try:
            out = frames[0]
            out.to_parquet(cache)
            return out
        except Exception:
            return frames[0]
    return pd.DataFrame()


_SIMPLE_POS = {"beat", "surge", "record", "upgrade", "bull", "gain", "rally", "growth", "profit", "strong", "buy", "outperform", "breakthrough", "approve"}
_SIMPLE_NEG = {"miss", "downgrade", "bear", "loss", "lawsuit", "probe", "cut", "warn", "plunge", "crash", "sell", "fraud", "hack", "delay", "layoff"}


def _headline_score(title: str) -> float:
    t = (title or "").lower()
    pos = sum(1 for w in _SIMPLE_POS if w in t)
    neg = sum(1 for w in _SIMPLE_NEG if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)


def fetch_news_sentiment(symbol: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Free yfinance news -> daily sentiment proxy. Cached. Columns: news_count, news_sentiment."""
    cache = DATA_DIR / f"news_{symbol.replace('-','')}.parquet"
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass
    try:
        r = obb.news.company(symbol if "-USD" not in symbol else "BTC",
                             start_date=start, provider="yfinance", limit=500)
        df = _to_df(r)
    except Exception:
        return pd.DataFrame(columns=["news_count", "news_sentiment"])
    if df.empty:
        return pd.DataFrame(columns=["news_count", "news_sentiment"])
    # find date + title cols robustly
    date_col = next((c for c in df.columns if "date" in c.lower() or "publish" in c.lower() or "datetime" in c.lower()), None)
    title_col = next((c for c in df.columns if c.lower() in ("title", "headline", "text")), df.columns[0])
    if date_col is None:
        return pd.DataFrame(columns=["news_count", "news_sentiment"])
    tmp = pd.DataFrame({
        "d": pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize(),
        "s": df[title_col].astype(str).map(_headline_score),
    })
    g = tmp.groupby("d")["s"].agg(news_count="size", news_sentiment="mean")
    g.index.name = "date"
    try:
        g.to_parquet(cache)
    except Exception:
        pass
    return g


def load_universe(symbols: list[str], start: str = "2015-01-01") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for whole universe. Returns {symbol: df}."""
    out = {}
    for s in symbols:
        try:
            out[s] = fetch_ohlcv(s, start=start)
            idx0, idx1 = pd.to_datetime(out[s].index.min()), pd.to_datetime(out[s].index.max())
            print(f"[data] {s}: {len(out[s])} bars {idx0.date()}->{idx1.date()}")
        except Exception as e:
            print(f"[data] {s} FAILED: {e}")
        time.sleep(1.5)  # be nice to yfinance rate limits
    return out


if __name__ == "__main__":
    from config import ALL_SYMBOLS
    load_universe(ALL_SYMBOLS[:3])
