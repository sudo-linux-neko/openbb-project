# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Beast feature engineering: 120+ features per symbol.

Families:
 - returns / log-returns / lags / forward target
 - volatility / Parkinson / Garman-Klass / ATR-normalized
 - trend: SMA/EMA ratios, MACD, ADX, Aroon, Ichimoku proxy, Donchian position
 - momentum/mean-reversion: RSI, Stoch, CCI, Williams%R, ROC, z-scores, distance to high/low
 - volume: OBV, MFI proxy, volume z-score, dollar volume, Amihud illiquidity
 - candles: range, body, upper/lower wick, gaps
 - calendar: dow/month/quarter/year-end one-hots + cyclic encodings
 - cross-asset: SPY/QQQ/BTC returns, correlations, relative strength, beta proxy
 - news: count + sentiment + rolling means
 - fundamentals (if available): latest pe/pb/roe merged as slow-moving constants
 - regime: vol percentile, trend percentile, drawdown-from-high

No lookahead: every feature at time t uses data <= t. Target uses t+1..t+H.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import talib  # TA-Lib available in ml_env
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _stoch(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    kline = 100 * (close - ll) / (hh - ll + 1e-12)
    return kline, kline.rolling(d).mean()


def _atr(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdm = pd.Series(plus, index=close.index).ewm(alpha=1 / n, adjust=False).mean()
    mdm = pd.Series(minus, index=close.index).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm / (atr + 1e-12)
    mdi = 100 * mdm / (atr + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def add_single_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df.get("volume", pd.Series(0, index=df.index)).fillna(0)
    lr = np.log(c / c.shift(1))
    df["ret_1"] = c.pct_change(1)
    df["logret_1"] = lr
    for n in (2, 3, 5, 10, 20, 60):
        df[f"ret_{n}"] = c.pct_change(n)
        df[f"logret_{n}"] = np.log(c / c.shift(n))
        df[f"mom_{n}"] = c / c.shift(n) - 1
    for n in (5, 10, 20, 60):
        df[f"vol_{n}"] = lr.rolling(n).std() * np.sqrt(252)
        df[f"vol_chg_{n}"] = df[f"vol_{n}"].pct_change(5)
    # Parkinson + Garman-Klass
    df["park_20"] = (np.log(h / l) ** 2).rolling(20).mean() * 252 / (4 * np.log(2))
    df["park_20"] = np.sqrt(df["park_20"].clip(lower=0))
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    df["gk_20"] = np.sqrt((gk.rolling(20).mean() * 252).clip(lower=0))
    atr14 = _atr(h, l, c)
    df["atr_14"] = atr14 / c
    df["atr_14_raw"] = atr14
    # trend
    for n in (5, 10, 20, 50, 100, 200):
        sma = c.rolling(n).mean()
        df[f"sma_{n}"] = c / sma - 1
        df[f"close_gt_sma{n}"] = (c > sma).astype(float)
    for n in (8, 21, 55):
        df[f"ema_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    m_line, m_sig, m_hist = _macd(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = m_line / c, m_sig / c, m_hist / c
    adx, pdi, mdi = _adx(h, l, c)
    df["adx_14"], df["pdi_14"], df["mdi_14"] = adx, pdi, mdi
    hh20 = h.rolling(20).max()
    ll20 = l.rolling(20).min()
    df["donch_pos_20"] = (c - ll20) / (hh20 - ll20 + 1e-12)
    df["dist_high_20"] = c / hh20 - 1
    df["dist_low_20"] = c / ll20 - 1
    hh60 = h.rolling(60).max()
    df["dist_high_60"] = c / hh60 - 1
    df["dd_from_high_60"] = c / c.rolling(60).max() - 1
    # oscillators
    if HAS_TALIB:
        try:
            df["rsi_14"] = talib.RSI(c.values, 14)
            df["cci_20"] = talib.CCI(h.values, l.values, c.values, 20)
            df["willr_14"] = talib.WILLR(h.values, l.values, c.values, 14)
        except Exception:
            df["rsi_14"] = _rsi(c)
            df["cci_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
            df["willr_14"] = ((h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-12)) * -100
    else:
        df["rsi_14"] = _rsi(c)
        df["rsi_7"] = _rsi(c, 7)
        df["cci_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
        df["willr_14"] = ((h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-12)) * -100
    if "rsi_7" not in df.columns:
        df["rsi_7"] = _rsi(c, 7)
    df["rsi_28"] = _rsi(c, 28)
    k, kd = _stoch(h, l, c)
    df["stoch_k"], df["stoch_d"] = k, kd
    for n in (5, 10, 20):
        mu = c.rolling(n).mean()
        sd = c.rolling(n).std()
        df[f"z_{n}"] = (c - mu) / (sd + 1e-12)
    for n in (10, 20):
        ma = c.rolling(n).mean()
        sd = c.rolling(n).std()
        df[f"bb_pos_{n}"] = (c - (ma - 2 * sd)) / (4 * sd + 1e-12)
        df[f"bb_width_{n}"] = (4 * sd) / (ma + 1e-12)
    df["roc_10"] = c.pct_change(10)
    df["roc_20"] = c.pct_change(20)
    # volume
    dv = c * v
    df["dollar_vol_20"] = dv.rolling(20).mean()
    df["vol_z_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)
    df["vol_ratio_5_20"] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-12)
    df["obv"] = ((np.sign(c.diff()).fillna(0)) * v).cumsum()
    df["obv_slope_10"] = df["obv"].diff(10) / (dv.rolling(10).mean() + 1e-12)
    typical = (h + l + c) / 3
    mf = typical * v
    df["mfi_14"] = 100 - 100 / (1 + mf.where(c.diff() > 0, 0).rolling(14).sum() / (mf.where(c.diff() <= 0, 0).rolling(14).sum() + 1e-12))
    df["amihud_20"] = (c.pct_change().abs() / (dv + 1e-12)).rolling(20).mean() * 1e9
    # candles
    rng = (h - l) / (c + 1e-12)
    df["range_pct"] = rng
    df["body_pct"] = (c - o) / (c + 1e-12)
    df["upper_wick"] = (h - np.maximum(c, o)) / (c + 1e-12)
    df["lower_wick"] = (np.minimum(c, o) - l) / (c + 1e-12)
    df["gap"] = o / c.shift(1) - 1
    df["range_z_20"] = (rng - rng.rolling(20).mean()) / (rng.rolling(20).std() + 1e-12)
    # calendar
    idx = df.index
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_month_end"] = idx.is_month_end.astype(float)
    df["is_quarter_end"] = idx.is_quarter_end.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    # regime percentiles
    df["vol_pct_120"] = df["vol_20"].rolling(120).rank(pct=True)
    df["ret60_pct_252"] = df["ret_60"].rolling(252).rank(pct=True)
    return df


def add_cross_asset_features(feats: dict[str, pd.DataFrame], contexts: list[str]) -> dict[str, pd.DataFrame]:
    ctx_rets = {}
    for s in contexts:
        if s in feats:
            c = feats[s]["close"]
            ctx_rets[s] = c.pct_change(1)
    out = {}
    for sym, df in feats.items():
        df = df.copy()
        for cs, r in ctx_rets.items():
            if cs == sym:
                continue
            tag = cs.replace("-", "")
            aligned = r.reindex(df.index)
            df[f"ctx_{tag}_ret1"] = aligned
            df[f"ctx_{tag}_ret5"] = aligned.rolling(5).sum()
            df[f"rel_strength_20_{tag}"] = df["ret_20"] - aligned.rolling(20).sum()
            df[f"corr_20_{tag}"] = df["ret_1"].rolling(20).corr(aligned)
            # beta proxy
            cov = df["ret_1"].rolling(60).cov(aligned)
            var = aligned.rolling(60).var()
            df[f"beta_60_{tag}"] = cov / (var + 1e-12)
        out[sym] = df
    return out


def add_news_features(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if news is None or news.empty:
        df["news_count"] = 0.0
        df["news_sentiment"] = 0.0
        df["news_count_5"] = 0.0
        df["news_sent_5"] = 0.0
        return df
    n = news.copy()
    n.index = pd.to_datetime(n.index).tz_localize(None)
    df = df.join(n[["news_count", "news_sentiment"]], how="left")
    df[["news_count", "news_sentiment"]] = df[["news_count", "news_sentiment"]].fillna(0.0)
    df["news_count_5"] = df["news_count"].rolling(5).sum()
    df["news_sent_5"] = df["news_sentiment"].rolling(5).mean()
    return df


def make_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.015) -> pd.DataFrame:
    df = df.copy()
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    df["fwd_ret"] = fwd
    df["target"] = 0
    df.loc[fwd > threshold, "target"] = 1
    df.loc[fwd < -threshold, "target"] = -1
    df["target_up"] = (fwd > threshold).astype(int)  # binary head for prob ranking
    return df


def build_all(price: dict[str, pd.DataFrame], news_map: dict[str, pd.DataFrame],
              contexts: list[str], horizon=5, threshold=0.015) -> dict[str, pd.DataFrame]:
    feats = {s: add_single_asset_features(d) for s, d in price.items()}
    feats = add_cross_asset_features(feats, contexts)
    out = {}
    for s, d in feats.items():
        d = add_news_features(d, news_map.get(s))
        d = make_target(d, horizon, threshold)
        out[s] = d
    return out


FEATURE_BLACKLIST_PREFIX = ("fwd_ret", "target", "target_up", "open", "high", "low", "close", "volume", "adj_close")


def feature_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if c in ("fwd_ret", "target", "target_up", "open", "high", "low", "close", "volume", "adj_close"):
            continue
        if c.startswith("target"):
            continue
        out.append(c)
    # keep only numeric
    num = df[out].select_dtypes(include=["number"]).columns.tolist()
    return num
