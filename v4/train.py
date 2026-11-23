# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Train beast: fetch max OpenBB data -> features -> per-symbol ensemble -> pooled signals matrix.

Usage:
  source ~/ml_env/bin/activate
  python train.py              # full universe
  python train.py --symbols AAPL,MSFT,NVDA --start 2018-01-01 --fast
"""
from __future__ import annotations
import argparse
import pickle
import pandas as pd
import torch

import config as C
from src.data_loader import load_universe, fetch_news_sentiment
from src.features import build_all, feature_cols
from src.models import train_beast
from src.backtest import backtest_portfolio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(C.ALL_SYMBOLS))
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--fast", action="store_true", help="fewer epochs for quick test")
    args = ap.parse_args()

    import src.models as M
    if args.fast:
        M.LSTM_EPOCHS = 6

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"[train] universe={symbols}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Intel XPU?
    try:
        import importlib
        xpu_mod = importlib.import_module("torch.xpu")
        if xpu_mod.is_available():
            device = "xpu"
    except Exception:
        pass
    print(f"[train] device={device}")

    prices = load_universe(symbols, start=args.start)
    news_map = {s: fetch_news_sentiment(s, start=args.start) for s in symbols}
    feats = build_all(prices, news_map, [c for c in C.CONTEXT_SYMBOLS if c in prices],
                      horizon=C.HORIZON, threshold=C.THRESHOLD)

    # pooled training: concat all symbols (with symbol one-hot-ish via relative features already)
    frames = []
    for s, d in feats.items():
        d2 = d.copy()
        d2["__sym"] = s
        frames.append(d2)
    pooled = pd.concat(frames).sort_index()
    fcols = feature_cols(pooled)
    # drop columns mostly-NaN
    keep = [c for c in fcols if pooled[c].notna().mean() > 0.7]
    print(f"[train] features: {len(fcols)} -> {len(keep)} after NaN filter")
    fcols = keep
    pooled = pooled.dropna(subset=fcols + ["target_up"])
    print(f"[train] pooled rows={len(pooled)} pos_rate={pooled['target_up'].mean():.3f}")

    model, stats = train_beast(pooled, fcols, n_splits=C.N_SPLITS,
                               embargo_pct=C.EMBARGO_PCT, device=device)
    print("[train] OOF stats:", stats)

    with open(C.MODELS_DIR / "beast.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"[train] saved -> {C.MODELS_DIR / 'beast.pkl'}")

    # ---- HONEST signals for backtest: purged OOF probs (model never saw future).
    # Final refit model is used ONLY for the latest live signal below.
    import numpy as np
    pooled["oof_prob"] = stats["oof_prob"]
    oof_sig = pooled.pivot_table(index=pooled.index, columns="__sym", values="oof_prob").sort_index().ffill()
    sig_shifted = oof_sig.shift(1)  # trade next bar
    sig_shifted.to_parquet(C.RESULTS_DIR / "signals_oof.parquet")

    trades, metrics, eq = backtest_portfolio(sig_shifted, prices, initial=C.INITIAL_CAPITAL,
                                             top_k=C.TOP_K, fee_bps=C.FEE_BPS,
                                             min_prob=C.MIN_PROB, max_positions=C.MAX_POSITIONS)
    print("[backtest HONEST OOF]", metrics)
    eq.to_parquet(C.RESULTS_DIR / "equity.parquet")
    trades.to_parquet(C.RESULTS_DIR / "trades.parquet")

    # ---- live signals: final model (trained on all data) for TODAY only
    Xall = pooled[fcols].astype(float).values
    Xall = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    pooled["prob"] = model.predict_proba(pooled[fcols].fillna(0))
    sig = pooled.pivot_table(index=pooled.index, columns="__sym", values="prob").sort_index().ffill()
    sig.to_parquet(C.RESULTS_DIR / "signals.parquet")
    # latest signals
    latest = sig.iloc[-1].sort_values(ascending=False)
    print("\n=== LATEST SIGNALS (prob of >1.5% in 5d) ===")
    print(latest.round(3).to_string())
    latest.to_csv(C.RESULTS_DIR / "latest_signals.csv")


if __name__ == "__main__":
    main()
