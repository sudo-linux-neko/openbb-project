# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Beast Trading ML — central config."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_cache"
MODELS_DIR = BASE_DIR / "models_saved"
RESULTS_DIR = BASE_DIR / "results"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Crypto + Stocks universe (user pick). Crypto uses yfinance '-USD' suffix.
EQUITIES = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "SPY", "QQQ"]
CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD"]

ALL_SYMBOLS = EQUITIES + CRYPTOS

# Market context symbols (always fetched for cross-asset features)
CONTEXT_SYMBOLS = ["SPY", "QQQ", "BTC-USD"]

# Swing setup
HORIZON = 5               # predict 5-day forward return
THRESHOLD = 0.015         # 1.5% move => class 1 (up) / class -1 (down < -1.5%) else 0
TEST_SIZE = 0.2           # last 20% walk-forward test
N_SPLITS = 5              # purged TimeSeriesSplit folds
EMBARGO_PCT = 0.01        # 1% embargo between train/test
SEQ_LEN = 30              # LSTM lookback
RANDOM_STATE = 42

# Backtest
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 8
TOP_K = 5                 # trade top-K signals per rebalance
LONG_ONLY = True
FEE_BPS = 5               # 5 bps commission + slippage each side
REBALANCE = "weekly"      # rebalance on signals weekly
MIN_PROB = 0.55           # min model prob to enter

# LSTM
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 40
LSTM_LR = 1e-3
LSTM_PATIENCE = 8
