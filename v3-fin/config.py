# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Central configuration — single source of truth.

Institution-grade principle: no magic numbers scattered in code.
Everything tunable lives here, is logged at run start, and is saved
with results for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class Config:
    # --- Universe ---
    start_date: str = "2010-01-01"
    end_date: str = "2026-09-01"          # ~15.7 years
    # 100 single-name equities + 5 ETFs (regime / hedge / benchmark)
    # Full ticker list lives in universe.py
    provider: str = "yfinance"            # $0 budget -> free provider only
    interval: str = "1d"
    adjustment: str = "splits_and_dividends"  # total-return approximation

    # --- Strategy (deliberately FEW params to resist overfitting) ---
    momentum_lookback: int = 252         # 12m total return
    momentum_skip: int = 21              # skip last month (12-1 momentum)
    trend_window: int = 200              # long-term trend filter (DMA)
    top_n: int = 20                      # hold top-N momentum names
    rebalance_freq: str = "ME"           # month-end
    regime_symbol: str = "SPY"
    hedge_symbols: tuple = ("TLT", "SHY")  # risk-off allocation 50/50
    hedge_weights: tuple = (0.5, 0.5)

    # --- Risk ---
    vol_target_ann: float = 0.10         # 10% vol target (low-risk mandate)
    vol_window: int = 60                 # vol estimation window (days)
    max_gross: float = 1.0
    min_gross: float = 0.0
    max_name_weight: float = 0.05        # 5% cap per name at rebalance

    # --- Costs (conservative for large-cap US) ---
    cost_bps_per_side: float = 5.0       # commission + spread
    slippage_bps: float = 1.0

    # --- Benchmark: 60/40 SPY/AGG monthly-rebalanced ---
    benchmark_weights: dict = field(default_factory=lambda: {"SPY": 0.6, "AGG": 0.4})

    # --- Walk-forward anti-overfit ---
    train_years: int = 5
    test_years: int = 1
    purge_days: int = 21                 # embargo between train/test
    n_bootstrap: int = 1000

    # --- Infra ---
    cache_dir: Path = Path("data_cache")
    results_dir: Path = Path("results")
    batch_size: int = 20                 # tickers per progress-bar chunk
    request_pause: float = 0.5           # seconds between API calls
    max_retries: int = 3

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        d["results_dir"] = str(self.results_dir)
        path.write_text(json.dumps(d, indent=2, default=str))

    @property
    def total_cost_bps(self) -> float:
        return self.cost_bps_per_side + self.slippage_bps
