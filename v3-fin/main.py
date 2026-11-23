# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Entry point: OpenBB -> signals -> backtest -> 60/40 -> robustness -> report.

Run:
    source ~/ml_env/bin/activate
    python main.py --help
    python main.py --max-symbols 105          # full 15y x 100+ run
    python main.py --max-symbols 15 --no-robust  # 2-min smoke test

Outputs (results/): config.json, compare.csv, walkforward.csv, bootstrap.json,
equity.csv, weights_last.csv, breadth.csv, REPORT.md
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import Config
from universe import EQUITIES_100, ALL_SYMBOLS
from data_loader import load_universe
from strategy import build_signals
from backtester import run_backtest, benchmark_6040
from metrics import perf_stats, compare_table, deflated_sharpe, per_ticker_breadth
from robustness import walk_forward, block_bootstrap_ci

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("v3-fin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Institution-style trend+hedge backtester (OpenBB, $0 data)")
    p.add_argument("--max-symbols", type=int, default=105,
                   help="cap universe size (default 105 = 100 eq + 5 etf)")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--trend", type=int, default=None)
    p.add_argument("--no-robust", action="store_true", help="skip walk-forward + bootstrap")
    p.add_argument("--quick", action="store_true", help="alias: 15 symbols, no robust")
    return p.parse_args()


def main() -> int:
    t0 = time.time()
    args = parse_args()
    cfg = Config()
    if args.start: object.__setattr__(cfg, "start_date", args.start)
    if args.end: object.__setattr__(cfg, "end_date", args.end)
    if args.top_n: object.__setattr__(cfg, "top_n", args.top_n)
    if args.trend: object.__setattr__(cfg, "trend_window", args.trend)
    if args.quick:
        args.max_symbols = 15
        args.no_robust = True

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.results_dir / "config.json")
    log.info("Config: %s -> %s | provider=%s | vol_target=%.0f%% | cost=%.0fbps",
             cfg.start_date, cfg.end_date, cfg.provider,
             cfg.vol_target_ann * 100, cfg.total_cost_bps)

    # 1) Universe (ETFs first so regime/hedge/bench never get capped out)
    must = ["SPY", "AGG", "TLT", "SHY", "QQQ", "GLD"]
    eq = [s for s in EQUITIES_100 if s not in must]
    symbols = must + eq
    symbols = symbols[:args.max_symbols] if args.max_symbols else symbols
    log.info("Universe: %d symbols (%s...)", len(symbols), symbols[:6])

    # 2) Data (OpenBB + tqdm + parquet cache)
    prices = load_universe(symbols, cfg.start_date, cfg.end_date, cfg.cache_dir,
                           cfg.provider, cfg.interval, cfg.adjustment,
                           cfg.batch_size, cfg.request_pause, cfg.max_retries)
    equities = [s for s in eq if s in prices.columns]
    log.info("Kept %d equities after coverage filter", len(equities))
    if cfg.regime_symbol not in prices.columns or "AGG" not in prices.columns:
        log.error("Missing SPY/AGG — cannot build regime or 60/40. Columns: %s",
                  list(prices.columns)[:10])
        return 2

    # 3) Signals + backtest
    weights, regime, month_ends = build_signals(
        prices, equities, cfg.regime_symbol, list(cfg.hedge_symbols),
        cfg.momentum_lookback, cfg.momentum_skip, cfg.trend_window, cfg.top_n)
    bt = run_backtest(prices, weights, cfg.total_cost_bps, cfg.vol_target_ann,
                      cfg.vol_window, cfg.min_gross, cfg.max_gross, cfg.max_name_weight)
    bench = benchmark_6040(prices)
    idx = bt["returns"].index.intersection(bench.index)
    strat_r, bench_r = bt["returns"].loc[idx], bench.loc[idx]

    comp = compare_table(strat_r, bench_r)
    comp.to_csv(cfg.results_dir / "compare.csv")
    pd.DataFrame({"strategy": (1 + strat_r).cumprod(),
                  "bench_6040": (1 + bench_r).cumprod(),
                  "gross": bt["gross"]}).to_csv(cfg.results_dir / "equity.csv")
    bt["weights"].iloc[[-1]].to_csv(cfg.results_dir / "weights_last.csv")
    log.info("\n%s", comp.round(4).to_string())

    # 4) Overfit diagnostics (full run only)
    dsr = deflated_sharpe(strat_r, n_trials=9)  # 3 trend x 3 topn grid
    breadth = per_ticker_breadth(prices, equities, cfg.momentum_lookback, cfg.momentum_skip)
    breadth.to_csv(cfg.results_dir / "breadth.csv")
    wf, boot = None, None
    if not args.no_robust:
        wf = walk_forward(prices, equities, cfg)
        wf.to_csv(cfg.results_dir / "walkforward.csv", index=False)
        boot = block_bootstrap_ci(strat_r, n=cfg.n_bootstrap)
        import json as _j
        (cfg.results_dir / "bootstrap.json").write_text(_j.dumps(boot, indent=2))
        log.info("Walk-forward OOS Sharpe mean=%.2f median=%.2f | IS mean=%.2f",
                 wf["OOS_Sharpe"].mean(), wf["OOS_Sharpe"].median(), wf["IS_Sharpe"].mean())
        log.info("Bootstrap Sharpe 95%% CI [%.2f, %.2f], P(Sharpe<0)=%.1f%%",
                 boot["p2.5"], boot["p97.5"], boot["P(Sharpe<0)"] * 100)
    log.info("DSR=%.3f (E[SR0]=%.2f | %d trials)", dsr["DSR"], dsr["E[SR0]"], dsr["trials"])

    # 5) Markdown report
    s_stats, b_stats = perf_stats(strat_r), perf_stats(bench_r)
    risk_on = float(regime.loc[idx].mean())
    rep = cfg.results_dir / "REPORT.md"
    rep.write_text(
        f"# Trend+Hedge vs 60/40 — backtest report\n\n"
        f"_Not investment advice. Research prototype. See Compliance note in README._\n\n"
        f"- Window: {idx.min().date()} -> {idx.max().date()} ({len(idx)} trading days)\n"
        f"- Universe: {len(equities)} equities + ETFs {must} (post-coverage filter)\n"
        f"- Rules: SPY>SMA{cfg.trend_window} -> top-{cfg.top_n} 12-1 momentum above SMA; "
        f"else 50/50 TLT/SHY. Monthly, 1-day lag, vol-target {cfg.vol_target_ann:.0%}.\n"
        f"- Costs: {cfg.total_cost_bps:.0f} bps/side incl. slippage. Per-name cap {cfg.max_name_weight:.0%}.\n"
        f"- Time in risk-on: {risk_on:.1%}\n\n"
        f"## Headline (net of costs, vol-scaled)\n\n"
        f"| | CAGR | Vol | Sharpe | MaxDD | Calmar |\n|---|---|---|---|---|---|\n"
        f"| Strategy | {s_stats['CAGR']:.2%} | {s_stats['Vol']:.2%} | {s_stats['Sharpe']:.2f} "
        f"| {s_stats['MaxDD']:.2%} | {s_stats['Calmar']:.2f} |\n"
        f"| 60/40 SPY/AGG | {b_stats['CAGR']:.2%} | {b_stats['Vol']:.2%} | {b_stats['Sharpe']:.2f} "
        f"| {b_stats['MaxDD']:.2%} | {b_stats['Calmar']:.2f} |\n\n"
        f"## Overfit controls\n\n"
        f"- Params: only 3 (lookback={cfg.momentum_lookback}, trend={cfg.trend_window}, top_n={cfg.top_n}).\n"
        f"- Deflated Sharpe: {dsr['DSR']:.3f} (null E[SR]={dsr['E[SR0]']:.2f} over {dsr['trials']} trials).\n"
        + (f"- Walk-forward: OOS Sharpe mean {wf['OOS_Sharpe'].mean():.2f}, "
           f"median {wf['OOS_Sharpe'].median():.2f} over {len(wf)} test years; "
           f"IS mean {wf['IS_Sharpe'].mean():.2f} (decay = "
           f"{wf['IS_Sharpe'].mean() - wf['OOS_Sharpe'].mean():.2f}).\n"
           f"- Bootstrap (block=21d, n={cfg.n_bootstrap}): Sharpe CI "
           f"[{boot['p2.5']:.2f}, {boot['p97.5']:.2f}], P(<0)={boot['P(Sharpe<0)']:.1%}.\n"
           if wf is not None else "- Robustness skipped (--no-robust / --quick).\n")
        + f"- Breadth: top-5 selected {breadth['months_selected'].head(5).to_dict()} "
        f"(check no 1-2 names dominate).\n\n"
        f"## Known biases (do not hide)\n\n"
        f"- Survivorship: static large-cap list backfilled; missing delisted names flatters equities.\n"
        f"- yfinance splits/dividends adjustment is approximate; borrow costs for shorts N/A (long-only).\n"
        f"- Past != future. Institutional use needs: paper-trade, execution/T-cost calibration, "
        f"risk-committee sign-off, employer compliance approval.\n\n"
        f"Elapsed: {(time.time() - t0) / 60:.1f} min. Artifacts in `{cfg.results_dir}/`.\n"
    )
    log.info("Wrote %s (%.1f min total)", rep, (time.time() - t0) / 60)
    # Honest console verdict (no hype)
    if s_stats["MaxDD"] < b_stats["MaxDD"] and s_stats["Sharpe"] > b_stats["Sharpe"]:
        log.info("VERDICT: strategy beat 60/40 on Sharpe AND MaxDD in-sample — "
                 "confirm OOS/walk-forward before any capital claims.")
    else:
        log.info("VERDICT: did NOT dominate 60/40 on both risk and return — "
                 "prefer honesty over tuning further (tuning = overfit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
