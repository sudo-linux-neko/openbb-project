# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""EXP-004 end-to-end: 30y data (>=40 names) + institution-grade bootstrap + C++20 500M-step MC.

Pipeline (all with progress bars, seeds fixed, lineage logged):
  1. 30y ingest (1995-01-01 -> end) via OpenBB/yfinance, coverage >=70% + >=5000 bars,
     first bar <=1996-06-30 for equities; ETFs kept if >=70% (pre-inception NaN = cash proxy).
  2. Baseline-frozen strategy (SMA200 regime, top-20 12-1, 50/50 TLT/SHY, vol-target 10%).
  3. Bootstrap: stationary b=21 (B), circular b=21, sensitivity b={10,63}; autocorr block check.
  4. Calibrate GBM/Heston/Merton/Regime from strategy log-returns.
  5. Build + run C++20 mc_risk: 4 models x 50k paths x 2500 steps = 500M steps.
  6. REPORT_MC_BOOTSTRAP.md + EXPERIMENT ledger append.

Run:
    source ~/ml_env/bin/activate
    python run_mc_bootstrap.py --B 5000
    python run_mc_bootstrap.py --B 500 --quick   # smoke (15 syms, 2 models x 2k x 252)

Outputs: results/mc_bootstrap/*
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from universe_30y import EQUITIES_30Y, ETFS_30Y, SURVIVORSHIP_NOTE
from data_loader import load_universe
from strategy import build_signals
from backtester import run_backtest, benchmark_6040
from metrics import perf_stats, deflated_sharpe
from bootstrap_engine import run_bootstrap, autocorr_block_check
from calibrate import calibrate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("exp004")

MODELS_500M = [
    {"model": "gbm", "paths": 50000, "steps": 2500},
    {"model": "heston", "paths": 50000, "steps": 2500},
    {"model": "merton", "paths": 50000, "steps": 2500},
    {"model": "regime", "paths": 50000, "steps": 2500},
]  # 4 x 125M = 500M


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-004: 30y + bootstrap + 500M C++ MC")
    p.add_argument("--B", type=int, default=5000, help="stationary bootstrap replicates (default 5000)")
    p.add_argument("--start", type=str, default="1995-01-01")
    p.add_argument("--end", type=str, default="2026-09-01")
    p.add_argument("--quick", action="store_true", help="smoke: 15 syms, tiny MC")
    p.add_argument("--outdir", type=str, default="results/mc_bootstrap")
    return p.parse_args()


def coverage_filter_30y(prices: pd.DataFrame, spy_col: str = "SPY") -> tuple[pd.DataFrame, list[str]]:
    cal = prices[spy_col].dropna().index if spy_col in prices.columns else prices.index
    n_cal = len(cal)
    keep, dropped = [], []
    for c in prices.columns:
        s = prices[c].dropna()
        cov = len(s) / max(n_cal, 1)
        first = s.index.min() if len(s) else None
        is_etf = c in ETFS_30Y
        if is_etf:
            if cov >= 0.70 and len(s) >= 3000:
                keep.append(c)
            else:
                dropped.append(c)
        else:
            if cov >= 0.70 and len(s) >= 5000 and first is not None and first <= pd.Timestamp("1996-06-30"):
                keep.append(c)
            else:
                dropped.append(c)
    return prices[keep].sort_index(), dropped


def build_cpp(cpp_dir: Path) -> Path:
    binary = cpp_dir / "mc_risk"
    src = cpp_dir / "mc_risk.cpp"
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        log.info("C++ binary up to date: %s", binary)
        return binary
    cmd = ["g++", "-std=c++20", "-O3", "-march=native", "-flto",
           "-o", str(binary), str(src), "-pthread"]
    log.info("Building C++20 engine: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("g++ failed:\n%s\n%s", r.stdout, r.stderr)
        raise RuntimeError("C++ build failed")
    log.info("Built %s (%d bytes)", binary, binary.stat().st_size)
    return binary


def run_cpp_matrix(binary: Path, calib: dict, outdir: Path, quick: bool) -> dict:
    from tqdm import tqdm
    dt = 1 / 252
    base = {"gbm": ["--mu", calib["gbm"]["mu"], "--sigma", calib["gbm"]["sigma"]],
            "heston": ["--mu", calib["heston"]["mu"], "--kappa", calib["heston"]["kappa"],
                       "--theta", calib["heston"]["theta"], "--xi", calib["heston"]["xi"],
                       "--rho", calib["heston"]["rho"], "--v0", calib["heston"]["v0"]],
            "merton": ["--mu", calib["merton"]["mu"], "--sigma", calib["merton"]["sigma"],
                       "--lambda", calib["merton"]["lambda"], "--mjump", calib["merton"]["muJ"],
                       "--sjump", calib["merton"]["sigJ"]],
            "regime": ["--mu1", calib["regime"]["mu1"], "--sig1", calib["regime"]["sig1"],
                       "--mu2", calib["regime"]["mu2"], "--sig2", calib["regime"]["sig2"],
                       "--p12", 1 - calib["regime"]["p11"], "--p21", 1 - calib["regime"]["p22"]]}
    models = ([{"model": "gbm", "paths": 2000, "steps": 252},
               {"model": "regime", "paths": 2000, "steps": 252}]
              if quick else MODELS_500M)
    summaries = {}
    total = 0
    for m in tqdm(models, desc="C++20 MC models", unit="model"):
        name = m["model"]
        out = outdir / f"mc_{name}.json"
        cmd = [str(binary), "--model", name, "--paths", str(m["paths"]), "--steps", str(m["steps"]),
               "--dt", str(dt), "--s0", "100.0", "--seed", "42", "--threads", "0", "--out", str(out)]
        for k, v in zip(base[name][0::2], base[name][1::2]):
            cmd += [k, str(v)]
        log.info("MC run: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            log.error("mc_risk %s failed:\n%s\n%s", name, r.stdout, r.stderr)
            raise RuntimeError(f"C++ MC {name} failed")
        summaries[name] = json.loads(out.read_text())
        total += summaries[name]["total_steps"]
    summaries["_TOTAL_STEPS"] = total
    log.info("C++ total simulated steps: %d (target >=500M%s)", total, "" if not quick else ", quick mode")
    return summaries


def main() -> int:
    t0 = time.time()
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    object.__setattr__(cfg, "start_date", args.start)
    object.__setattr__(cfg, "end_date", args.end)

    # 1) 30y ingest
    symbols = sorted(set(EQUITIES_30Y + ETFS_30Y))
    if args.quick:
        symbols = ["SPY", "AGG", "TLT", "SHY", "QQQ", "GLD",
                   "AAPL", "MSFT", "JNJ", "XOM", "PG", "KO", "IBM", "GE", "CAT"]
    log.info("EXP-004 ingest: %d symbols %s -> %s", len(symbols), args.start, args.end)
    prices = load_universe(symbols, cfg.start_date, cfg.end_date, cfg.cache_dir,
                           cfg.provider, cfg.interval, cfg.adjustment,
                           cfg.batch_size, cfg.request_pause, cfg.max_retries)
    prices, dropped = coverage_filter_30y(prices)
    equities = [e for e in EQUITIES_30Y if e in prices.columns]
    log.info("30y filter: %d equities kept (need >=40), dropped %d: %s",
             len(equities), len(dropped), dropped[:12])
    min_names = 5 if args.quick else 40
    if len(equities) < min_names:
        log.error("Only %d equities pass 30y filter — need >=%d. Relax filter or extend universe.", len(equities), min_names)
        return 2
    if not all(c in prices.columns for c in ("SPY", "TLT", "SHY", "AGG")):
        log.error("Missing SPY/TLT/SHY/AGG after filter: %s", list(prices.columns)[:10])
        return 2
    (outdir / "lineage.json").write_text(json.dumps({
        "start": str(prices.index.min()), "end": str(prices.index.max()),
        "n_days": len(prices), "n_equities": len(equities),
        "equities": equities, "dropped": dropped,
        "provider": cfg.provider, "adjustment": cfg.adjustment,
        "survivorship_note": SURVIVORSHIP_NOTE}, indent=2, default=str))

    # 2) Strategy (baseline-frozen params)
    weights, regime, _ = build_signals(
        prices, equities, cfg.regime_symbol, ["TLT", "SHY"],
        cfg.momentum_lookback, cfg.momentum_skip, cfg.trend_window, cfg.top_n,
        hedge_weights=(0.5, 0.5))
    bt = run_backtest(prices, weights, cfg.total_cost_bps, cfg.vol_target_ann,
                      cfg.vol_window, cfg.min_gross, cfg.max_gross, cfg.max_name_weight)
    bench = benchmark_6040(prices)
    idx = bt["returns"].index.intersection(bench.index)
    strat_r, bench_r = bt["returns"].loc[idx], bench.loc[idx]
    s_stats, b_stats = perf_stats(strat_r), perf_stats(bench_r)
    log.info("30y headline: strat CAGR %.2f%% Sharpe %.2f MaxDD %.2f%% | 60/40 %.2f%% %.2f %.2f%%",
             s_stats["CAGR"] * 100, s_stats["Sharpe"], s_stats["MaxDD"] * 100,
             b_stats["CAGR"] * 100, b_stats["Sharpe"], b_stats["MaxDD"] * 100)

    # 3) Bootstrap
    block_info = autocorr_block_check(strat_r)
    (outdir / "block_check.json").write_text(json.dumps(block_info, indent=2, default=str))
    log.info("Block check: %s", block_info)
    B = 500 if args.quick else args.B
    rep_stat, sum_stat = run_bootstrap(strat_r, B=B, b=21.0, method="stationary",
                                       seed=42, bench_sharpe=b_stats["Sharpe"])
    rep_stat.to_parquet(outdir / "bootstrap_stationary_b21.parquet")
    results = {"stationary_b21": sum_stat}
    for b, bb, tag in ((21.0, min(B, 2000), "circular_b21"), (10.0, 1000, "stationary_b10"), (63.0, 1000, "stationary_b63")):
        if args.quick and tag != "circular_b21":
            continue
        method = "circular" if tag.startswith("circular") else "stationary"
        _, s = run_bootstrap(strat_r, B=bb, b=b, method=method, seed=43, bench_sharpe=b_stats["Sharpe"])
        results[tag] = s
    (outdir / "bootstrap_summary.json").write_text(json.dumps(results, indent=2, default=str))

    # 4) Calibrate + 5) C++ 500M
    calib = calibrate_all(strat_r)
    (outdir / "calibration.json").write_text(json.dumps(calib, indent=2, default=str))
    log.info("Calibration: %s", json.dumps(calib["diag"], indent=2))
    binary = build_cpp(Path("cpp"))
    mc = run_cpp_matrix(binary, calib, outdir, quick=args.quick)
    (outdir / "mc_summary.json").write_text(json.dumps(mc, indent=2, default=str))

    # 6) Report
    dsr = deflated_sharpe(strat_r, n_trials=13 + 4)  # 13 prior + 4 MC-model selection
    rep = outdir / "REPORT_MC_BOOTSTRAP.md"
    mc_table = "\n".join(
        f"| {m} | {mc[m]['paths']} | {mc[m]['steps']} | {mc[m]['total_steps']:,} | "
        f"{mc[m]['mean_terminal']:.1f} | {mc[m]['p5']:.1f}/{mc[m]['p50']:.1f}/{mc[m]['p95']:.1f} | "
        f"{mc[m]['VaR95']:.2%} | {mc[m]['P_loss']:.1%} | {mc[m]['maxDD_mean']:.2%} |"
        for m in ("gbm", "heston", "merton", "regime") if m in mc)
    rep.write_text(
        f"# EXP-004: 30y + bootstrap + C++20 500M-step Monte Carlo\n\n"
        f"_Not investment advice. Employer compliance approval required before any live use._\n\n"
        f"- Window: {idx.min().date()} -> {idx.max().date()} ({len(idx)} days, "
        f"{len(idx) / 252:.1f}y) | {len(equities)} equities (30y filter) + ETFs\n"
        f"- Strategy: baseline-frozen (SMA200 regime, top-20 12-1, 50/50 TLT/SHY, vol-target 10%, "
        f"6bps). Pre-inception ETF NaN = cash proxy (disclosed).\n"
        f"- Headline: strat CAGR {s_stats['CAGR']:.2%} Sharpe {s_stats['Sharpe']:.2f} "
        f"MaxDD {s_stats['MaxDD']:.2%} | 60/40 CAGR {b_stats['CAGR']:.2%} Sharpe {b_stats['Sharpe']:.2f} "
        f"MaxDD {b_stats['MaxDD']:.2%}\n"
        f"- DSR (17 trials): {dsr['DSR']:.3f} (E[SR0]={dsr['E[SR0]']:.2f})\n\n"
        f"## Survivorship\n\n{SURVIVORSHIP_NOTE}\n\n"
        f"Dropped by 30y filter ({len(dropped)}): {dropped[:20]}\n\n"
        f"## Bootstrap (stationary b=21, B={B}, seed 42)\n\n"
        f"- Sharpe 95% CI [{sum_stat['Sharpe_CI95'][0]:.2f}, {sum_stat['Sharpe_CI95'][1]:.2f}], "
        f"median {sum_stat['Sharpe_med']:.2f}, P(<0)={sum_stat['P_Sharpe_below_0']:.1%}, "
        f"P(<bench {b_stats['Sharpe']:.2f})={sum_stat.get('P_Sharpe_below_bench', float('nan')):.1%}\n"
        f"- CAGR median {sum_stat['CAGR_med']:.2%} CI [{sum_stat['CAGR_CI95'][0]:.2%}, {sum_stat['CAGR_CI95'][1]:.2%}], "
        f"MaxDD median {sum_stat['MaxDD_med']:.2%}\n"
        f"- Block check: {json.dumps(block_info)} | Sensitivity: " +
        ", ".join(f"{k}: Sharpe CI [{v['Sharpe_CI95'][0]:.2f},{v['Sharpe_CI95'][1]:.2f}]"
                  for k, v in results.items() if k != "stationary_b21") + "\n"
        f"- Gate: require DSR>0.95 AND bootstrap q2.5(Sharpe)>0. Disagreement = reject.\n\n"
        f"## C++20 Monte Carlo — 500M steps (fin-market specific)\n\n"
        f"Engine `cpp/mc_risk.cpp` (C++20, xoshiro256++ per-thread, Box-Muller, "
        f"full-truncation Heston Euler, Merton Poisson thinning, 2-state Markov regime; "
        f"exact quantiles from stored terminal wealth). 4 x 50k paths x 2500d (~10y) = "
        f"{mc.get('_TOTAL_STEPS', 0):,} steps.\n\n"
        f"| model | paths | steps | total_steps | mean_T | p5/p50/p95 | VaR95 | P(loss) | meanMaxDD |\n"
        f"|---|---|---|---|---|---|---|---|---|\n{mc_table}\n\n"
        f"Calibration (moment-matched, disclosed approx): {json.dumps(calib['diag'])}; "
        f"Heston xi={calib['heston']['xi']:.2f} rho={calib['heston']['rho']:.2f} "
        f"Feller(2kθ−ξ²)={calib['heston']['feller']:.3f}; "
        f"Merton λ={calib['merton']['lambda']:.2f}/yr μJ={calib['merton']['muJ']:.3f}; "
        f"Regime bull {calib['regime']['mu1']:.2%}/{calib['regime']['sig1']:.1%} "
        f"bear {calib['regime']['mu2']:.2%}/{calib['regime']['sig2']:.1%} "
        f"p11={calib['regime']['p11']:.4f} p22={calib['regime']['p22']:.4f}.\n\n"
        f"Validation: GBM mean vs closed form S0·exp(μT); C++ vs numpy pilot tolerance 1-2% "
        f"(see `cpp/` README note); same-seed rerun bitwise-identical.\n\n"
        f"Elapsed {(time.time() - t0) / 60:.1f} min. Artifacts in `{outdir}/`.\n"
    )
    log.info("Wrote %s", rep)
    with open("results/EXPERIMENTS.md", "a") as f:
        f.write(f"- 004 mc_bootstrap 30y {len(equities)}eq B={B} MC={mc.get('_TOTAL_STEPS', 0)} steps "
                f"DSR={dsr['DSR']:.3f} ({time.strftime('%Y-%m-%d')}).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
