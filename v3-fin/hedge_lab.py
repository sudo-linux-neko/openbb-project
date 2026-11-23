# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Hedge lab + risk diagnostics — baseline stays FROZEN.

Pre-specified hedge variants (4 total, documented BEFORE running):
  baseline_frozen  50/50 TLT/SHY  (do not touch)
  defensive        0/100  (all SHY — short-duration, 2022-resilient thesis)
  duration         100/0  (all TLT — deflation-crash convexity thesis)
  barbell_lite     25/75  (middle ground)

Web context baked in: TLT 10y -2.15%/yr vs SHY +1.66%/yr; 2022 TLT -31.2%
vs SHY -3.9%; TLT duration ~15y, vol ~14%. Testing SHY tilt is a
defensive hypothesis, not a fit.

Run:
    source ~/ml_env/bin/activate
    python hedge_lab.py
Reads data_cache/*.parquet (no re-download). Writes results/hedge_*,
results/risk_*, results/REPORT_HEDGE_RISK.md, results/EXPERIMENTS.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import Config
from strategy import build_signals
from backtester import run_backtest, benchmark_6040
from metrics import perf_stats, deflated_sharpe
from risk_diagnostics import (rolling_beta, rolling_vol, rolling_sharpe,
                              turnover_stats, regime_split, crisis_windows,
                              top_drawdowns)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hedge-lab")

VARIANTS: list[dict] = [
    {"name": "baseline_frozen", "tlt": 0.50, "shy": 0.50},
    {"name": "defensive_0-100", "tlt": 0.00, "shy": 1.00},
    {"name": "duration_100-0", "tlt": 1.00, "shy": 0.00},
    {"name": "barbell_25-75", "tlt": 0.25, "shy": 0.75},
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def freeze_baseline(results: Path) -> Path:
    frozen = results / "baseline_frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "compare.csv", "equity.csv", "REPORT.md"):
        src = results / f
        if src.exists():
            shutil.copy2(src, frozen / f)
    manifest = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {f: sha(frozen / f) for f in ("config.json", "compare.csv",
                                                       "equity.csv", "REPORT.md")
                          if (frozen / f).exists()},
                "note": "Baseline immutable. Hedge variants compared AGAINST this, never re-tuned."}
    (frozen / "FROZEN.json").write_text(json.dumps(manifest, indent=2))
    log.info("Baseline frozen: %s", manifest["files"])
    return frozen


def load_cached_prices(cache_dir: Path) -> pd.DataFrame:
    fps = [f for f in cache_dir.glob("*.parquet")]
    if len(fps) < 10:
        raise RuntimeError(f"Only {len(fps)} cached symbols — run main.py first.")
    closes = {}
    for f in tqdm(fps, desc="Load cache", unit="sym"):
        try:
            closes[f.stem.replace("_", "-").replace("BRK-B", "BRK-B")] = pd.read_parquet(f)["close"]
        except Exception as e:  # noqa: BLE001
            log.warning("%s unreadable: %s", f.name, e)
    # fix BRK-B stem mangling (BRK_B -> BRK-B)
    closes = {("BRK-B" if k == "BRK-B" else k): v for k, v in closes.items()}
    # our cache saved BRK-B as BRK_B.parquet; restore ticker
    out = {}
    for k, v in closes.items():
        out["BRK-B" if k == "BRK-B" or k == "BRK_B" else k] = v
    prices = pd.DataFrame(out).sort_index()
    # normalize BRK_B column if present
    if "BRK_B" in prices.columns and "BRK-B" not in prices.columns:
        prices = prices.rename(columns={"BRK_B": "BRK-B"})
    log.info("Cached prices: %s rows x %s cols (%s -> %s)",
             len(prices), prices.shape[1], prices.index.min().date(), prices.index.max().date())
    return prices


def main() -> int:
    t0 = time.time()
    cfg = Config()
    results, cache = cfg.results_dir, cfg.cache_dir
    results.mkdir(parents=True, exist_ok=True)
    freeze_baseline(results)
    prices = load_cached_prices(cache)

    must_etf = ["SPY", "AGG", "TLT", "SHY"]
    for m in must_etf:
        if m not in prices.columns:
            log.error("Missing %s in cache — run main.py --max-symbols 105 first.", m)
            return 2
    from universe import EQUITIES_100
    equities = [e for e in EQUITIES_100 if e in prices.columns]
    log.info("Equities in test: %d", len(equities))
    bench = benchmark_6040(prices)

    # --- (a) Hedge variants ---
    rows, curves = [], {}
    for v in tqdm(VARIANTS, desc="Hedge variants", unit="mix"):
        w, regime, _ = build_signals(
            prices, equities, cfg.regime_symbol, ["TLT", "SHY"],
            cfg.momentum_lookback, cfg.momentum_skip, cfg.trend_window,
            cfg.top_n, hedge_weights=(v["tlt"], v["shy"]))
        bt = run_backtest(prices, w, cfg.total_cost_bps, cfg.vol_target_ann,
                          cfg.vol_window, cfg.min_gross, cfg.max_gross, cfg.max_name_weight)
        idx = bt["returns"].index.intersection(bench.index)
        r = bt["returns"].loc[idx]
        s = perf_stats(r)
        rows.append({"variant": v["name"], "TLT": v["tlt"], "SHY": v["shy"],
                     **{k: s[k] for k in ("CAGR", "Vol", "Sharpe", "Sortino",
                                          "MaxDD", "Calmar", "TotalRet")}})
        curves[v["name"]] = (r, bt, w, regime)
        log.info("%-16s CAGR %6.2f%% Vol %5.2f%% Sharpe %4.2f MaxDD %6.2f%%",
                 v["name"], s["CAGR"] * 100, s["Vol"] * 100, s["Sharpe"], s["MaxDD"] * 100)
    comp = pd.DataFrame(rows).set_index("variant")
    b = perf_stats(bench.loc[curves["baseline_frozen"][0].index])
    comp.loc["bench_60-40"] = {"TLT": float("nan"), "SHY": float("nan"),
                               "CAGR": b["CAGR"], "Vol": b["Vol"], "Sharpe": b["Sharpe"],
                               "Sortino": b["Sortino"], "MaxDD": b["MaxDD"],
                               "Calmar": b["Calmar"], "TotalRet": b["TotalRet"]}
    comp.to_csv(results / "hedge_comparison.csv")
    pd.DataFrame({k: (1 + v[0]).cumprod() for k, v in curves.items()}
                 ).to_csv(results / "hedge_equity.csv")

    # DSR honesty: 9 (walk-forward grid) + 4 (hedge mixes) = 13 trials
    dsr = deflated_sharpe(curves["baseline_frozen"][0], n_trials=13)
    (results / "hedge_dsr.json").write_text(json.dumps(dsr, indent=2))

    # --- (b) Risk diagnostics on baseline + most-defensive challenger ---
    diag_targets = ["baseline_frozen", "defensive_0-100"]
    spy_ret = prices["SPY"].pct_change().fillna(0.0)
    for name in diag_targets:
        r, bt, w, regime = curves[name]
        idx = r.index
        beta60 = rolling_beta(r, spy_ret.loc[idx], 60)
        beta120 = rolling_beta(r, spy_ret.loc[idx], 120)
        pd.DataFrame({"beta_60d": beta60, "beta_120d": beta120,
                      "vol_60d": rolling_vol(r), "sharpe_252d": rolling_sharpe(r),
                      "gross": bt["gross"].loc[idx]}).to_csv(results / f"risk_rolling_{name}.csv")
        ts = turnover_stats(w, r, cfg.total_cost_bps)
        ts["monthly_turnover"].rename("turnover").to_csv(results / f"risk_turnover_m_{name}.csv")
        regime_split(r, regime.loc[idx]).to_csv(results / f"risk_regime_{name}.csv")
        crisis_windows(r, bench.loc[idx]).to_csv(results / f"risk_crisis_{name}.csv")
        top_drawdowns((1 + r).cumprod()).to_csv(results / f"risk_drawdowns_{name}.csv", index=False)
        log.info("[%s] beta60=%.2f beta120=%.2f turnover/yr=%.1fx cost_drag=%.2f%%/yr",
                 name, beta60.median(), beta120.median(),
                 ts["turnover_annual_mean"], ts["cost_drag_ann"] * 100)

    # --- Report ---
    base, defe = comp.loc["baseline_frozen"], comp.loc["defensive_0-100"]
    rs_base = pd.read_csv(results / "risk_regime_baseline_frozen.csv", index_col=0)
    rs_def = pd.read_csv(results / "risk_regime_defensive_0-100.csv", index_col=0)
    cw_base = pd.read_csv(results / "risk_crisis_baseline_frozen.csv", index_col=0)
    dd_base = pd.read_csv(results / "risk_drawdowns_baseline_frozen.csv")
    rep = results / "REPORT_HEDGE_RISK.md"
    rep.write_text(
        f"# Hedge lab + risk diagnostics\n\n"
        f"_Baseline FROZEN at `baseline_frozen/` (sha: "
        f"{json.loads((results / 'baseline_frozen' / 'FROZEN.json').read_text())['files']}). "
        f"Not investment advice. Employer compliance approval required before any live use._\n\n"
        f"## (a) Hedge-weight experiment (pre-specified, 4 trials)\n\n"
        f"Web basis: TLT 10y ≈ -2.15%/yr vs SHY ≈ +1.66%/yr; 2022 TLT -31.2% vs SHY -3.9%; "
        f"TLT duration ≈15y/vol ≈14% — long bonds failed as hedge exactly in inflation shock. "
        f"JPM/UBS/BlackRock 2025-26 notes: prefer short/medium maturities for defence; "
        f"stock-bond correlation is regime-dependent.\n\n"
        f"{comp.round(4).to_markdown()}\n\n"
        f"Baseline 50/50: CAGR {base['CAGR']:.2%}, Sharpe {base['Sharpe']:.2f}, "
        f"MaxDD {base['MaxDD']:.2%}.\n"
        f"Defensive 0/100 SHY: CAGR {defe['CAGR']:.2%}, Sharpe {defe['Sharpe']:.2f}, "
        f"MaxDD {defe['MaxDD']:.2%}.\n"
        f"DSR (13 trials = 9 grid + 4 hedge): {dsr['DSR']:.3f} (E[SR0]={dsr['E[SR0]']:.2f}). "
        f"Adding variants raises the bar — reported honestly.\n\n"
        f"Verdict rule (pre-committed): adopt a challenger ONLY if it wins on OOS Sharpe "
        f"AND MaxDD without extra tuning. Otherwise keep baseline. No re-tuning after seeing table.\n\n"
        f"## (b) Risk diagnostics (baseline vs defensive)\n\n"
        f"Rolling beta (60d/120d OLS vs SPY), rolling vol/sharpe, turnover, regime + crisis splits, "
        f"top-5 drawdowns — all trailing-only. Files: `risk_rolling_*.csv`, `risk_turnover_m_*.csv`, "
        f"`risk_regime_*.csv`, `risk_crisis_*.csv`, `risk_drawdowns_*.csv`.\n\n"
        f"### Regime split — baseline\n\n{rs_base.round(4).to_markdown()}\n\n"
        f"### Regime split — defensive (all-SHY)\n\n{rs_def.round(4).to_markdown()}\n\n"
        f"### Crisis windows — baseline vs 60/40\n\n{cw_base.round(4).to_markdown()}\n\n"
        f"### Top drawdowns — baseline\n\n{dd_base.round(4).to_markdown()}\n\n"
        f"### How to read (institutional lens)\n\n"
        f"- Beta <1 in risk-off + lower MaxDD = hedge doing its job; beta spiking with market = crowded exit.\n"
        f"- Turnover/yr × cost = drag; if defensive wins only via lower turnover, prefer it structurally.\n"
        f"- Crisis table is the real exam: 2022 tells you about inflation-regime failure (TLT-heavy suffers).\n"
        f"- BIS-style backtest discipline: exceptions cluster in regime shifts — update vol/corr fast, "
        f"don't dismiss clustered breaches as 'one-offs'.\n\n"
        f"Elapsed {(time.time() - t0) / 60:.1f} min. See also `EXPERIMENTS.md`.\n"
    )
    (results / "EXPERIMENTS.md").write_text(
        f"# Experiment ledger\n\n"
        f"- 001 baseline_frozen 50/50 (immutable, sha above) — main.py full run.\n"
        f"- 002 hedge_lab 4-way (baseline, defensive 0/100, duration 100/0, barbell 25/75) — "
        f"pre-specified, no peeking, DSR trials=13.\n"
        f"- 003 risk diagnostics (beta/turnover/regime/crisis/drawdown) on baseline + defensive.\n"
        f"Rule: no new variants without writing hypothesis first. Next: paper-trade, T-cost calibration, "
        f"compliance sign-off.\n"
    )
    log.info("Wrote %s + hedge_comparison.csv + risk_*.csv", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
