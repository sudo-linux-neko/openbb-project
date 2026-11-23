# v3-fin — Regime-filtered trend + bond hedge (OpenBB, $0 data)
Research prototype. NOT investment advice. NOT a live trading bot.

## What this is
Institution-*style* research pipeline (process, not promises):
`OpenBB/yfinance ingest (cached)` → `3-parameter trend+hedge signals` →
`cost-aware vol-targeted backtest` → `60/40 benchmark` →
`walk-forward + bootstrap + Deflated Sharpe` → `REPORT.md`.

Complexity is in validation, not in rules. Few params = harder to overfit.

## Goldman / employer compliance (read first)
If you are subject to personal-trading / outside-activity / IP rules, get
written compliance approval BEFORE running live, paper-trading on firm
systems, or sharing results externally. This repo must stay personal-side,
no firm data, no firm infra, no client implications.

## Setup (you asked for this env)
```bash
source ~/ml_env/bin/activate
python -c "from openbb import obb; print('openbb ok')"
```

## Runs
```bash
# 2-min smoke test (15 symbols, no robustness)
python main.py --quick

# Full 15y+ x 100+ run with progress bars + robustness (~10-20 min first time, ~2 min cached)
python main.py --max-symbols 105
```

## Outputs (`results/`)
`REPORT.md`, `compare.csv`, `equity.csv`, `walkforward.csv`,
`bootstrap.json`, `breadth.csv`, `weights_last.csv`, `config.json`.

## Why low-risk / beat-bonds can't be promised
- Low MaxDD + high return + 100 tickers + 15y is a trilemma; vol-targeting
  buys lower risk by sacrificing upside.
- Static large-cap universe has survivorship bias (flatters backtest).
- Free yfinance data has corporate-action noise; costs are modeled, not measured.
- Any further tuning after seeing results = overfitting. Trust OOS, not IS.
