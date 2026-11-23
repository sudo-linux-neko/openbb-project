# Experiment ledger

- 001 baseline_frozen 50/50 (immutable, sha above) — main.py full run.
- 002 hedge_lab 4-way (baseline, defensive 0/100, duration 100/0, barbell 25/75) — pre-specified, no peeking, DSR trials=13.
- 003 risk diagnostics (beta/turnover/regime/crisis/drawdown) on baseline + defensive.
Rule: no new variants without writing hypothesis first. Next: paper-trade, T-cost calibration, compliance sign-off.
- 004 mc_bootstrap 30y 9eq B=500 MC=1008000 steps DSR=1.000 (2026-09-12).
- 004 mc_bootstrap 30y 65eq B=5000 MC=500000000 steps DSR=1.000 (2026-09-12).
- 005 mc_100B 4x25B=100B steps, gbm/heston/merton/regime, wall ~15.2min (2026-09-12).
