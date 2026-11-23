# EXP-004: 30y + bootstrap + C++20 500M-step Monte Carlo

_Not investment advice. Employer compliance approval required before any live use._

- Window: 1995-01-03 -> 2026-09-01 (7969 days, 31.6y) | 65 equities (30y filter) + ETFs
- Strategy: baseline-frozen (SMA200 regime, top-20 12-1, 50/50 TLT/SHY, vol-target 10%, 6bps). Pre-inception ETF NaN = cash proxy (disclosed).
- Headline: strat CAGR 9.23% Sharpe 0.94 MaxDD -18.59% | 60/40 CAGR 7.94% Sharpe 0.72 MaxDD -34.70%
- DSR (17 trials): 1.000 (E[SR0]=0.02)

## Survivorship

30y results use current large-cap survivors with continuous data since 1995; delisted, bankrupt, and acquired names are absent, and early small-cap histories were illiquid. This overstates absolute return/Sharpe and understates drawdowns; treat long-horizon levels as upper bounds and judge the strategy on relative/breadth-robust metrics, not headline CAGR.

Dropped by 30y filter (1): ['GLD']

## Bootstrap (stationary b=21, B=5000, seed 42)

- Sharpe 95% CI [0.63, 1.24], median 0.94, P(<0)=0.0%, P(<bench 0.72)=8.6%
- CAGR median 9.23% CI [5.99%, 12.53%], MaxDD median -18.50%
- Block check: {"ret": {"Lstar": 4, "suggested_b": 10}, "absret": {"Lstar": null, "suggested_b": 20}, "T_cuberoot_rule": 30.0} | Sensitivity: circular_b21: Sharpe CI [0.62,1.25], stationary_b10: Sharpe CI [0.62,1.27], stationary_b63: Sharpe CI [0.66,1.20]
- Gate: require DSR>0.95 AND bootstrap q2.5(Sharpe)>0. Disagreement = reject.

## C++20 Monte Carlo — 500M steps (fin-market specific)

Engine `cpp/mc_risk.cpp` (C++20, xoshiro256++ per-thread, Box-Muller, full-truncation Heston Euler, Merton Poisson thinning, 2-state Markov regime; exact quantiles from stored terminal wealth). 4 x 50k paths x 2500d (~10y) = 500,000,000 steps.

| model | paths | steps | total_steps | mean_T | p5/p50/p95 | VaR95 | P(loss) | meanMaxDD |
|---|---|---|---|---|---|---|---|---|
| gbm | 50000 | 2500 | 125,000,000 | 240.3 | 136.9/228.6/383.7 | -36.94% | 0.4% | 17.53% |
| heston | 50000 | 2500 | 125,000,000 | 240.1 | 133.5/230.5/378.8 | -33.50% | 0.9% | 20.28% |
| merton | 50000 | 2500 | 125,000,000 | 240.5 | 134.4/228.8/388.0 | -34.41% | 0.6% | 18.22% |
| regime | 50000 | 2500 | 125,000,000 | 240.2 | 136.5/228.3/384.1 | -36.50% | 0.4% | 17.72% |

Calibration (moment-matched, disclosed approx): {"n": 7969, "skew_log": -0.4892024506983096, "exkurt_log": 3.135451845776521, "sigma_total": 0.09991666571731198, "sigma_dejumped": 0.09913982943898297}; Heston xi=0.22 rho=-0.13 Feller(2kθ−ξ²)=-0.007; Merton λ=0.30/yr μJ=-0.012; Regime bull 9.27%/8.4% bear 7.57%/13.6% p11=0.9765 p22=0.9327.

Validation: GBM mean vs closed form S0·exp(μT); C++ vs numpy pilot tolerance 1-2% (see `cpp/` README note); same-seed rerun bitwise-identical.

Elapsed 2.4 min. Artifacts in `results/mc_bootstrap/`.
