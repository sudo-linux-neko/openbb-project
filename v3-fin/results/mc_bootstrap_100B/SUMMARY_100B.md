# 100B-step Monte Carlo — run log

Total steps: 100,000,000,000 (4 x 25B). Wall ~15.2 min (gbm 155s, heston 342s, merton 229s, regime 193s).

| model | mean_T | sd_T | p5/p50/p95 | VaR95 | CVaR95 | P(loss) | meanMaxDD | sps |
|---|---|---|---|---|---|---|---|---|
| gbm | 240.15 | 77.52 | 136.2/228.5/383.6 | -36.21% | -20.20% | 0.43% | 17.56% | 161092000 |
| heston | 240.13 | 76.89 | 133.7/230.4/379.1 | -33.69% | -14.79% | 0.84% | 20.28% | 73154400 |
| merton | 240.14 | 79.42 | 134.0/228.0/387.3 | -34.05% | -17.87% | 0.54% | 18.22% | 109118000 |
| regime | 240.21 | 77.47 | 136.2/228.6/383.5 | -36.18% | -20.12% | 0.44% | 17.72% | 129501000 |

S0=100, ~10y horizon (2500d), calibration from results/mc_bootstrap/calibration.json. Same seed 42 per model (independent per-thread streams).
