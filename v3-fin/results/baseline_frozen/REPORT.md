# Trend+Hedge vs 60/40 — backtest report

_Not investment advice. Research prototype. See Compliance note in README._

- Window: 2010-01-04 -> 2026-09-01 (4191 trading days)
- Universe: 97 equities + ETFs ['SPY', 'AGG', 'TLT', 'SHY', 'QQQ', 'GLD'] (post-coverage filter)
- Rules: SPY>SMA200 -> top-20 12-1 momentum above SMA; else 50/50 TLT/SHY. Monthly, 1-day lag, vol-target 10%.
- Costs: 6 bps/side incl. slippage. Per-name cap 5%.
- Time in risk-on: 78.6%

## Headline (net of costs, vol-scaled)

| | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| Strategy | 8.14% | 10.27% | 0.81 | -20.02% | 0.41 |
| 60/40 SPY/AGG | 9.90% | 10.44% | 0.96 | -21.72% | 0.46 |

## Overfit controls

- Params: only 3 (lookback=252, trend=200, top_n=20).
- Deflated Sharpe: 1.000 (null E[SR]=0.02 over 9 trials).
- Walk-forward: OOS Sharpe mean 0.83, median 1.15 over 12 test years; IS mean 0.87 (decay = 0.04).
- Bootstrap (block=21d, n=1000): Sharpe CI [0.41, 1.25], P(<0)=0.0%.
- Breadth: top-5 selected {'AVGO': 102, 'NVDA': 97, 'AMAT': 81, 'NOW': 78, 'AMD': 73} (check no 1-2 names dominate).

## Known biases (do not hide)

- Survivorship: static large-cap list backfilled; missing delisted names flatters equities.
- yfinance splits/dividends adjustment is approximate; borrow costs for shorts N/A (long-only).
- Past != future. Institutional use needs: paper-trade, execution/T-cost calibration, risk-committee sign-off, employer compliance approval.

Elapsed: 0.9 min. Artifacts in `results/`.
