# Hedge lab + risk diagnostics

_Baseline FROZEN at `baseline_frozen/` (sha: {'config.json': '4f7aeaac6017', 'compare.csv': 'd0323d4834c2', 'equity.csv': '266cf280fb90', 'REPORT.md': '9f8ff43e080a'}). Not investment advice. Employer compliance approval required before any live use._

## (a) Hedge-weight experiment (pre-specified, 4 trials)

Web basis: TLT 10y ≈ -2.15%/yr vs SHY ≈ +1.66%/yr; 2022 TLT -31.2% vs SHY -3.9%; TLT duration ≈15y/vol ≈14% — long bonds failed as hedge exactly in inflation shock. JPM/UBS/BlackRock 2025-26 notes: prefer short/medium maturities for defence; stock-bond correlation is regime-dependent.

| variant         |    TLT |    SHY |   CAGR |    Vol |   Sharpe |   Sortino |   MaxDD |   Calmar |   TotalRet |
|:----------------|-------:|-------:|-------:|-------:|---------:|----------:|--------:|---------:|-----------:|
| baseline_frozen |   0.5  |   0.5  | 0.0814 | 0.1027 |   0.8136 |    1.0386 | -0.2002 |   0.4066 |     2.6757 |
| defensive_0-100 |   0    |   1    | 0.0777 | 0.1013 |   0.7898 |    0.9438 | -0.1901 |   0.4088 |     2.4714 |
| duration_100-0  |   1    |   0    | 0.082  | 0.1028 |   0.8181 |    1.0615 | -0.2039 |   0.4022 |     2.7098 |
| barbell_25-75   |   0.25 |   0.75 | 0.0796 | 0.1021 |   0.8013 |    0.9892 | -0.1985 |   0.4009 |     2.5725 |
| bench_60-40     | nan    | nan    | 0.099  | 0.1044 |   0.9568 |    1.1778 | -0.2172 |   0.4558 |     3.8057 |

Baseline 50/50: CAGR 8.14%, Sharpe 0.81, MaxDD -20.02%.
Defensive 0/100 SHY: CAGR 7.77%, Sharpe 0.79, MaxDD -19.01%.
DSR (13 trials = 9 grid + 4 hedge): 1.000 (E[SR0]=0.03). Adding variants raises the bar — reported honestly.

Verdict rule (pre-committed): adopt a challenger ONLY if it wins on OOS Sharpe AND MaxDD without extra tuning. Otherwise keep baseline. No re-tuning after seeing table.

## (b) Risk diagnostics (baseline vs defensive)

Rolling beta (60d/120d OLS vs SPY), rolling vol/sharpe, turnover, regime + crisis splits, top-5 drawdowns — all trailing-only. Files: `risk_rolling_*.csv`, `risk_turnover_m_*.csv`, `risk_regime_*.csv`, `risk_crisis_*.csv`, `risk_drawdowns_*.csv`.

### Regime split — baseline

|          |   CAGR |    Vol |   Sharpe |   Sortino |   MaxDD | Peak                | Trough              |   Calmar |    Skew |   Kurt |   TotalRet |   Years |   BestDay |   WorstDay |    N |   days |   share |
|:---------|-------:|-------:|---------:|----------:|--------:|:--------------------|:--------------------|---------:|--------:|-------:|-----------:|--------:|----------:|-----------:|-----:|-------:|--------:|
| risk_on  | 0.0983 | 0.1122 |   0.892  |    1.1843 | -0.164  | 2021-11-08 00:00:00 | 2023-03-17 00:00:00 |   0.5996 | -0.5491 | 2.0763 |     2.4058 | 13.0675 |    0.0269 |    -0.041  | 3293 |   3293 |  0.7857 |
| risk_off | 0.0216 | 0.0552 |   0.4153 |    0.5219 | -0.1329 | 2020-03-09 00:00:00 | 2022-10-24 00:00:00 |   0.1627 |  0.2702 | 4.9099 |     0.0793 |  3.5635 |    0.021  |    -0.0147 |  898 |    898 |  0.2143 |

### Regime split — defensive (all-SHY)

|          |   CAGR |    Vol |   Sharpe |   Sortino |   MaxDD | Peak                | Trough              |   Calmar |    Skew |    Kurt |   TotalRet |   Years |   BestDay |   WorstDay |    N |   days |   share |
|:---------|-------:|-------:|---------:|----------:|--------:|:--------------------|:--------------------|---------:|--------:|--------:|-----------:|--------:|----------:|-----------:|-----:|-------:|--------:|
| risk_on  | 0.0986 | 0.1141 |   0.8818 |    1.1588 | -0.1801 | 2021-11-08 00:00:00 | 2023-03-17 00:00:00 |   0.5474 | -0.6171 |  2.6435 |     2.4174 | 13.0675 |    0.0269 |    -0.0522 | 3293 |   3293 |  0.7857 |
| risk_off | 0.0044 | 0.0116 |   0.3871 |    0.4481 | -0.0291 | 2022-03-01 00:00:00 | 2022-10-20 00:00:00 |   0.152  |  0.5555 | 10.4724 |     0.0158 |  3.5635 |    0.0054 |    -0.0041 |  898 |    898 |  0.2143 |

### Crisis windows — baseline vs 60/40

|                        |   strat_CAGR |   strat_MaxDD |   bench_CAGR |   bench_MaxDD |   excess |
|:-----------------------|-------------:|--------------:|-------------:|--------------:|---------:|
| COVID_crash_2020-02/03 |      -0.5386 |       -0.1053 |      -0.9213 |       -0.2172 |   0.1562 |
| Inflation_2022         |      -0.159  |       -0.1898 |      -0.1568 |       -0.2032 |  -0.0199 |
| Q4_2018                |      -0.4631 |       -0.1399 |      -0.397  |       -0.1135 |  -0.0274 |
| Taper_2013             |       0.0872 |       -0.0515 |       0.0053 |       -0.0474 |   0.0273 |

### Top drawdowns — baseline

|    | peak       | trough     | recovery   |   depth |   length_d |
|---:|:-----------|:-----------|:-----------|--------:|-----------:|
|  0 | 2021-11-08 | 2023-03-17 | 2024-06-17 | -0.2002 |        653 |
|  1 | 2018-10-01 | 2018-12-24 | 2020-02-11 | -0.1399 |        341 |
|  2 | 2011-07-07 | 2011-08-08 | 2012-02-09 | -0.1184 |        149 |
|  3 | 2020-02-19 | 2020-03-18 | 2020-08-26 | -0.1053 |        131 |
|  4 | 2025-02-19 | 2025-06-18 | 2025-12-04 | -0.0872 |        199 |

### How to read (institutional lens)

- Beta <1 in risk-off + lower MaxDD = hedge doing its job; beta spiking with market = crowded exit.
- Turnover/yr × cost = drag; if defensive wins only via lower turnover, prefer it structurally.
- Crisis table is the real exam: 2022 tells you about inflation-regime failure (TLT-heavy suffers).
- BIS-style backtest discipline: exceptions cluster in regime shifts — update vol/corr fast, don't dismiss clustered breaches as 'one-offs'.

Elapsed 0.0 min. See also `EXPERIMENTS.md`.
