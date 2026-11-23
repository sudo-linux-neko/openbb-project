# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

import numpy as np
import pandas as pd
from openbb import obb

print("🦁 Step 1: Loading raw asset classes from OpenBB...")
tickers = ["AAPL", "MSFT", "JPM", "XOM"]
data_dict = {}

for ticker in tickers:
    res = obb.equity.price.historical(symbol=ticker, provider="yfinance", limit=750)
    df = res.to_df()
    data_dict[ticker] = df["close"]

prices_df = pd.DataFrame(data_dict).dropna()
returns_df = prices_df.pct_change().dropna()

print("📊 Step 2: Math engine calculating advanced risk analytics...")
# 1. Allocation Vector Matrix
weights = np.array([1.0 / len(tickers)] * len(tickers))
portfolio_daily_returns = returns_df.dot(weights)

# 2. Performance Aggregates
total_p_return = (prices_df.iloc[-1] / prices_df.iloc[0]) - 1
ann_returns = returns_df.mean() * 252
ann_vols = returns_df.std() * np.sqrt(252)

p_ann_return = portfolio_daily_returns.mean() * 252
p_ann_vol = portfolio_daily_returns.std() * np.sqrt(252)

# 3. Sharpe & Sortino Formulations
rf_rate = 0.04  # Assuming standard 4% risk-free rate proxy
sharpe = (p_ann_return - rf_rate) / p_ann_vol

downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
sortino = (p_ann_return - rf_rate) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

# 4. Value at Risk (VaR 95% Historical Simulation Method)
var_95 = np.percentile(portfolio_daily_returns, 5)

# 5. Maximum Drawdown Calculation
cumulative_returns = (1 + portfolio_daily_returns).cumprod()
running_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns - running_max) / running_max
max_drawdown = drawdowns.min()

print("🏆 Calculations finished! Generating production-level report structure...")

# Advanced Responsive HTML Dashboard UI Generation
html_dashboard = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Institutional Risk Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 30px; background-color: #0f172a; color: #f8fafc; }}
        .wrapper {{ max-width: 1200px; margin: auto; }}
        h1 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 15px; margin-bottom: 30px; font-weight: 600; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; text-align: center; }}
        .card h3 {{ margin: 0 0 10px 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #f1f5f9; }}
        .card .value.green {{ color: #4ade80; }}
        .card .value.red {{ color: #f87171; }}
        table {{ width: 100%; border-collapse: collapse; margin: 30px 0; background: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid #334155; }}
        th, td {{ padding: 15px; text-align: left; }}
        th {{ background: #0f172a; color: #38bdf8; font-weight: 600; border-bottom: 2px solid #334155; }}
        tr {{ border-bottom: 1px solid #334155; }}
        tr:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="wrapper">
        <h1>📊 Institutional Portfolio Risk & Performance Ledger</h1>
        
        <!-- TOP LEVEL EXECUTIVE KPI BLOCK -->
        <div class="grid">
            <div class="card">
                <h3>Annualized Return</h3>
                <div class="value green">{p_ann_return*100:.2f}%</div>
            </div>
            <div class="card">
                <h3>Portfolio Volatility</h3>
                <div class="value">{p_ann_vol*100:.2f}%</div>
            </div>
            <div class="card">
                <h3>Sharpe Ratio (Rf=4%)</h3>
                <div class="value green">{sharpe:.2f}</div>
            </div>
            <div class="card">
                <h3>Sortino Ratio</h3>
                <div class="value green">{sortino:.2f}</div>
            </div>
            <div class="card">
                <h3>95% Daily VaR</h3>
                <div class="value red">{var_95*100:.2f}%</div>
            </div>
            <div class="card">
                <h3>Max Drawdown</h3>
                <div class="value red">{max_drawdown*100:.2f}%</div>
            </div>
        </div>

        <!-- DETAILED ASSET BREAKDOWN STRUCTURE -->
        <h2>Component Security Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker Token</th>
                    <th>Weight</th>
                    <th>Period Aggregate Return</th>
                    <th>Annualized Return</th>
                    <th>Annualized Volatility</th>
                </tr>
            </thead>
            <tbody>
"""

for i, ticker in enumerate(tickers):
    is_positive = total_p_return[ticker] > 0
    color_style = '#4ade80' if is_positive else '#f87171'
    html_dashboard += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td>{weights[i]*100:.1f}%</td>
                    <td style="color: {color_style}">{total_p_return[ticker]*100:.2f}%</td>
                    <td>{ann_returns[ticker]*100:.2f}%</td>
                    <td>{ann_vols[ticker]*100:.2f}%</td>
                </tr>
    """

html_dashboard += f"""
            </tbody>
        </table>
    </div>
</body>
</html>
"""

with open("institutional_portfolio.html", "w") as f:
    f.write(html_dashboard)

print("\n🚀 Framework successfully constructed! View the dashboard in Windows by running: explorer.exe .")
