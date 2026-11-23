# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

from openbb import obb

# Fetch the data
data = obb.equity.price.historical(symbol="AAPL", provider="yfinance", limit=100)

# Open an interactive chart in your browser
data.charting.show()
