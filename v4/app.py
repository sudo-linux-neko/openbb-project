# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vaibhav Agarwal

"""Beast Trading ML — Streamlit dashboard.

Run:
  source ~/ml_env/bin/activate
  streamlit run app.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C

st.set_page_config(page_title="BEAST Trading ML", page_icon="🐂", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {background: #0b0e14;}
h1 {color:#00e676 !important;}
.stMetric {background:#151a24; padding:12px; border-radius:12px; border:1px solid #232b3d;}
</style>
""", unsafe_allow_html=True)

st.title("🐂 BEAST Trading ML — Crypto + Stocks Swing Engine")
st.caption("Ensemble: LightGBM + HistGradientBoosting + LSTM + logistic meta | Data: OpenBB (yfinance free) | Backtest + Alpaca paper")

with st.sidebar:
    st.header("⚙️ Controls")
    symbols = st.multiselect("Universe", C.ALL_SYMBOLS, default=C.ALL_SYMBOLS)
    min_prob = st.slider("Min prob to trade", 0.5, 0.8, C.MIN_PROB, 0.01)
    top_k = st.slider("Top-K positions", 1, 10, C.TOP_K)
    st.divider()
    st.markdown("**Train** (terminal):\n`python train.py --fast` for quick test")
    if st.button("🔄 Reload results"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data
def load_results():
    out = {}
    # honest OOF signals for history/charts; live-model signals for latest row
    for name, fname in (("signals", "signals_oof.parquet"), ("equity", "equity.parquet"), ("trades", "trades.parquet")):
        p = C.RESULTS_DIR / fname
        if p.exists():
            out[name] = pd.read_parquet(p)
        else:
            fb = C.RESULTS_DIR / f"{name}.parquet"
            out[name] = pd.read_parquet(fb) if fb.exists() else None
    lp = C.RESULTS_DIR / "latest_signals.csv"
    out["latest"] = pd.read_csv(lp, index_col=0) if lp.exists() else None
    mp = C.MODELS_DIR / "beast.pkl"
    out["has_model"] = mp.exists()
    return out

res = load_results()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Backtest", "⚡ Live Signals", "🧪 Paper Trading"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    eq = res["equity"]
    if eq is not None and not eq.empty:
        last = float(eq["equity"].iloc[-1])
        total = last / float(eq["equity"].iloc[0]) - 1
        dd = float(eq["equity"].div(eq["equity"].cummax()).sub(1).min())
        c1.metric("Equity", f"${last:,.0f}", f"{total:+.1%}")
        c2.metric("Max Drawdown", f"{dd:.1%}")
        c3.metric("Bars", f"{len(eq)}")
        c4.metric("Model", "✅ trained" if res["has_model"] else "❌ train first")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="Beast equity", line=dict(color="#00e676", width=2)))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10), title="Equity curve")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No backtest yet. Run `python train.py --fast` in the terminal, then click Reload.")
        st.code("source ~/ml_env/bin/activate\npython train.py --fast", language="bash")

with tab2:
    st.subheader("Trades")
    tr = res["trades"]
    if tr is not None and not tr.empty:
        st.dataframe(tr.tail(100), use_container_width=True)
        buys = (tr["side"] == "BUY").sum()
        st.write(f"Total trades: **{len(tr)}** ({buys} buys / {len(tr)-buys} sells)")
    else:
        st.info("No trades file yet — train first.")

with tab3:
    st.subheader("Latest swing signals — P(up >1.5% in 5 days)")
    sig = res["signals"]
    if sig is not None and not sig.empty:
        latest = sig.iloc[-1].sort_values(ascending=False)
        df = pd.DataFrame({"prob": latest.round(4)})
        df["action"] = df["prob"].map(lambda p: "🟢 BUY" if p >= min_prob else ("🟡 WATCH" if p >= 0.5 else "🔴 SKIP"))
        st.dataframe(df.style.background_gradient(subset=["prob"], cmap="Greens"), use_container_width=True)
        fig = go.Figure(go.Bar(x=latest.index, y=latest.values, marker_color=["#00e676" if v >= min_prob else "#546e7a" for v in latest.values]))
        fig.update_layout(template="plotly_dark", title="Signal strength by symbol", height=360)
        st.plotly_chart(fig, use_container_width=True)
        # history of a picked symbol
        pick = st.selectbox("Signal history", list(sig.columns), index=0)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sig.index, y=sig[pick], name=pick, line=dict(color="#29b6f6")))
        fig2.add_hline(y=min_prob, line_dash="dash", line_color="#00e676", annotation_text="trade threshold")
        fig2.update_layout(template="plotly_dark", height=300, title=f"{pick} prob history")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No signals yet — train first.")

with tab4:
    st.subheader("Alpaca paper trading (safe dry-run default)")
    sig = res["signals"]
    dry = st.checkbox("Dry-run (don't send real orders)", value=True)
    notional = st.number_input("Notional per trade ($)", 100, 50000, 2000, 100)
    if st.button("🚀 Send paper orders"):
        if sig is None or sig.empty:
            st.error("No signals — train first.")
        else:
            from src.paper import submit_orders
            latest = sig.iloc[-1].to_dict()
            with st.spinner("Submitting..."):
                out = submit_orders(latest, notional_per_trade=notional, dry_run=dry, min_prob=min_prob)
            st.success(f"Orders: {out}")
            st.caption("Set APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars to enable live paper. Crypto (-USD) auto-skipped (Alpaca stocks only).")
    st.info("Signals + backtest only = zero risk. Paper = Alpaca simulated money. Live broker stays DISABLED unless you uncheck dry-run.")
