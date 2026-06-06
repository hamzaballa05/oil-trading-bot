"""Strategy Comparison page."""

import sys
import os
from datetime import date

# Ajout de la racine du projet ET du dossier src au chemin Python
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oil_bot.dto import BacktestConfig, RiskConfig, StrategyConfig
from oil_bot.services.comparison_service import ComparisonService
from streamlit_app.state import init_state

st.set_page_config(page_title="Comparison", page_icon="bars", layout="wide")
init_state()

st.title("Strategy Comparison")

st.subheader("Configuration commune")
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Actif", ["CL=F", "BZ=F"])
    start = st.date_input("Debut", value=date(2018, 1, 1))
with col2:
    end = st.date_input("Fin", value=date(2023, 12, 31))
    capital = st.number_input("Capital", value=100_000, step=10_000)
with col3:
    fees = st.slider("Commission (bps)", 1, 20, 5)
    slippage = st.slider("Slippage (bps)", 1, 10, 3)

st.divider()
st.subheader("Strategies a comparer")

OPTIONS = {
    "RSI (30/70)": ("rsi", {"period": 14, "oversold": 30, "overbought": 70}),
    "RSI (25/75)": ("rsi", {"period": 14, "oversold": 25, "overbought": 75}),
    "EMA Crossover (20/50)": (
        "ma_crossover",
        {"fast_period": 20, "slow_period": 50},
    ),
    "EMA Crossover (10/30)": (
        "ma_crossover",
        {"fast_period": 10, "slow_period": 30},
    ),
    "Combined RSI + MA": ("combined", {}),
}

selected = st.multiselect(
    "Choisissez 2 a 5 strategies",
    options=list(OPTIONS.keys()),
    default=["RSI (30/70)", "EMA Crossover (20/50)", "Combined RSI + MA"],
)

if st.button(
    "Lancer la comparaison", type="primary", disabled=len(selected) < 2
):
    labeled_configs = {}
    for label in selected:
        name, params = OPTIONS[label]
        labeled_configs[label] = BacktestConfig(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=float(capital),
            strategy=StrategyConfig(name, tuple(params.items())),
            risk=RiskConfig(),
            fees=fees / 10000,
            slippage=slippage / 10000,
        )
    with st.spinner(f"Backtest de {len(selected)} strategies..."):
        try:
            results = ComparisonService().run_batch(labeled_configs)
            st.session_state["comparison_results"] = results
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            st.stop()

results = st.session_state.get("comparison_results")
if not results:
    st.info("Selectionnez des strategies et lancez la comparaison.")
    st.stop()

# Equity curves overlay
st.subheader("Courbes d'equity comparees")
fig = go.Figure()
for label, result in results.items():
    eq = result.equity_curve / result.equity_curve.iloc[0]
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name=label))
fig.update_layout(
    height=400, hovermode="x unified", yaxis_title="Rendement normalise"
)
st.plotly_chart(fig, use_container_width=True)

# Metrics table
st.subheader("Tableau de metriques")
rows = []
for label, result in results.items():
    m = result.metrics
    rows.append(
        {
            "Strategie": label,
            "Rendement": f"{m.get('total_return', 0):.1%}",
            "CAGR": f"{m.get('cagr', 0):.1%}",
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "Sortino": f"{m.get('sortino', 0):.2f}",
            "Max DD": f"{m.get('max_drawdown', 0):.1%}",
            "Trades": int(m.get("n_trades", 0)),
        }
    )
st.dataframe(
    pd.DataFrame(rows), use_container_width=True, hide_index=True
)