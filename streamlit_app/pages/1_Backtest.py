"""Backtest Runner page."""

import sys
import os
from datetime import date

# Ajout de la racine du projet ET du dossier src au chemin Python
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from oil_bot.dto import BacktestConfig, RiskConfig, StrategyConfig
from oil_bot.services.backtest_service import BacktestService
from streamlit_app.state import init_state

st.set_page_config(page_title="Backtest", page_icon="chart", layout="wide")
init_state()

st.title("Backtest Runner")

# ----- SIDEBAR : configuration -----
with st.sidebar:
    st.header("Configuration")

    with st.form("backtest_form"):
        st.subheader("Donnees")
        symbol = st.selectbox("Actif", ["CL=F", "BZ=F"])
        col_a, col_b = st.columns(2)
        with col_a:
            start = st.date_input("Debut", value=date(2018, 1, 1))
        with col_b:
            end = st.date_input("Fin", value=date(2023, 12, 31))
        capital = st.number_input(
            "Capital initial", value=100_000, step=10_000
        )

        st.subheader("Strategie")
        strategy_name = st.selectbox(
            "Type",
            ["rsi", "ma_crossover", "combined"],
            format_func=lambda x: {
                "rsi": "RSI Mean Reversion",
                "ma_crossover": "EMA Crossover",
                "combined": "Combined RSI + MA",
            }[x],
        )

        params: dict = {}
        if strategy_name == "rsi":
            params["period"] = st.slider("Periode RSI", 7, 30, 14)
            params["oversold"] = st.slider("Survente", 20, 40, 30)
            params["overbought"] = st.slider("Surachat", 60, 80, 70)
        elif strategy_name == "ma_crossover":
            params["fast_period"] = st.slider("EMA rapide", 5, 30, 20)
            params["slow_period"] = st.slider("EMA lente", 20, 100, 50)

        st.subheader("Risque")
        risk_per_trade = st.slider("Risque/trade (%)", 0.5, 5.0, 2.0, 0.5)
        stop_loss = st.slider("Stop loss (%)", 2.0, 10.0, 5.0, 0.5)

        st.subheader("Frais")
        fees = st.slider("Commission (bps)", 1, 20, 5)
        slippage = st.slider("Slippage (bps)", 1, 10, 3)

        submitted = st.form_submit_button(
            "Lancer le backtest", type="primary", use_container_width=True
        )

# ----- LANCEMENT -----
if submitted:
    config = BacktestConfig(
        symbol=symbol,
        start=start,
        end=end,
        initial_capital=float(capital),
        strategy=StrategyConfig(strategy_name, tuple(params.items())),
        risk=RiskConfig(
            risk_per_trade=risk_per_trade / 100,
            stop_loss_pct=stop_loss / 100,
        ),
        fees=fees / 10000,
        slippage=slippage / 10000,
    )
    with st.spinner("Backtest en cours..."):
        try:
            result = BacktestService().run(config)
            st.session_state["last_result"] = result
            st.success(f"Termine - Run {result.run_id[:8]}")
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            st.stop()

# ----- RESULTATS -----
result = st.session_state.get("last_result")
if result is None:
    st.info("Configurez et lancez un backtest dans la barre laterale.")
    st.stop()

m = result.metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rendement total", f"{m.get('total_return', 0):.1%}")
c2.metric("Sharpe", f"{m.get('sharpe', 0):.2f}")
c3.metric("Max Drawdown", f"{m.get('max_drawdown', 0):.1%}")
c4.metric("CAGR", f"{m.get('cagr', 0):.1%}")
c5.metric("Trades", f"{int(m.get('n_trades', 0))}")

tab_eq, tab_tr, tab_sig = st.tabs(["Equity", "Trades", "Signaux"])

with tab_eq:
    eq = result.equity_curve
    dd = (eq - eq.cummax()) / eq.cummax()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=["Courbe d'equity", "Drawdown"],
    )
    fig.add_trace(
        go.Scatter(x=eq.index, y=eq, name="Equity"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy"),
        row=2,
        col=1,
    )
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab_tr:
    if result.trades.empty:
        st.info("Aucun trade sur cette periode.")
    else:
        st.dataframe(result.trades, use_container_width=True)
        st.download_button(
            "Telecharger CSV",
            result.trades.to_csv(index=False),
            "trades.csv",
        )

with tab_sig:
    if result.signals.empty:
        st.info("Aucun signal enregistre.")
    else:
        actionable = result.signals[result.signals["action"] != "HOLD"]
        st.metric("Signaux BUY/SELL", len(actionable))
        st.dataframe(actionable.head(50), use_container_width=True)