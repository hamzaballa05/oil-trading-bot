"""Home page of the Oil Trading Bot app."""

import streamlit as st

st.set_page_config(
    page_title="Oil Trading Bot",
    page_icon="oil",
    layout="wide",
)

st.title("Oil Trading Bot")
st.markdown("Systeme de backtesting algorithmique sur le petrole WTI")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("A propos")
    st.markdown(
        """
        Ce projet simule des strategies de trading sur le marche
        du petrole (WTI Crude Oil, ticker CL=F).

        **Strategies disponibles (V1)**
        - RSI Mean Reversion
        - EMA Crossover
        - Combined RSI + Trend
        """
    )

with col2:
    st.subheader("Navigation")
    st.page_link("pages/1_Backtest.py", label="Lancer un backtest")
    st.page_link("pages/2_Comparison.py", label="Comparer des strategies")

st.divider()
st.caption("Projet academique de trading algorithmique")