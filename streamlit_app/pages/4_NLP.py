"""NLP Sentiment Analysis page."""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import streamlit as st
import plotly.graph_objects as go

from oil_bot.services.nlp_service import NlpService
from streamlit_app.state import init_state

st.set_page_config(
    page_title="NLP Sentiment", page_icon="📰", layout="wide"
)
init_state()

st.title("📰 NLP — Analyse de Sentiment")
st.markdown(
    "Analyse le sentiment des news pétrole en temps réel "
    "via RSS feeds (Yahoo Finance, Reuters)."
)

# Bouton de chargement
if st.button("🔄 Charger les news et analyser", type="primary"):
    with st.spinner("Récupération des news RSS..."):
        svc = NlpService()
        result = svc.fetch_and_analyze()
        st.session_state["nlp_result"] = result

result = st.session_state.get("nlp_result")

if result is None:
    st.info("👆 Cliquez sur 'Charger les news' pour analyser le sentiment.")
    st.stop()

# Score global
st.subheader("Sentiment actuel")
score = result["current_score"]
signal = result["signal"]
n = result["n_articles"]

col1, col2, col3 = st.columns(3)
col1.metric(
    "Score de sentiment",
    f"{score:+.3f}",
    help="Score moyen des articles [-1 très négatif, +1 très positif]",
)
col2.metric("Signal généré", signal)
col3.metric("Articles analysés", n)

# Jauge de sentiment
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={"text": "Sentiment pétrole"},
    gauge={
        "axis": {"range": [-1, 1]},
        "bar": {"color": "darkblue"},
        "steps": [
            {"range": [-1, -0.1], "color": "#d62728"},
            {"range": [-0.1, 0.1], "color": "#aec7e8"},
            {"range": [0.1, 1], "color": "#2ca02c"},
        ],
        "threshold": {
            "line": {"color": "white", "width": 4},
            "thickness": 0.75,
            "value": score,
        },
    },
    number={"suffix": "", "valueformat": ".3f"},
))
fig_gauge.update_layout(height=300)
st.plotly_chart(fig_gauge, use_container_width=True)

# Signal visuel
if signal == "BUY":
    st.success(f"📈 Signal : **BUY** — Sentiment positif ({score:+.3f})")
elif signal == "SELL":
    st.error(f"📉 Signal : **SELL** — Sentiment négatif ({score:+.3f})")
else:
    st.warning(f"⏸️ Signal : **HOLD** — Sentiment neutre ({score:+.3f})")

# Tableau des articles
if result["news_df"] is not None and not result["news_df"].empty:
    st.subheader("Articles analysés")
    news_df = result["news_df"].copy()

    # Colonne couleur selon sentiment
    news_df["Sentiment"] = news_df["polarity"].apply(
        lambda x: "🟢 Positif" if x > 0.1
        else ("🔴 Négatif" if x < -0.1 else "⚪ Neutre")
    )

    display = news_df[[
        "published", "source", "title", "polarity", "Sentiment"
    ]].copy()
    display.columns = [
        "Date", "Source", "Titre", "Score", "Sentiment"
    ]
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d %H:%M")
    display["Score"] = display["Score"].round(3)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Distribution des scores
    st.subheader("Distribution des scores de sentiment")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=news_df["polarity"],
        nbinsx=20,
        marker_color="#1f77b4",
        name="Articles",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="gray",
        annotation_text="Neutre",
    )
    fig.add_vline(
        x=score, line_dash="solid", line_color="#ff7f0e",
        annotation_text=f"Moyenne ({score:+.3f})",
    )
    fig.update_layout(
        height=300,
        xaxis_title="Score de sentiment",
        yaxis_title="Nombre d'articles",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Interprétation** : TextBlob analyse la polarité "
        "du texte de chaque titre/résumé. Score > +0.1 → "
        "sentiment positif → signal BUY potentiel. "
        "Score < -0.1 → sentiment négatif → signal SELL potentiel."
    )