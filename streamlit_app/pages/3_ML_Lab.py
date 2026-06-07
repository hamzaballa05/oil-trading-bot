"""ML Lab page — train and evaluate the Random Forest model."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oil_bot.backtesting.engine import Backtester
from oil_bot.data.yahoo_loader import YahooFinanceLoader
from oil_bot.execution.simulated import SimulatedExecutor
from oil_bot.features.engine import FeatureEngine
from oil_bot.ml.model_store import ModelStore
from oil_bot.ml.pipeline import MlPipeline
from oil_bot.risk.fixed_fraction import FixedFractionRisk
from oil_bot.strategies.ma_crossover import MaCrossoverStrategy
from oil_bot.strategies.ml_strategy import MlStrategy
from oil_bot.strategies.rsi_strategy import RsiStrategy
from streamlit_app.state import init_state

st.set_page_config(page_title="ML Lab", page_icon="🤖", layout="wide")
init_state()

st.title("🤖 ML Lab — Random Forest")
st.markdown(
    "Entraînez un modèle Random Forest sur les données pétrole "
    "et comparez-le aux stratégies V1."
)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Données")
    symbol = st.selectbox("Actif", ["CL=F", "BZ=F"])
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Début train", value=date(2014, 1, 1))
    with col2:
        end = st.date_input("Fin", value=date(2023, 12, 31))

    st.subheader("Modèle")
    n_estimators = st.slider("Nombre d'arbres", 50, 500, 200, 50)
    max_depth = st.slider("Profondeur max", 3, 20, 6)
    train_ratio = st.slider("% données train", 60, 90, 80) / 100

    st.subheader("Signal")
    buy_threshold = st.slider("Seuil BUY (%)", 0.05, 0.5, 0.1, 0.05) / 100
    sell_threshold = st.slider("Seuil SELL (%)", 0.05, 0.5, 0.1, 0.05) / 100

    st.subheader("Walk-forward")
    n_folds = st.slider("Nombre de folds", 3, 8, 5)

    train_btn = st.button(
        "🚀 Entraîner le modèle",
        type="primary",
        use_container_width=True,
    )

# ENTRAÎNEMENT
if train_btn:
    if start >= end:
        st.error("La date de début doit être avant la date de fin.")
        st.stop()

    with st.spinner("Chargement des données..."):
        df_full = YahooFinanceLoader().load(symbol, start, end)

    with st.spinner("Construction des features ML..."):
        pipeline = MlPipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        X, y, feature_cols = pipeline.prepare_data(df_full)
        X_tr, X_te, y_tr, y_te = pipeline.train_test_split(
            X, y, train_ratio=train_ratio
        )

    with st.spinner(f"Entraînement Random Forest ({n_estimators} arbres)..."):
        result = pipeline.train(X_tr, y_tr, X_te, y_te)

    with st.spinner("Walk-forward validation..."):
        wf = pipeline.walk_forward(X, y, n_folds=n_folds)

    Path("models").mkdir(exist_ok=True)
    store = ModelStore()
    store.save(result.model, result.feature_columns, "rf_model")

    st.session_state["ml_result"] = result
    st.session_state["ml_wf"] = wf
    st.session_state["ml_test_start"] = X_te.index[0].date()
    st.session_state["ml_test_end"] = X_te.index[-1].date()
    st.session_state["ml_symbol"] = symbol
    st.session_state["ml_buy_thr"] = buy_threshold
    st.session_state["ml_sell_thr"] = sell_threshold
    st.session_state["ml_bt_results"] = None
    st.success("✅ Modèle entraîné et sauvegardé dans `models/rf_model.joblib`")

# RÉSULTATS
result = st.session_state.get("ml_result")
wf = st.session_state.get("ml_wf")

if result is None:
    st.info("👈 Configurez et lancez l'entraînement dans la barre latérale.")
    st.stop()

test_start = st.session_state["ml_test_start"]
test_end = st.session_state["ml_test_end"]

tab_metrics, tab_features, tab_wf, tab_backtest = st.tabs([
    "📊 Métriques",
    "🔍 Feature Importance",
    "📈 Walk-Forward",
    "⚙️ Backtest comparatif",
])

# TAB 1 — Métriques
with tab_metrics:
    st.subheader("Performance du modèle")
    st.caption(f"Test set : {test_start} → {test_end}")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Direction Accuracy",
        f"{result.test_metrics['direction_accuracy']:.1%}",
        help="% de fois où le signe du rendement prédit est correct",
    )
    col2.metric(
        "MAE",
        f"{result.test_metrics['mae']:.4f}",
        help="Erreur absolue moyenne",
    )
    col3.metric(
        "R²",
        f"{result.test_metrics['r2']:.4f}",
        help="Variance expliquée",
    )

    st.divider()
    col4, col5, col6 = st.columns(3)
    col4.metric("Train MAE", f"{result.train_metrics['mae']:.4f}")
    col5.metric(
        "Train Direction Acc.",
        f"{result.train_metrics['direction_accuracy']:.1%}",
    )
    col6.metric("Train R²", f"{result.train_metrics['r2']:.4f}")

    st.subheader("Distribution des prédictions vs réalité")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result.predictions["actual"],
        name="Réel", opacity=0.7, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=result.predictions["predicted"],
        name="Prédit", opacity=0.7, nbinsx=50,
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Rendement journalier",
        yaxis_title="Fréquence",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2 — Feature Importance
with tab_features:
    st.subheader("Top 15 features les plus importantes")
    top15 = result.feature_importance.head(15)
    fig = go.Figure(go.Bar(
        x=top15["importance"],
        y=top15["feature"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        height=500,
        xaxis_title="Importance",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 3 — Walk-Forward
with tab_wf:
    st.subheader("Résultats Walk-Forward")
    st.caption(
        "Le walk-forward simule le ré-entraînement périodique. "
        "C'est la méthode d'évaluation la plus réaliste."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Direction Acc. globale",
        f"{wf.overall_metrics['direction_accuracy']:.1%}",
    )
    col2.metric("MAE globale", f"{wf.overall_metrics['mae']:.4f}")
    col3.metric("R² global", f"{wf.overall_metrics['r2']:.4f}")

    st.subheader("Détail par fold")
    folds_df = pd.DataFrame(wf.fold_results)
    display = folds_df[
        ["fold", "test_start", "test_end",
         "direction_accuracy", "mae", "r2"]
    ].copy()
    display.columns = [
        "Fold", "Début test", "Fin test",
        "Dir. Accuracy", "MAE", "R²"
    ]
    display["Dir. Accuracy"] = display["Dir. Accuracy"].apply(
        lambda x: f"{x:.1%}"
    )
    display["MAE"] = display["MAE"].apply(lambda x: f"{x:.4f}")
    display["R²"] = display["R²"].apply(lambda x: f"{x:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {f['fold']}" for f in wf.fold_results],
        y=[f["direction_accuracy"] for f in wf.fold_results],
        marker_color=[
            "#2ca02c" if f["direction_accuracy"] > 0.5 else "#d62728"
            for f in wf.fold_results
        ],
    ))
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="gray",
        annotation_text="50% (aléatoire)"
    )
    fig.update_layout(
        height=300,
        yaxis_title="Direction Accuracy",
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 4 — Backtest comparatif
with tab_backtest:
    st.subheader("Backtest out-of-sample")
    st.caption(
        f"Période test : {test_start} → {test_end}. "
        "Le modèle n'a JAMAIS vu ces données."
    )

    buy_thr = st.session_state.get("ml_buy_thr", 0.001)
    sell_thr = st.session_state.get("ml_sell_thr", 0.001)

    st.info(
        f"Seuils utilisés : BUY > {buy_thr:.3%} | "
        f"SELL < -{sell_thr:.3%}"
    )

    run_bt = st.button("▶ Lancer le backtest comparatif", type="primary")

    if run_bt:
        symbol = st.session_state["ml_symbol"]

        with st.spinner("Backtest en cours..."):
            df_test = YahooFinanceLoader().load(
                symbol, test_start, test_end
            )
            enriched_test = FeatureEngine().transform(df_test)

            strategies = {
                "ML RandomForest": MlStrategy(
                    model=result.model,
                    feature_columns=result.feature_columns,
                    buy_threshold=buy_thr,
                    sell_threshold=-sell_thr,
                ),
                "RSI(14, 30/70)": RsiStrategy(),
                "MA Crossover(20/50)": MaCrossoverStrategy(),
            }

            bt_results = {}
            for name, strat in strategies.items():
                data = df_test if "ML" in name else enriched_test
                bt = Backtester(
                    strat,
                    FixedFractionRisk(),
                    SimulatedExecutor(),
                    100_000.0,
                )
                bt_results[name] = bt.run(data)

        st.session_state["ml_bt_results"] = bt_results

    bt_results = st.session_state.get("ml_bt_results")
    if not bt_results:
        st.info("Cliquez sur 'Lancer le backtest comparatif'.")
        st.stop()

    # Equity curves
    st.subheader("Courbes d'equity comparées")
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (name, r) in enumerate(bt_results.items()):
        eq = r.equity_curve / r.equity_curve.iloc[0]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq,
            name=name,
            line=dict(color=colors[i]),
        ))
    fig.update_layout(
        height=400,
        hovermode="x unified",
        yaxis_title="Rendement normalisé",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau métriques
    st.subheader("Tableau de métriques")
    rows = []
    for name, r in bt_results.items():
        m = r.metrics
        rows.append({
            "Stratégie": name,
            "Rendement": f"{m.get('total_return', 0):+.1%}",
            "Sharpe": f"{m.get('sharpe', 0):+.2f}",
            "Max DD": f"{m.get('max_drawdown', 0):.1%}",
            "CAGR": f"{m.get('cagr', 0):+.1%}",
            "Trades": int(m.get("n_trades", 0)),
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Note** : Résultats honnêtement out-of-sample. "
        "Un Random Forest simple sur données journalières "
        "ne bat pas systématiquement le marché — "
        "conforme à la littérature académique."
    )