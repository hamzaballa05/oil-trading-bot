"""Train an ML model and backtest it honestly on out-of-sample data."""

import sys
from datetime import date

sys.path.insert(0, "src")

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


def main():
    symbol = "CL=F"
    start_full = date(2014, 1, 1)
    end_full = date(2023, 12, 31)

    print("\n" + "=" * 60)
    print("  OIL TRADING BOT — ML V2 Training Pipeline")
    print("=" * 60)

    # 1. Charger les données complètes
    print("\n[1/5] Chargement des donnees...")
    df_full = YahooFinanceLoader().load(symbol, start_full, end_full)
    print(f"      {len(df_full)} bars charges ({start_full} -> {end_full})")

    # 2. Préparer les features
    print("\n[2/5] Construction des features ML...")
    pipeline = MlPipeline(n_estimators=200, max_depth=10, random_state=42)
    X, y, feature_cols = pipeline.prepare_data(df_full)
    print(f"      {len(X)} lignes, {len(feature_cols)} features")

    # 3. Split temporel 80/20
    X_tr, X_te, y_tr, y_te = pipeline.train_test_split(X, y, train_ratio=0.8)
    test_start = X_te.index[0].date()
    test_end = X_te.index[-1].date()

    # 4. Entraîner le modèle
    print("\n[3/5] Entrainement Random Forest...")
    result = pipeline.train(X_tr, y_tr, X_te, y_te)

    print(f"\n      --- Metriques sur TEST SET ({test_start} -> {test_end}) ---")
    print(f"      MAE               : {result.test_metrics['mae']:.6f}")
    print(f"      R2                : {result.test_metrics['r2']:.4f}")
    print(f"      Direction Accuracy : {result.test_metrics['direction_accuracy']:.1%}")

    print(f"\n      --- Top 10 Features ---")
    for _, row in result.feature_importance.head(10).iterrows():
        print(f"      {row['feature']:30s}: {row['importance']:.4f}")

    # 5. Walk-forward validation
    print("\n[4/5] Walk-forward validation (5 folds)...")
    wf = pipeline.walk_forward(X, y, n_folds=5)

    print(f"\n      Direction Accuracy globale : "
          f"{wf.overall_metrics['direction_accuracy']:.1%}")
    print(f"      MAE globale               : "
          f"{wf.overall_metrics['mae']:.6f}")
    print(f"      R2 global                 : "
          f"{wf.overall_metrics['r2']:.4f}")

    print(f"\n      Detail par fold :")
    for fold in wf.fold_results:
        print(
            f"      Fold {fold['fold']} "
            f"[{fold['test_start']} -> {fold['test_end']}] "
            f"Dir.Acc={fold['direction_accuracy']:.1%} "
            f"MAE={fold['mae']:.6f}"
        )

    # 6. Sauvegarder
    store = ModelStore()
    store.save(result.model, result.feature_columns, "rf_model")
    print(f"\n      Modele sauvegarde : models/rf_model.joblib")

    # 7. Backtest OUT-OF-SAMPLE uniquement
    # Le modèle a été entraîné sur [start_full -> test_start]
    # On backtest UNIQUEMENT sur [test_start -> test_end]
    # C'est la seule façon honnête de mesurer la performance
    print(f"\n[5/5] Backtest comparatif sur TEST SET uniquement")
    print(f"      Periode : {test_start} -> {test_end}")
    print(f"      (Le modele ML n'a JAMAIS vu cette periode)\n")

    df_test = YahooFinanceLoader().load(symbol, test_start, test_end)
    enriched_test = FeatureEngine().transform(df_test)

    strategies = {
        "ML RandomForest": MlStrategy(
            model=result.model,
            feature_columns=result.feature_columns,
            buy_threshold=0.001,
            sell_threshold=-0.001,
        ),
        "RSI(14, 30/70)": RsiStrategy(),
        "MA Crossover(20/50)": MaCrossoverStrategy(),
    }

    print(f"      {'Strategie':<22s} | {'Return':>8s} | "
          f"{'Sharpe':>7s} | {'MDD':>8s} | {'Trades':>6s}")
    print("      " + "-" * 58)

    for name, strat in strategies.items():
        data = df_test if "ML" in name else enriched_test
        bt = Backtester(
            strat,
            FixedFractionRisk(),
            SimulatedExecutor(),
            100_000.0,
        )
        r = bt.run(data)
        m = r.metrics
        print(
            f"      {name:<22s} | {m['total_return']:>+7.1%} | "
            f"{m['sharpe']:>+6.2f} | "
            f"{m['max_drawdown']:>7.1%} | "
            f"{int(m['n_trades']):>6d}"
        )

    print("\n" + "=" * 60)
    print("  IMPORTANT pour le rapport :")
    print("  Ces resultats sont OUT-OF-SAMPLE (periode jamais vue")
    print("  par le modele pendant l'entrainement).")
    print("  C'est la seule mesure valide de performance ML.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()