"""Tests for the ML pipeline."""

import numpy as np
import pandas as pd
import pytest

from oil_bot.features.engine import FeatureEngine
from oil_bot.ml.feature_engineering import MlFeatureEngine
from oil_bot.ml.pipeline import MlPipeline
from oil_bot.ml.model_store import ModelStore
from oil_bot.strategies.ml_strategy import MlStrategy


def make_ohlcv(n=400):
    """Create realistic OHLCV data for testing."""
    np.random.seed(42)
    prices = 80 + np.cumsum(np.random.normal(0, 1.2, n))
    prices = np.clip(prices, 30, 150)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    p = pd.Series(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": p.values,
            "high": p.values + np.random.uniform(0, 2, n),
            "low": p.values - np.random.uniform(0, 2, n),
            "close": p.values,
            "volume": np.random.uniform(5e5, 2e6, n),
        },
        index=idx,
    )


class TestMlFeatureEngine:
    def test_build_features_adds_columns(self):
        df = make_ohlcv()
        engine = MlFeatureEngine()
        features = engine.build_features(df)
        assert len(features.columns) > len(df.columns) + 20

    def test_build_target_is_next_day_return(self):
        df = make_ohlcv()
        engine = MlFeatureEngine(horizon=1)
        target = engine.build_target(df)
        # target_t = (close_{t+1} - close_t) / close_t
        expected = (df["close"].iloc[1] - df["close"].iloc[0]) / df["close"].iloc[0]
        assert target.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_target_last_value_is_nan(self):
        df = make_ohlcv()
        engine = MlFeatureEngine(horizon=1)
        target = engine.build_target(df)
        assert pd.isna(target.iloc[-1])

    def test_feature_columns_exclude_ohlcv(self):
        df = make_ohlcv()
        engine = MlFeatureEngine()
        features = engine.build_features(df)
        cols = engine.get_feature_columns(features)
        assert "close" not in cols
        assert "open" not in cols


class TestMlPipeline:
    def test_prepare_data_no_nan(self):
        df = make_ohlcv()
        pipeline = MlPipeline()
        X, y, cols = pipeline.prepare_data(df)
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    def test_temporal_split_order(self):
        df = make_ohlcv()
        pipeline = MlPipeline()
        X, y, _ = pipeline.prepare_data(df)
        X_tr, X_te, y_tr, y_te = pipeline.train_test_split(X, y, 0.8)
        # Train dates must be BEFORE test dates
        assert X_tr.index[-1] < X_te.index[0]

    def test_train_produces_model(self):
        df = make_ohlcv()
        pipeline = MlPipeline(n_estimators=10, max_depth=5)
        X, y, _ = pipeline.prepare_data(df)
        X_tr, X_te, y_tr, y_te = pipeline.train_test_split(X, y, 0.8)
        result = pipeline.train(X_tr, y_tr, X_te, y_te)
        assert result.model is not None
        assert "mae" in result.test_metrics
        assert "direction_accuracy" in result.test_metrics
        assert len(result.feature_importance) > 0

    def test_walk_forward_produces_folds(self):
        df = make_ohlcv()
        pipeline = MlPipeline(n_estimators=10, max_depth=5)
        X, y, _ = pipeline.prepare_data(df)
        wf = pipeline.walk_forward(X, y, n_folds=3)
        assert len(wf.fold_results) == 3
        assert "mae" in wf.overall_metrics


class TestMlStrategy:
    def test_ml_strategy_returns_valid_signal(self):
        df = make_ohlcv()
        pipeline = MlPipeline(n_estimators=10, max_depth=5)
        X, y, cols = pipeline.prepare_data(df)
        X_tr, X_te, y_tr, y_te = pipeline.train_test_split(X, y, 0.8)
        result = pipeline.train(X_tr, y_tr, X_te, y_te)

        strategy = MlStrategy(
            model=result.model,
            feature_columns=result.feature_columns,
        )
        signal = strategy.generate_signal(df)
        assert signal.action in ("BUY", "SELL", "HOLD")
        assert "predicted_return" in signal.metadata

    def test_ml_strategy_name(self):
        strategy = MlStrategy(model=None, feature_columns=[])
        assert "ML_RF" in strategy.name


class TestModelStore:
    def test_save_and_load(self, tmp_path):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0.1, 0.2, 0.3])
        model.fit(X, y)

        store = ModelStore(store_dir=tmp_path)
        store.save(model, ["a", "b"], "test_model")

        loaded_model, loaded_cols = store.load("test_model")
        assert loaded_cols == ["a", "b"]
        assert loaded_model.predict(X[:1])[0] == pytest.approx(
            model.predict(X[:1])[0]
        )