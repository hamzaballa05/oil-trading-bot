"""ML Service — facade for ML training and backtesting.

Called by Streamlit and CLI. Orchestrates:
1. Load data
2. Build features + target
3. Train/evaluate model
4. Optionally backtest with MlStrategy
"""

from datetime import date

import pandas as pd

from oil_bot.backtesting.engine import Backtester, BacktestResult
from oil_bot.data.yahoo_loader import YahooFinanceLoader
from oil_bot.dto import BacktestConfig, RiskConfig, StrategyConfig
from oil_bot.execution.simulated import SimulatedExecutor
from oil_bot.ml.feature_engineering import MlFeatureEngine
from oil_bot.ml.model_store import ModelStore
from oil_bot.ml.pipeline import MlPipeline, MlTrainResult, WalkForwardResult
from oil_bot.risk.fixed_fraction import FixedFractionRisk
from oil_bot.strategies.ml_strategy import MlStrategy
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class MlService:
    """Facade for all ML operations.

    Usage:
        svc = MlService()

        # Train and evaluate
        result = svc.train_and_evaluate("CL=F", date(2014,1,1), date(2023,12,31))

        # Walk-forward validation
        wf = svc.walk_forward("CL=F", date(2014,1,1), date(2023,12,31))

        # Backtest with trained model
        bt = svc.backtest_with_model(result.model, result.feature_columns, ...)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 10,
        random_state: int = 42,
        horizon: int = 1,
    ) -> None:
        self.pipeline = MlPipeline(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            horizon=horizon,
        )
        self.model_store = ModelStore()

    def train_and_evaluate(
        self,
        symbol: str = "CL=F",
        start: date = date(2014, 1, 1),
        end: date = date(2023, 12, 31),
        train_ratio: float = 0.8,
    ) -> MlTrainResult:
        """Load data, build features, train model, evaluate.

        Args:
            symbol: Ticker symbol.
            start, end: Date range.
            train_ratio: Fraction of data for training.

        Returns:
            MlTrainResult with model, metrics, importance, predictions.
        """
        logger.info(f"ML train: {symbol} {start} → {end}")

        # Load data
        df = YahooFinanceLoader().load(symbol, start, end)

        # Prepare features and target
        X, y, feature_cols = self.pipeline.prepare_data(df)

        # Split temporally
        X_train, X_test, y_train, y_test = self.pipeline.train_test_split(
            X, y, train_ratio
        )

        # Train and evaluate
        result = self.pipeline.train(X_train, y_train, X_test, y_test)

        # Save model
        self.model_store.save(result.model, result.feature_columns, "rf_model")

        return result

    def walk_forward(
        self,
        symbol: str = "CL=F",
        start: date = date(2014, 1, 1),
        end: date = date(2023, 12, 31),
        n_folds: int = 5,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            symbol: Ticker symbol.
            start, end: Date range.
            n_folds: Number of walk-forward folds.

        Returns:
            WalkForwardResult with per-fold and overall metrics.
        """
        df = YahooFinanceLoader().load(symbol, start, end)
        X, y, _ = self.pipeline.prepare_data(df)
        return self.pipeline.walk_forward(X, y, n_folds=n_folds)

    def backtest_with_model(
        self,
        model,
        feature_columns: list[str],
        symbol: str = "CL=F",
        start: date = date(2018, 1, 1),
        end: date = date(2023, 12, 31),
        initial_capital: float = 100_000.0,
        buy_threshold: float = 0.002,
        sell_threshold: float = -0.002,
        fees: float = 0.0005,
        slippage: float = 0.0003,
    ) -> BacktestResult:
        """Backtest a trained ML model as a strategy.

        The MlStrategy plugs into the SAME Backtester as V1 strategies.
        This is the test of good architecture: zero changes to the Backtester.

        Args:
            model: Trained sklearn model.
            feature_columns: Feature names the model expects.
            buy_threshold: Predicted return threshold for BUY.
            sell_threshold: Predicted return threshold for SELL.

        Returns:
            BacktestResult (same as V1 strategies).
        """
        df = YahooFinanceLoader().load(symbol, start, end)

        # DO NOT call FeatureEngine here — MlStrategy builds its own features
        # because it needs the ML-specific ones (lags, rolling stats)

        strategy = MlStrategy(
            model=model,
            feature_columns=feature_columns,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        risk = FixedFractionRisk()
        executor = SimulatedExecutor(fees=fees, slippage=slippage)

        backtester = Backtester(
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            initial_capital=initial_capital,
        )

        config = BacktestConfig(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=initial_capital,
            strategy=StrategyConfig("ml_random_forest", ()),
            fees=fees,
            slippage=slippage,
        )

        return backtester.run(df, config_snapshot=config)

    def load_and_backtest(
        self,
        model_name: str = "rf_model",
        **backtest_kwargs,
    ) -> BacktestResult:
        """Load a saved model and backtest it.

        Convenience method: loads from disk and runs backtest.
        """
        model, feature_cols = self.model_store.load(model_name)
        return self.backtest_with_model(model, feature_cols, **backtest_kwargs)