"""Integration test for the full backtest pipeline."""

import numpy as np
import pandas as pd

from oil_bot.backtesting.engine import Backtester
from oil_bot.dto import BacktestConfig
from oil_bot.execution.simulated import SimulatedExecutor
from oil_bot.features.engine import FeatureEngine
from oil_bot.risk.fixed_fraction import FixedFractionRisk
from oil_bot.strategies.rsi_strategy import RsiStrategy


def make_enriched(n=300):
    np.random.seed(7)
    prices = 80 + np.cumsum(np.random.uniform(-1.5, 1.5, n))
    prices = np.clip(prices, 20, 200)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    p = pd.Series(prices, dtype=float)
    df = pd.DataFrame(
        {
            "open": p.values,
            "high": (p + np.random.uniform(0, 2, n)),
            "low": (p - np.random.uniform(0, 2, n)),
            "close": p.values,
            "volume": [1e6] * n,
        },
        index=idx,
    )
    return FeatureEngine().transform(df)


def test_full_backtest_runs():
    df = make_enriched()
    backtester = Backtester(
        strategy=RsiStrategy(),
        risk_manager=FixedFractionRisk(),
        executor=SimulatedExecutor(),
        initial_capital=100_000.0,
    )
    result = backtester.run(df, config_snapshot=BacktestConfig())

    assert len(result.equity_curve) == len(df)
    assert "sharpe" in result.metrics
    assert "total_return" in result.metrics
    assert result.run_id is not None


def test_backtest_equity_starts_at_capital():
    df = make_enriched()
    backtester = Backtester(
        strategy=RsiStrategy(),
        risk_manager=FixedFractionRisk(),
        executor=SimulatedExecutor(),
        initial_capital=100_000.0,
    )
    result = backtester.run(df)
    assert result.equity_curve.iloc[0] == 100_000.0


def test_backtest_with_no_trades_keeps_capital():
    """A strategy that never trades should keep equity flat."""
    df = make_enriched()

    class NeverTradeStrategy(RsiStrategy):
        def generate_signal(self, data):
            from oil_bot.dto import Signal

            return Signal(data.index[-1], "HOLD")

    backtester = Backtester(
        strategy=NeverTradeStrategy(),
        risk_manager=FixedFractionRisk(),
        executor=SimulatedExecutor(),
        initial_capital=100_000.0,
    )
    result = backtester.run(df)
    assert result.equity_curve.iloc[-1] == 100_000.0