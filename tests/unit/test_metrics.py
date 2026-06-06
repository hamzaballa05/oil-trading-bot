"""Tests for the metrics calculator."""

import pandas as pd

from oil_bot.metrics.calculator import MetricsCalculator


def test_total_return_positive():
    eq = pd.Series([100, 110, 120], dtype=float)
    m = MetricsCalculator().compute(eq)
    assert m["total_return"] == 0.2


def test_flat_equity_zero_return():
    eq = pd.Series([100, 100, 100], dtype=float)
    m = MetricsCalculator().compute(eq)
    assert m["total_return"] == 0.0


def test_max_drawdown_negative():
    eq = pd.Series([100, 120, 80, 90], dtype=float)
    m = MetricsCalculator().compute(eq)
    assert m["max_drawdown"] < 0