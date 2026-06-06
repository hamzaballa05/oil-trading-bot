"""Tests for trading strategies."""

import numpy as np
import pandas as pd

from oil_bot.features.engine import FeatureEngine
from oil_bot.strategies.combined import CombinedStrategy
from oil_bot.strategies.ma_crossover import MaCrossoverStrategy
from oil_bot.strategies.rsi_strategy import RsiStrategy


def make_enriched(n=120):
    np.random.seed(1)
    prices = 80 + np.cumsum(np.random.uniform(-1, 1, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    p = pd.Series(prices, dtype=float)
    df = pd.DataFrame(
        {
            "open": p.values,
            "high": (p + 1).values,
            "low": (p - 1).values,
            "close": p.values,
            "volume": [1e6] * n,
        },
        index=idx,
    )
    return FeatureEngine().transform(df)


def test_rsi_strategy_returns_valid_action():
    df = make_enriched()
    sig = RsiStrategy().generate_signal(df)
    assert sig.action in ("BUY", "SELL", "HOLD")


def test_ma_crossover_returns_valid_action():
    df = make_enriched()
    sig = MaCrossoverStrategy().generate_signal(df)
    assert sig.action in ("BUY", "SELL", "HOLD")


def test_combined_returns_valid_action():
    df = make_enriched()
    sig = CombinedStrategy().generate_signal(df)
    assert sig.action in ("BUY", "SELL", "HOLD")


def test_rsi_strategy_buys_when_oversold():
    df = make_enriched()
    df = df.copy()
    df.loc[df.index[-1], "rsi_14"] = 20.0
    sig = RsiStrategy(oversold=30, overbought=70).generate_signal(df)
    assert sig.action == "BUY"


def test_rsi_strategy_sells_when_overbought():
    df = make_enriched()
    df = df.copy()
    df.loc[df.index[-1], "rsi_14"] = 80.0
    sig = RsiStrategy(oversold=30, overbought=70).generate_signal(df)
    assert sig.action == "SELL"
