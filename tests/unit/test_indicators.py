"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.atr import ATR
from oil_bot.indicators.bollinger import BollingerBands
from oil_bot.indicators.macd import MACD
from oil_bot.indicators.moving_average import EMA, SMA
from oil_bot.indicators.rsi import RSI


def make_df(prices):
    idx = pd.date_range("2023-01-01", periods=len(prices), freq="D", tz="UTC")
    p = pd.Series(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": p.values,
            "high": (p + 1).values,
            "low": (p - 1).values,
            "close": p.values,
            "volume": [1e6] * len(prices),
        },
        index=idx,
    )


def test_sma_last_value():
    df = make_df(list(range(1, 11)))
    sma = SMA(period=3).compute(df)
    assert sma.iloc[-1] == pytest.approx(9.0)


def test_sma_insufficient_data():
    with pytest.raises(InsufficientDataError):
        SMA(period=5).compute(make_df([1, 2]))


def test_ema_reacts_to_spike():
    df = make_df([10.0] * 20 + [100.0])
    ema = EMA(period=10).compute(df).iloc[-1]
    sma = SMA(period=10).compute(df).iloc[-1]
    assert ema > sma


def test_rsi_bounds():
    np.random.seed(42)
    df = make_df(np.random.uniform(50, 100, 100).tolist())
    rsi = RSI(14).compute(df).dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_rsi_high_on_rising_prices():
    df = make_df(list(range(1, 51)))
    rsi = RSI(14).compute(df).dropna()
    assert rsi.iloc[-1] > 90


def test_rsi_low_on_falling_prices():
    df = make_df(list(range(50, 0, -1)))
    rsi = RSI(14).compute(df).dropna()
    assert rsi.iloc[-1] < 10


def test_macd_columns():
    df = make_df(list(range(1, 60)))
    macd = MACD().compute(df)
    assert list(macd.columns) == ["macd_line", "macd_signal", "macd_hist"]


def test_bollinger_columns():
    df = make_df(list(range(1, 40)))
    bb = BollingerBands().compute(df)
    assert "bb_upper" in bb.columns
    assert "bb_lower" in bb.columns


def test_atr_positive():
    df = make_df(list(range(1, 40)))
    atr = ATR(14).compute(df).dropna()
    assert (atr >= 0).all()