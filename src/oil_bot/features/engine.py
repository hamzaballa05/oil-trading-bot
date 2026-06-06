"""Feature Engine - enriches OHLCV data with technical indicators."""

import pandas as pd

from oil_bot.indicators.atr import ATR
from oil_bot.indicators.bollinger import BollingerBands
from oil_bot.indicators.macd import MACD
from oil_bot.indicators.moving_average import EMA, SMA
from oil_bot.indicators.rsi import RSI
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngine:
    """Adds technical indicators to an OHLCV DataFrame."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all standard indicators and add them as columns."""
        result = df.copy()
        n_before = len(result.columns)

        result["sma_20"] = SMA(period=20).compute(df)
        result["sma_50"] = SMA(period=50).compute(df)
        result["ema_20"] = EMA(period=20).compute(df)
        result["ema_50"] = EMA(period=50).compute(df)
        result["rsi_14"] = RSI(period=14).compute(df)

        bb = BollingerBands(period=20, std_dev=2.0).compute(df)
        result = pd.concat([result, bb], axis=1)

        macd = MACD(fast=12, slow=26, signal=9).compute(df)
        result = pd.concat([result, macd], axis=1)

        result["atr_14"] = ATR(period=14).compute(df)
        result["close_pct_change"] = df["close"].pct_change()

        logger.info(
            f"FeatureEngine: added {len(result.columns) - n_before} columns."
        )
        return result