"""Relative Strength Index using Wilder's smoothing."""

import pandas as pd

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.interfaces import IIndicator


class RSI(IIndicator):
    """RSI with Wilder's exponential smoothing (industry standard)."""

    def __init__(self, period: int = 14, column: str = "close") -> None:
        self.period = period
        self.column = column

    @property
    def min_periods(self) -> int:
        return self.period + 1

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.min_periods:
            raise InsufficientDataError(
                f"RSI({self.period}) needs >= {self.min_periods} bars."
            )

        prices = data[self.column]
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        alpha = 1.0 / self.period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # When avg_loss is 0 (only gains) -> RSI = 100
        rsi = rsi.where(avg_loss != 0, 100.0)
        # When avg_gain is 0 (only losses) -> RSI = 0
        rsi = rsi.where(avg_gain != 0, 0.0)

        rsi.iloc[: self.period] = float("nan")
        return rsi