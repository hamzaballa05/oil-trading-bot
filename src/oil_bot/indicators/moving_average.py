"""Moving average indicators: SMA and EMA."""

import pandas as pd

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.interfaces import IIndicator


class SMA(IIndicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20, column: str = "close") -> None:
        self.period = period
        self.column = column

    @property
    def min_periods(self) -> int:
        return self.period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"SMA({self.period}) needs >= {self.period} bars."
            )
        return data[self.column].rolling(
            window=self.period, min_periods=self.period
        ).mean()


class EMA(IIndicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 20, column: str = "close") -> None:
        self.period = period
        self.column = column

    @property
    def min_periods(self) -> int:
        return self.period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"EMA({self.period}) needs >= {self.period} bars."
            )
        return data[self.column].ewm(span=self.period, adjust=False).mean()