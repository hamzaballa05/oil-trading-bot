"""Bollinger Bands indicator."""

import pandas as pd

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.interfaces import IIndicator


class BollingerBands(IIndicator):
    """Bollinger Bands: middle, upper, lower bands."""

    def __init__(
        self, period: int = 20, std_dev: float = 2.0, column: str = "close"
    ) -> None:
        self.period = period
        self.std_dev = std_dev
        self.column = column

    @property
    def min_periods(self) -> int:
        return self.period

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.period:
            raise InsufficientDataError(
                f"BollingerBands({self.period}) needs >= {self.period} bars."
            )

        price = data[self.column]
        middle = price.rolling(self.period, min_periods=self.period).mean()
        std = price.rolling(self.period, min_periods=self.period).std()

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        width = (upper - lower) / middle
        pct = (price - lower) / (upper - lower)

        return pd.DataFrame(
            {
                "bb_middle": middle,
                "bb_upper": upper,
                "bb_lower": lower,
                "bb_width": width,
                "bb_pct": pct,
            },
            index=data.index,
        )