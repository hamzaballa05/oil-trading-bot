"""Average True Range indicator."""

import pandas as pd

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.interfaces import IIndicator


class ATR(IIndicator):
    """Average True Range using Wilder's smoothing."""

    def __init__(self, period: int = 14) -> None:
        self.period = period

    @property
    def min_periods(self) -> int:
        return self.period + 1

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if len(data) < self.min_periods:
            raise InsufficientDataError(
                f"ATR({self.period}) needs >= {self.min_periods} bars."
            )

        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.ewm(alpha=1 / self.period, adjust=False).mean()