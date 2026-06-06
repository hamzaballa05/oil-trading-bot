"""MACD indicator."""

import pandas as pd

from oil_bot.exceptions import InsufficientDataError
from oil_bot.indicators.interfaces import IIndicator


class MACD(IIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def min_periods(self) -> int:
        return self.slow + self.signal

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(data) < self.min_periods:
            raise InsufficientDataError(
                f"MACD needs >= {self.min_periods} bars."
            )

        close = data["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_hist": histogram,
            },
            index=data.index,
        )