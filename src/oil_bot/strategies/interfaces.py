"""Strategy contract."""

from abc import ABC, abstractmethod

import pandas as pd

from oil_bot.dto import Signal


class IStrategy(ABC):
    """Contract for trading strategies."""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Produce a BUY/SELL/HOLD signal from data up to the latest bar."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    @abstractmethod
    def min_periods(self) -> int:
        """Minimum bars needed before the strategy can signal."""
        ...
