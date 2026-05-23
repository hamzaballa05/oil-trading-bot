"""Risk management contract."""

from abc import ABC, abstractmethod

import pandas as pd

from oil_bot.dto import Order, PortfolioState, Signal


class IRiskManager(ABC):
    """Contract for risk managers."""

    @abstractmethod
    def evaluate(
        self, signal: Signal, portfolio: PortfolioState, current_bar: pd.Series
    ) -> Order | None:
        """Turn a signal into a sized order, or None to reject it."""
        ...
