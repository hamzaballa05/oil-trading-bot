"""Indicator contract."""

from abc import ABC, abstractmethod

import pandas as pd


class IIndicator(ABC):
    """Contract for indicators. Must be pure and causal."""

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Compute the indicator from OHLCV data."""
        ...

    @property
    @abstractmethod
    def min_periods(self) -> int:
        """Minimum rows needed for a valid output."""
        ...
