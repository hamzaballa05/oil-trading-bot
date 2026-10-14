"""Data layer contract."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class IDataLoader(ABC):
    """Contract for data loaders."""

    @abstractmethod
    def load(
        self, symbol: str, start: date, end: date, timeframe: str = "1d"
    ) -> pd.DataFrame:
        """Load OHLCV data. Returns DataFrame with UTC DatetimeIndex."""
        ...