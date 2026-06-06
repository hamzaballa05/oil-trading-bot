"""Base class for strategies."""

import pandas as pd

from oil_bot.strategies.interfaces import IStrategy
from oil_bot.utils.logging import get_logger


class BaseStrategy(IStrategy):
    """Shared utilities for all strategies."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def _is_ready(self, data: pd.DataFrame) -> bool:
        """Check there is enough data for this strategy."""
        return len(data) >= self.min_periods