"""Execution contract."""

from abc import ABC, abstractmethod

import pandas as pd

from oil_bot.dto import Order, Trade


class IExecutor(ABC):
    """Contract for order executors."""

    @abstractmethod
    def execute(self, order: Order, current_bar: pd.Series) -> Trade:
        """Execute an order and return the resulting trade."""
        ...
