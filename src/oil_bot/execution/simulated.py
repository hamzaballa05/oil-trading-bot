"""Simulated executor for backtesting."""

import pandas as pd

from oil_bot.dto import Order, Trade
from oil_bot.execution.interfaces import IExecutor
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class SimulatedExecutor(IExecutor):
    """Simulates order execution at the next bar's open price."""

    def __init__(self, fees: float = 0.0005, slippage: float = 0.0003) -> None:
        self.fees = fees
        self.slippage = slippage

    def execute(self, order: Order, current_bar: pd.Series) -> Trade:
        """Execute the order at the open of the given bar (the T+1 bar)."""
        base_price = float(current_bar["open"])

        if order.side == "BUY":
            fill_price = base_price * (1 + self.slippage)
        else:
            fill_price = base_price * (1 - self.slippage)

        trade_value = fill_price * order.quantity
        fees_amount = trade_value * self.fees
        slippage_amount = abs(fill_price - base_price) * order.quantity

        return Trade(
            order=order,
            fill_timestamp=current_bar.name,
            fill_price=fill_price,
            quantity=order.quantity,
            fees=fees_amount,
            slippage=slippage_amount,
        )