"""Fixed Fraction Risk Manager."""

import pandas as pd

from oil_bot.dto import Order, PortfolioState, Signal
from oil_bot.risk.interfaces import IRiskManager
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class FixedFractionRisk(IRiskManager):
    """Risks a fixed fraction of equity per trade."""

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        stop_loss_pct: float = 0.05,
        max_positions: int = 1,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.max_positions = max_positions

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        current_bar: pd.Series,
    ) -> Order | None:
        if signal.action == "HOLD":
            return None

        current_price = float(current_bar["close"])

        if signal.action == "BUY":
            if not portfolio.is_flat():
                return None

            prices = {"CL=F": current_price}
            equity = portfolio.equity(prices)

            risk_amount = equity * self.risk_per_trade
            stop_price = current_price * (1 - self.stop_loss_pct)
            risk_per_unit = current_price - stop_price

            if risk_per_unit <= 0:
                return None

            quantity = risk_amount / risk_per_unit
            required_cash = quantity * current_price

            if required_cash > portfolio.cash:
                quantity = (portfolio.cash * 0.95) / current_price
                if quantity <= 0:
                    return None

            return Order(
                timestamp=signal.timestamp,
                side="BUY",
                quantity=round(quantity, 6),
                order_type="MARKET",
                stop_loss=round(stop_price, 4),
            )

        if signal.action == "SELL":
            if portfolio.is_flat():
                return None
            symbol = next(iter(portfolio.positions))
            quantity = portfolio.positions[symbol].quantity
            return Order(
                timestamp=signal.timestamp,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
            )

        return None