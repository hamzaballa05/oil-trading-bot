"""Data Transfer Objects — structures shared between all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


@dataclass
class Signal:
    """A trading decision produced by a Strategy."""

    timestamp: datetime
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        """True if the signal is BUY or SELL (not HOLD)."""
        return self.action != "HOLD"


@dataclass
class Order:
    """A sized order, ready for execution."""

    timestamp: datetime
    side: Literal["BUY", "SELL"]
    quantity: float
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class Trade:
    """A filled order with real price, fees and slippage."""

    order: Order
    fill_timestamp: datetime
    fill_price: float
    quantity: float
    fees: float
    slippage: float


@dataclass
class Position:
    """An open position on an asset."""

    symbol: str
    quantity: float
    avg_entry_price: float
    entry_timestamp: datetime

    def market_value(self, current_price: float) -> float:
        """Current value of the position."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Profit or loss not yet realized."""
        return (current_price - self.avg_entry_price) * self.quantity


@dataclass
class PortfolioState:
    """Snapshot of the portfolio at one moment."""

    timestamp: datetime
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        """Total value: cash + positions."""
        positions_value = sum(
            pos.market_value(prices.get(sym, pos.avg_entry_price))
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value

    def is_flat(self) -> bool:
        """True if there are no open positions."""
        return not self.positions or all(
            p.quantity == 0 for p in self.positions.values()
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy configuration. frozen=True so it can be a cache key."""

    name: str
    params: tuple = ()


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration."""

    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05


@dataclass(frozen=True)
class BacktestConfig:
    """Full backtest configuration."""

    symbol: str = "CL=F"
    start: date = date(2018, 1, 1)
    end: date = date(2023, 12, 31)
    initial_capital: float = 100_000.0
    strategy: StrategyConfig = field(
        default_factory=lambda: StrategyConfig("rsi", (("period", 14),))
    )
    risk: RiskConfig = field(default_factory=RiskConfig)
    fees: float = 0.0005
    slippage: float = 0.0003
