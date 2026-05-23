"""Tests for Data Transfer Objects."""

from datetime import datetime

from oil_bot.dto import PortfolioState, Position, Signal


def test_buy_signal_is_actionable():
    sig = Signal(timestamp=datetime.now(), action="BUY")
    assert sig.is_actionable() is True


def test_hold_signal_is_not_actionable():
    sig = Signal(timestamp=datetime.now(), action="HOLD")
    assert sig.is_actionable() is False


def test_position_pnl_positive_when_price_rises():
    pos = Position("CL=F", 10, 80.0, datetime.now())
    assert pos.unrealized_pnl(90.0) == 100.0


def test_empty_portfolio_is_flat():
    portfolio = PortfolioState(datetime.now(), cash=100_000.0)
    assert portfolio.is_flat() is True
    assert portfolio.equity({}) == 100_000.0
