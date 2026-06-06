"""Backtesting engine - orchestrates the full simulation loop."""

import uuid
from dataclasses import dataclass, field

import pandas as pd

from oil_bot.dto import BacktestConfig, PortfolioState, Position, Trade
from oil_bot.execution.interfaces import IExecutor
from oil_bot.metrics.calculator import MetricsCalculator
from oil_bot.risk.interfaces import IRiskManager
from oil_bot.strategies.interfaces import IStrategy
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)

SYMBOL_KEY = "ASSET"


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    run_id: str
    config_snapshot: BacktestConfig
    equity_curve: pd.Series
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict = field(default_factory=dict)


class Backtester:
    """Runs a historical simulation, bar by bar."""

    def __init__(
        self,
        strategy: IStrategy,
        risk_manager: IRiskManager,
        executor: IExecutor,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.executor = executor
        self.initial_capital = initial_capital

    def run(
        self,
        data: pd.DataFrame,
        config_snapshot: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Run the backtest on an enriched DataFrame."""
        run_id = str(uuid.uuid4())
        logger.info(
            f"[{run_id[:8]}] Backtest start: {self.strategy.name}, "
            f"{len(data)} bars."
        )

        portfolio = PortfolioState(
            timestamp=data.index[0], cash=self.initial_capital
        )

        equity_curve: list[float] = []
        trades_log: list[dict] = []
        signals_log: list[dict] = []

        n = len(data)
        for i in range(n):
            current_bar = data.iloc[i]
            current_price = float(current_bar["close"])
            prices = {SYMBOL_KEY: current_price}

            # 1. Record equity at this bar
            equity_curve.append(portfolio.equity(prices))

            # 2. Generate signal using only data up to bar i
            history = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(history)
            signals_log.append(
                {
                    "timestamp": current_bar.name,
                    "action": signal.action,
                    "confidence": signal.confidence,
                }
            )

            # 3. Risk manager evaluates the signal
            order = self.risk_manager.evaluate(signal, portfolio, current_bar)

            # 4. Execute at the OPEN of bar i+1 (anti look-ahead)
            if order is not None and i < n - 1:
                next_bar = data.iloc[i + 1]
                trade = self.executor.execute(order, next_bar)
                self._apply_trade(trade, portfolio, trades_log)

        # Close any remaining position at the last close
        if not portfolio.is_flat():
            self._force_close(portfolio, data.iloc[-1], trades_log)

        equity_series = pd.Series(
            equity_curve, index=data.index, name="equity"
        )
        metrics = MetricsCalculator().compute(equity_series, trades_log)

        logger.info(
            f"[{run_id[:8]}] Done. Trades: {metrics.get('n_trades', 0)}, "
            f"Return: {metrics.get('total_return', 0):.2%}, "
            f"Sharpe: {metrics.get('sharpe', 0):.2f}"
        )

        return BacktestResult(
            run_id=run_id,
            config_snapshot=config_snapshot or BacktestConfig(),
            equity_curve=equity_series,
            trades=pd.DataFrame(trades_log),
            signals=pd.DataFrame(signals_log),
            metrics=metrics,
        )

    def _apply_trade(
        self, trade: Trade, portfolio: PortfolioState, trades_log: list[dict]
    ) -> None:
        """Update portfolio after a trade and record it."""
        side = trade.order.side

        if side == "BUY":
            cost = trade.fill_price * trade.quantity + trade.fees
            portfolio.cash -= cost
            portfolio.positions[SYMBOL_KEY] = Position(
                symbol=SYMBOL_KEY,
                quantity=trade.quantity,
                avg_entry_price=trade.fill_price,
                entry_timestamp=trade.fill_timestamp,
            )
            trades_log.append(
                {
                    "timestamp": trade.fill_timestamp,
                    "side": "BUY",
                    "quantity": trade.quantity,
                    "fill_price": trade.fill_price,
                    "fees": trade.fees,
                    "pnl": 0.0,
                }
            )

        elif side == "SELL" and SYMBOL_KEY in portfolio.positions:
            pos = portfolio.positions[SYMBOL_KEY]
            revenue = trade.fill_price * trade.quantity - trade.fees
            cost_basis = pos.avg_entry_price * pos.quantity
            pnl = revenue - cost_basis
            portfolio.cash += revenue
            del portfolio.positions[SYMBOL_KEY]
            trades_log.append(
                {
                    "timestamp": trade.fill_timestamp,
                    "side": "SELL",
                    "quantity": trade.quantity,
                    "fill_price": trade.fill_price,
                    "fees": trade.fees,
                    "pnl": pnl,
                }
            )

    def _force_close(
        self,
        portfolio: PortfolioState,
        last_bar: pd.Series,
        trades_log: list[dict],
    ) -> None:
        """Force-close any open position at the last close price."""
        if SYMBOL_KEY not in portfolio.positions:
            return
        pos = portfolio.positions[SYMBOL_KEY]
        close_price = float(last_bar["close"])
        revenue = close_price * pos.quantity
        cost_basis = pos.avg_entry_price * pos.quantity
        pnl = revenue - cost_basis
        portfolio.cash += revenue
        del portfolio.positions[SYMBOL_KEY]
        trades_log.append(
            {
                "timestamp": last_bar.name,
                "side": "SELL",
                "quantity": pos.quantity,
                "fill_price": close_price,
                "fees": 0.0,
                "pnl": pnl,
            }
        )