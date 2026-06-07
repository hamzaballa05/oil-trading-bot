"""Backtesting engine."""

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
        rid = str(uuid.uuid4())
        logger.info(
            f"[{rid[:8]}] Backtest: {self.strategy.name}, "
            f"{len(data)} bars."
        )

        portfolio = PortfolioState(
            timestamp=data.index[0],
            cash=self.initial_capital,
        )

        eq, tl, sl = [], [], []
        n = len(data)

        for i in range(n):
            bar = data.iloc[i]
            current_price = float(bar["close"])
            low_price = float(bar["low"])
            prices = {SYMBOL_KEY: current_price}

            # 1. Enregistrer l'equity
            eq.append(portfolio.equity(prices))

            # 2. Vérifier le stop loss
            stop_triggered = False
            if SYMBOL_KEY in portfolio.positions:
                pos = portfolio.positions[SYMBOL_KEY]
                stop = getattr(pos, "stop_loss", None)
                if stop is not None and low_price <= stop:
                    # Stop loss déclenché
                    stop_price = float(stop)
                    revenue = stop_price * pos.quantity
                    pnl = revenue - pos.avg_entry_price * pos.quantity
                    pnl_pct = (
                        (stop_price - pos.avg_entry_price)
                        / pos.avg_entry_price
                    )
                    portfolio.cash += revenue
                    del portfolio.positions[SYMBOL_KEY]

                    tl.append({
                        "timestamp": bar.name,
                        "side": "SELL (SL)",
                        "quantity": pos.quantity,
                        "fill_price": stop_price,
                        "entry_price": pos.avg_entry_price,
                        "stop_loss": stop_price,
                        "fees": 0.0,
                        "pnl": pnl,
                        "pnl_pct": round(pnl_pct, 6),
                    })
                    sl.append({
                        "timestamp": bar.name,
                        "action": "STOP_LOSS",
                        "confidence": 1.0,
                    })
                    stop_triggered = True

            if stop_triggered:
                continue

            # 3. Générer le signal
            history = data.iloc[: i + 1]
            signal = self.strategy.generate_signal(history)
            sl.append({
                "timestamp": bar.name,
                "action": signal.action,
                "confidence": signal.confidence,
            })

            # 4. Évaluer le risque
            order = self.risk_manager.evaluate(signal, portfolio, bar)

            # 5. Exécuter au bar suivant (anti look-ahead)
            if order is not None and i < n - 1:
                next_bar = data.iloc[i + 1]
                trade = self.executor.execute(order, next_bar)
                self._apply(trade, portfolio, tl)

        # Fermer la position restante
        if not portfolio.is_flat():
            self._close(portfolio, data.iloc[-1], tl)

        es = pd.Series(eq, index=data.index, name="equity")
        metrics = MetricsCalculator().compute(es, tl)

        logger.info(
            f"[{rid[:8]}] Done. "
            f"Trades: {metrics.get('n_trades', 0)}, "
            f"Return: {metrics.get('total_return', 0):.2%}, "
            f"Sharpe: {metrics.get('sharpe', 0):.2f}"
        )

        return BacktestResult(
            rid,
            config_snapshot or BacktestConfig(),
            es,
            pd.DataFrame(tl),
            pd.DataFrame(sl),
            metrics,
        )

    def _apply(
        self, trade: Trade, portfolio: PortfolioState, tl: list
    ) -> None:
        """Update portfolio after a trade."""
        if trade.order.side == "BUY":
            portfolio.cash -= (
                trade.fill_price * trade.quantity + trade.fees
            )
            pos = Position(
                symbol=SYMBOL_KEY,
                quantity=trade.quantity,
                avg_entry_price=trade.fill_price,
                entry_timestamp=trade.fill_timestamp,
            )
            # Stocker le stop loss dans la position
            pos.stop_loss = trade.order.stop_loss
            portfolio.positions[SYMBOL_KEY] = pos

            tl.append({
                "timestamp": trade.fill_timestamp,
                "side": "BUY",
                "quantity": trade.quantity,
                "fill_price": trade.fill_price,
                "entry_price": trade.fill_price,
                "stop_loss": trade.order.stop_loss,
                "fees": trade.fees,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            })

        elif (
            trade.order.side == "SELL"
            and SYMBOL_KEY in portfolio.positions
        ):
            pos = portfolio.positions[SYMBOL_KEY]
            revenue = trade.fill_price * trade.quantity - trade.fees
            pnl = revenue - pos.avg_entry_price * pos.quantity
            pnl_pct = (
                (trade.fill_price - pos.avg_entry_price)
                / pos.avg_entry_price
            )
            portfolio.cash += revenue
            del portfolio.positions[SYMBOL_KEY]

            tl.append({
                "timestamp": trade.fill_timestamp,
                "side": "SELL",
                "quantity": trade.quantity,
                "fill_price": trade.fill_price,
                "entry_price": pos.avg_entry_price,
                "stop_loss": None,
                "fees": trade.fees,
                "pnl": pnl,
                "pnl_pct": round(pnl_pct, 6),
            })

    def _close(
        self,
        portfolio: PortfolioState,
        last_bar: pd.Series,
        tl: list,
    ) -> None:
        """Force-close any open position at last close price."""
        if SYMBOL_KEY not in portfolio.positions:
            return
        pos = portfolio.positions[SYMBOL_KEY]
        price = float(last_bar["close"])
        pnl = price * pos.quantity - pos.avg_entry_price * pos.quantity
        pnl_pct = (price - pos.avg_entry_price) / pos.avg_entry_price
        portfolio.cash += price * pos.quantity
        del portfolio.positions[SYMBOL_KEY]

        tl.append({
            "timestamp": last_bar.name,
            "side": "SELL",
            "quantity": pos.quantity,
            "fill_price": price,
            "entry_price": pos.avg_entry_price,
            "stop_loss": None,
            "fees": 0.0,
            "pnl": pnl,
            "pnl_pct": round(pnl_pct, 6),
        })