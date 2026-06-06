"""BacktestService - orchestrates a full backtest."""

from oil_bot.backtesting.engine import Backtester, BacktestResult
from oil_bot.data.yahoo_loader import YahooFinanceLoader
from oil_bot.dto import BacktestConfig
from oil_bot.exceptions import StrategyNotFoundError
from oil_bot.execution.simulated import SimulatedExecutor
from oil_bot.features.engine import FeatureEngine
from oil_bot.risk.fixed_fraction import FixedFractionRisk
from oil_bot.strategies.combined import CombinedStrategy
from oil_bot.strategies.ma_crossover import MaCrossoverStrategy
from oil_bot.strategies.rsi_strategy import RsiStrategy
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)

STRATEGIES = {
    "rsi": RsiStrategy,
    "ma_crossover": MaCrossoverStrategy,
    "combined": CombinedStrategy,
}


class BacktestService:
    """The single entry point for running a backtest."""

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run a complete backtest from a config."""
        df = YahooFinanceLoader().load(config.symbol, config.start, config.end)
        df = FeatureEngine().transform(df)

        strat_class = STRATEGIES.get(config.strategy.name)
        if strat_class is None:
            raise StrategyNotFoundError(
                f"Unknown strategy '{config.strategy.name}'. "
                f"Available: {list(STRATEGIES)}"
            )
        strategy = strat_class(**dict(config.strategy.params))

        risk = FixedFractionRisk(
            risk_per_trade=config.risk.risk_per_trade,
            stop_loss_pct=config.risk.stop_loss_pct,
        )
        executor = SimulatedExecutor(
            fees=config.fees, slippage=config.slippage
        )

        backtester = Backtester(
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            initial_capital=config.initial_capital,
        )
        return backtester.run(df, config_snapshot=config)