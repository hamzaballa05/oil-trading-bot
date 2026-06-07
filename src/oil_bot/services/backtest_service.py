"""BacktestService - orchestrates a full backtest."""

from datetime import date, timedelta

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

# 120 jours de chauffe = assez pour SMA(50) + MACD(26+9)
WARMUP_DAYS = 120


class BacktestService:
    """Single entry point for running a backtest."""

    def run(self, config: BacktestConfig) -> BacktestResult:

        # 1. Calculer la date de début étendue (avec chauffe)
        extended_start = date.fromordinal(
            config.start.toordinal() - WARMUP_DAYS
        )

        # 2. Charger plus de données que demandé
        df_full = YahooFinanceLoader().load(
            config.symbol, extended_start, config.end
        )

        # 3. Calculer les indicateurs sur la période complète
        df_enriched = FeatureEngine().transform(df_full)

        # 4. Couper à la vraie date de début demandée par l'utilisateur
        start_ts = str(config.start)
        df = df_enriched[df_enriched.index >= start_ts].copy()

        if df.empty:
            from oil_bot.exceptions import DataNotAvailableError
            raise DataNotAvailableError(
                f"Aucune donnée à partir du {config.start}."
            )

        logger.info(
            f"Backtest: {len(df)} bars effectifs "
            f"({config.start} → {config.end})"
        )

        # 5. Créer les objets
        strat_class = STRATEGIES.get(config.strategy.name)
        if strat_class is None:
            raise StrategyNotFoundError(
                f"Stratégie inconnue : '{config.strategy.name}'. "
                f"Disponibles : {list(STRATEGIES)}"
            )
        strategy = strat_class(**dict(config.strategy.params))
        risk = FixedFractionRisk(
            config.risk.risk_per_trade,
            config.risk.stop_loss_pct,
        )
        executor = SimulatedExecutor(
            fees=config.fees,
            slippage=config.slippage,
        )

        # 6. Lancer le backtest
        backtester = Backtester(
            strategy=strategy,
            risk_manager=risk,
            executor=executor,
            initial_capital=config.initial_capital,
        )
        return backtester.run(df, config_snapshot=config)