"""ComparisonService - runs multiple backtests for comparison."""

from oil_bot.backtesting.engine import BacktestResult
from oil_bot.dto import BacktestConfig
from oil_bot.services.backtest_service import BacktestService


class ComparisonService:
    """Runs several backtests and returns their results together."""

    def run_batch(
        self, labeled_configs: dict[str, BacktestConfig]
    ) -> dict[str, BacktestResult]:
        """Run a backtest for each labeled config."""
        service = BacktestService()
        results: dict[str, BacktestResult] = {}
        for label, config in labeled_configs.items():
            results[label] = service.run(config)
        return results