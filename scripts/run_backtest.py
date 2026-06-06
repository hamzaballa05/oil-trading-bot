"""Run a backtest from the command line."""

import sys
from datetime import date

sys.path.insert(0, "src")

from oil_bot.dto import BacktestConfig, StrategyConfig
from oil_bot.services.backtest_service import BacktestService


def main() -> None:
    config = BacktestConfig(
        symbol="CL=F",
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
        initial_capital=100_000.0,
        strategy=StrategyConfig("rsi", (("period", 14),)),
    )

    result = BacktestService().run(config)

    print("\n" + "=" * 52)
    print(f"  Run ID   : {result.run_id[:8]}")
    print(f"  Strategy : {config.strategy.name}")
    print("-" * 52)
    for name, value in result.metrics.items():
        print(f"  {name:22s}: {value}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()