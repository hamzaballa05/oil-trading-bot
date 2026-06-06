"""Metrics calculator for backtest evaluation."""

import numpy as np
import pandas as pd


class MetricsCalculator:
    """Computes performance metrics from an equity curve."""

    TRADING_DAYS = 252

    def compute(
        self, equity: pd.Series, trades: list[dict] | None = None
    ) -> dict[str, float]:
        """Compute all performance metrics."""
        metrics: dict[str, float] = {}

        if len(equity) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "n_trades": 0}

        returns = equity.pct_change().dropna()

        metrics["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1)
        metrics["cagr"] = self._cagr(equity)
        metrics["sharpe"] = self._sharpe(returns)
        metrics["sortino"] = self._sortino(returns)
        metrics["max_drawdown"] = self._max_drawdown(equity)
        metrics["volatility"] = float(
            returns.std() * np.sqrt(self.TRADING_DAYS)
        )

        mdd = metrics["max_drawdown"]
        metrics["calmar"] = (
            metrics["cagr"] / abs(mdd) if mdd != 0 else 0.0
        )

        if trades:
            metrics.update(self._trade_metrics(trades))
        else:
            metrics["n_trades"] = 0
            metrics["win_rate"] = 0.0

        return {k: round(float(v), 6) for k, v in metrics.items()}

    def _cagr(self, equity: pd.Series) -> float:
        n_years = len(equity) / self.TRADING_DAYS
        if n_years <= 0 or equity.iloc[0] <= 0:
            return 0.0
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)

    def _sharpe(self, returns: pd.Series) -> float:
        std = returns.std()
        if std == 0 or pd.isna(std):
            return 0.0
        return float(returns.mean() / std * np.sqrt(self.TRADING_DAYS))

    def _sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < 0].std()
        if downside == 0 or pd.isna(downside):
            return 0.0
        return float(returns.mean() / downside * np.sqrt(self.TRADING_DAYS))

    def _max_drawdown(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    def _trade_metrics(self, trades: list[dict]) -> dict:
        sells = [t for t in trades if t.get("side") == "SELL"]
        n = len(sells)
        if n == 0:
            return {"n_trades": 0, "win_rate": 0.0}
        wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
        return {"n_trades": n, "win_rate": wins / n}