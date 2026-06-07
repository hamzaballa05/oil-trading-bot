"""ML-specific feature engineering.

Extends the V1 FeatureEngine with ML-ready features:
- Lagged returns (to capture momentum)
- Rolling statistics (volatility regimes)
- Price ratios (relative position)
- Target variable construction

CRITICAL: all features are CAUSAL — computed only from past data.
No look-ahead bias.
"""

import numpy as np
import pandas as pd

from oil_bot.features.engine import FeatureEngine
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class MlFeatureEngine:
    """Creates ML-ready feature matrix from OHLCV data.

    Combines V1 indicators with additional ML features:
    - Lagged returns (1, 2, 3, 5, 10 days)
    - Rolling volatility (5, 10, 20 days)
    - Rolling mean return (5, 10, 20 days)
    - Price relative to moving averages
    - Volume change
    - Day of week

    All features are strictly causal (no future data).
    """

    def __init__(
        self,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        horizon: int = 1,
    ) -> None:
        """Initialize ML feature engine.

        Args:
            lags: List of lag periods for returns. Default: [1, 2, 3, 5, 10].
            rolling_windows: Rolling stats windows. Default: [5, 10, 20].
            horizon: Prediction horizon in days. Default: 1.
        """
        self.lags = lags or [1, 2, 3, 5, 10]
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.horizon = horizon
        self._v1_engine = FeatureEngine()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build complete feature matrix from OHLCV data.

        Args:
            df: Validated OHLCV DataFrame.

        Returns:
            DataFrame with all features. NaN rows NOT dropped
            (caller decides how to handle them).
        """
        # Start with V1 indicators
        result = self._v1_engine.transform(df)

        close = df["close"]
        returns = close.pct_change()

        # --- Lagged returns ---
        for lag in self.lags:
            result[f"return_lag_{lag}"] = returns.shift(lag)

        # --- Rolling statistics ---
        for w in self.rolling_windows:
            result[f"rolling_mean_{w}"] = returns.rolling(w).mean()
            result[f"rolling_std_{w}"] = returns.rolling(w).std()
            result[f"rolling_min_{w}"] = returns.rolling(w).min()
            result[f"rolling_max_{w}"] = returns.rolling(w).max()

        # --- Price ratios ---
        result["price_to_sma20"] = close / result["sma_20"]
        result["price_to_sma50"] = close / result["sma_50"]
        result["price_to_ema20"] = close / result["ema_20"]

        # --- Volume features ---
        result["volume_change"] = df["volume"].pct_change()
        result["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

        # --- High-Low range ---
        result["range_pct"] = (df["high"] - df["low"]) / close

        # --- Day of week (0=Monday, 4=Friday) ---
        result["day_of_week"] = df.index.dayofweek

        n_features = len(result.columns) - len(df.columns)
        logger.debug(f"MlFeatureEngine: created {n_features} features.")

        return result

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """Build the target variable: next-day return.

        target_t = (close_{t+horizon} - close_t) / close_t

        This is the value we want the model to PREDICT.
        It uses future data — that's normal for a target.
        But it must NEVER be included in the features.

        Args:
            df: OHLCV DataFrame (needs 'close' column).

        Returns:
            Series of next-day returns. Last `horizon` values are NaN.
        """
        target = df["close"].pct_change(self.horizon).shift(-self.horizon)
        target.name = "target"
        return target

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names (excluding OHLCV and target)."""
        exclude = {"open", "high", "low", "close", "volume", "target"}
        return [c for c in df.columns if c not in exclude]