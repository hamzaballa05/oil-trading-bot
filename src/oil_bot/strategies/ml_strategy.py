"""ML-based trading strategy."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from oil_bot.dto import Signal
from oil_bot.ml.feature_engineering import MlFeatureEngine
from oil_bot.strategies.base import BaseStrategy


class MlStrategy(BaseStrategy):
    """Trading strategy powered by a trained ML model.

    Args:
        model: Trained sklearn model with a .predict() method.
        feature_columns: List of feature names the model expects.
        buy_threshold: Minimum predicted return to trigger BUY.
        sell_threshold: Maximum predicted return to trigger SELL.
        horizon: Prediction horizon in days.
    """

    def __init__(
        self,
        model,
        feature_columns: list[str],
        buy_threshold: float = 0.002,
        sell_threshold: float = -0.002,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.feature_columns = feature_columns
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.feature_engine = MlFeatureEngine(horizon=horizon)

    @property
    def name(self) -> str:
        return (
            f"ML_RF(buy>{self.buy_threshold:.3f},"
            f"sell<{self.sell_threshold:.3f})"
        )

    @property
    def min_periods(self) -> int:
        return 60

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Predict next-day return and convert to a trading signal."""
        if not self._is_ready(data):
            return Signal(datetime.now(timezone.utc), "HOLD")

        bar = data.iloc[-1]

        try:
            features = self.feature_engine.build_features(data)
            last_features = features[self.feature_columns].iloc[[-1]]

            # Nettoyer les infinis et NaN
            last_features = last_features.replace(
                [np.inf, -np.inf], np.nan
            )

            if last_features.isna().any().any():
                return Signal(
                    bar.name,
                    "HOLD",
                    metadata={"reason": "nan_or_inf_features"},
                )

            predicted_return = float(
                self.model.predict(last_features)[0]
            )

        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return Signal(
                bar.name,
                "HOLD",
                metadata={"reason": f"error:{e}"},
            )

        if predicted_return > self.buy_threshold:
            confidence = min(
                predicted_return / (self.buy_threshold * 5), 1.0
            )
            return Signal(
                timestamp=bar.name,
                action="BUY",
                confidence=confidence,
                metadata={
                    "predicted_return": round(predicted_return, 6),
                    "model": "RandomForest",
                },
            )

        if predicted_return < self.sell_threshold:
            confidence = min(
                abs(predicted_return) / (abs(self.sell_threshold) * 5),
                1.0,
            )
            return Signal(
                timestamp=bar.name,
                action="SELL",
                confidence=confidence,
                metadata={
                    "predicted_return": round(predicted_return, 6),
                    "model": "RandomForest",
                },
            )

        return Signal(
            timestamp=bar.name,
            action="HOLD",
            metadata={
                "predicted_return": round(predicted_return, 6)
            },
        )