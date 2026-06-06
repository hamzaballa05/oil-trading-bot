"""RSI mean-reversion strategy."""

from datetime import datetime, timezone

import pandas as pd

from oil_bot.dto import Signal
from oil_bot.strategies.base import BaseStrategy


class RsiStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI({self.period}, {self.oversold}/{self.overbought})"

    @property
    def min_periods(self) -> int:
        return self.period + 1

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_ready(data):
            return Signal(datetime.now(timezone.utc), "HOLD")

        bar = data.iloc[-1]
        rsi = bar.get("rsi_14")

        if rsi is None or pd.isna(rsi):
            return Signal(bar.name, "HOLD", metadata={"reason": "rsi_nan"})

        if rsi < self.oversold:
            action = "BUY"
            confidence = (self.oversold - rsi) / self.oversold
        elif rsi > self.overbought:
            action = "SELL"
            confidence = (rsi - self.overbought) / (100 - self.overbought)
        else:
            action = "HOLD"
            confidence = 0.5

        return Signal(
            timestamp=bar.name,
            action=action,
            confidence=min(max(confidence, 0.0), 1.0),
            metadata={"rsi": round(float(rsi), 2)},
        )