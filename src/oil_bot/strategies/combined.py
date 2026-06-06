"""Combined RSI + MA trend strategy."""

from datetime import datetime, timezone

import pandas as pd

from oil_bot.dto import Signal
from oil_bot.strategies.base import BaseStrategy
from oil_bot.strategies.rsi_strategy import RsiStrategy


class CombinedStrategy(BaseStrategy):
    """RSI signal confirmed by EMA trend."""

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
    ) -> None:
        super().__init__()
        self._rsi = RsiStrategy(rsi_period, rsi_oversold, rsi_overbought)

    @property
    def name(self) -> str:
        return "Combined(RSI+MA)"

    @property
    def min_periods(self) -> int:
        return 51

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_ready(data):
            return Signal(datetime.now(timezone.utc), "HOLD")

        bar = data.iloc[-1]
        rsi_signal = self._rsi.generate_signal(data)

        ema_fast = bar.get("ema_20")
        ema_slow = bar.get("ema_50")

        if any(pd.isna(v) for v in [ema_fast, ema_slow]):
            return Signal(bar.name, "HOLD")

        in_uptrend = ema_fast > ema_slow
        in_downtrend = ema_fast < ema_slow

        if rsi_signal.action == "BUY" and in_uptrend:
            return Signal(
                bar.name,
                "BUY",
                confidence=min(rsi_signal.confidence * 1.1, 1.0),
                metadata={**rsi_signal.metadata, "trend": "UP"},
            )

        if rsi_signal.action == "SELL" and in_downtrend:
            return Signal(
                bar.name,
                "SELL",
                confidence=min(rsi_signal.confidence * 1.1, 1.0),
                metadata={**rsi_signal.metadata, "trend": "DOWN"},
            )

        return Signal(bar.name, "HOLD")