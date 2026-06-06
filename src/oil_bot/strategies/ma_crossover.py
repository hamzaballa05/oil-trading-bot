"""EMA crossover strategy."""

from datetime import datetime, timezone

import pandas as pd

from oil_bot.dto import Signal
from oil_bot.strategies.base import BaseStrategy


class MaCrossoverStrategy(BaseStrategy):
    """EMA fast/slow crossover strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        return f"MA_Crossover(EMA{self.fast_period}/EMA{self.slow_period})"

    @property
    def min_periods(self) -> int:
        return self.slow_period + 1

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_ready(data):
            return Signal(datetime.now(timezone.utc), "HOLD")

        curr = data.iloc[-1]
        prev = data.iloc[-2]

        fc = curr.get("ema_20")
        sc = curr.get("ema_50")
        fp = prev.get("ema_20")
        sp = prev.get("ema_50")

        if any(pd.isna(v) for v in [fc, sc, fp, sp]):
            return Signal(curr.name, "HOLD")

        if fp <= sp and fc > sc:
            return Signal(
                curr.name,
                "BUY",
                confidence=0.6,
                metadata={"ema_fast": round(float(fc), 2)},
            )

        if fp >= sp and fc < sc:
            return Signal(
                curr.name,
                "SELL",
                confidence=0.6,
                metadata={"ema_fast": round(float(fc), 2)},
            )

        return Signal(curr.name, "HOLD")