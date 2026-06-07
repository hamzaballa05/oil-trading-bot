"""Sentiment-based trading strategy."""

from datetime import datetime, timezone

import pandas as pd

from oil_bot.dto import Signal
from oil_bot.nlp.news_loader import NewsLoader
from oil_bot.nlp.sentiment import SentimentAnalyzer
from oil_bot.strategies.base import BaseStrategy
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentStrategy(BaseStrategy):
    """Trading strategy based on news sentiment.

    Logic:
        daily_sentiment > buy_threshold  → BUY
        daily_sentiment < sell_threshold → SELL
        otherwise                        → HOLD

    The sentiment score is computed from RSS feeds using TextBlob.
    Score range: [-1, +1]

    Args:
        buy_threshold: Minimum sentiment to trigger BUY.
        sell_threshold: Maximum sentiment to trigger SELL.
        news_loader: NewsLoader instance (optional, auto-created).
        analyzer: SentimentAnalyzer instance (optional, auto-created).
    """

    def __init__(
        self,
        buy_threshold: float = 0.1,
        sell_threshold: float = -0.1,
        news_loader: NewsLoader | None = None,
        analyzer: SentimentAnalyzer | None = None,
    ) -> None:
        super().__init__()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._loader = news_loader or NewsLoader()
        self._analyzer = analyzer or SentimentAnalyzer()
        self._cached_score: float | None = None
        self._cache_date: str | None = None

    @property
    def name(self) -> str:
        return (
            f"Sentiment(buy>{self.buy_threshold},"
            f"sell<{self.sell_threshold})"
        )

    @property
    def min_periods(self) -> int:
        return 1

    def get_current_sentiment(self) -> float:
        """Fetch news and compute today's sentiment score.

        Caches the result for the current day to avoid
        re-fetching on every call.

        Returns:
            Float in [-1, +1]. 0.0 if no news found.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._cache_date == today and self._cached_score is not None:
            return self._cached_score

        try:
            news = self._loader.fetch()
            if not news:
                logger.warning("No news fetched. Using neutral score.")
                return 0.0

            news_df = self._analyzer.analyze_news(news)
            if news_df.empty:
                return 0.0

            score = float(news_df["polarity"].mean())
            self._cached_score = score
            self._cache_date = today

            logger.info(
                f"Sentiment score: {score:.3f} "
                f"(from {len(news_df)} articles)"
            )
            return score

        except Exception as e:
            logger.warning(f"Sentiment fetch failed: {e}")
            return 0.0

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signal based on current news sentiment."""
        if not self._is_ready(data):
            return Signal(datetime.now(timezone.utc), "HOLD")

        bar = data.iloc[-1]
        score = self.get_current_sentiment()

        if score > self.buy_threshold:
            confidence = min(score / max(self.buy_threshold, 0.01), 1.0)
            return Signal(
                timestamp=bar.name,
                action="BUY",
                confidence=confidence,
                metadata={
                    "sentiment_score": round(score, 4),
                    "strategy": self.name,
                },
            )

        if score < self.sell_threshold:
            confidence = min(
                abs(score) / max(abs(self.sell_threshold), 0.01),
                1.0,
            )
            return Signal(
                timestamp=bar.name,
                action="SELL",
                confidence=confidence,
                metadata={
                    "sentiment_score": round(score, 4),
                    "strategy": self.name,
                },
            )

        return Signal(
            timestamp=bar.name,
            action="HOLD",
            metadata={"sentiment_score": round(score, 4)},
        )