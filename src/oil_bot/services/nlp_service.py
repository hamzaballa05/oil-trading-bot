"""NLP Service — facade for sentiment analysis."""

from oil_bot.nlp.news_loader import NewsLoader
from oil_bot.nlp.sentiment import SentimentAnalyzer
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class NlpService:
    """Facade for news sentiment operations.

    Used by Streamlit and CLI.
    """

    def __init__(self) -> None:
        self._loader = NewsLoader()
        self._analyzer = SentimentAnalyzer()

    def fetch_and_analyze(self) -> dict:
        """Fetch news and return sentiment analysis.

        Returns:
            Dict with:
            - news_df: DataFrame of articles with scores
            - current_score: float, today's mean polarity
            - signal: 'BUY', 'SELL', or 'HOLD'
            - n_articles: int
        """
        news = self._loader.fetch()

        if not news:
            return {
                "news_df": None,
                "current_score": 0.0,
                "signal": "HOLD",
                "n_articles": 0,
            }

        news_df = self._analyzer.analyze_news(news)
        current_score = float(news_df["polarity"].mean())

        if current_score > 0.1:
            signal = "BUY"
        elif current_score < -0.1:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "news_df": news_df,
            "current_score": current_score,
            "signal": signal,
            "n_articles": len(news_df),
        }