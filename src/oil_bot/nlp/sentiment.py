"""Sentiment analyzer using TextBlob."""

import pandas as pd
from textblob import TextBlob

from oil_bot.nlp.news_loader import NewsItem
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of news articles using TextBlob.

    TextBlob returns:
        polarity    : float in [-1, +1]
                      -1 = very negative, +1 = very positive
        subjectivity: float in [0, 1]
                      0 = objective, 1 = subjective

    For trading, we use polarity as the sentiment score.

    Args:
        min_subjectivity: Ignore articles below this subjectivity.
                          Very objective text (news) has low scores.
    """

    def __init__(self, min_subjectivity: float = 0.0) -> None:
        self.min_subjectivity = min_subjectivity

    def analyze(self, text: str) -> dict[str, float]:
        """Analyze sentiment of a text.

        Args:
            text: Article title or summary.

        Returns:
            Dict with 'polarity' and 'subjectivity'.
        """
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def analyze_news(self, news: list[NewsItem]) -> pd.DataFrame:
        """Analyze a list of news items.

        Args:
            news: List of NewsItem objects.

        Returns:
            DataFrame with columns:
            [title, published, source, polarity, subjectivity]
        """
        rows = []
        for item in news:
            text = item.title + ". " + item.summary
            scores = self.analyze(text)
            rows.append({
                "title": item.title[:100],
                "published": item.published,
                "source": item.source,
                "polarity": scores["polarity"],
                "subjectivity": scores["subjectivity"],
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["published"] = pd.to_datetime(df["published"])
        df = df.sort_values("published", ascending=False)
        logger.info(
            f"Analyzed {len(df)} articles. "
            f"Mean polarity: {df['polarity'].mean():.3f}"
        )
        return df

    def daily_score(self, news_df: pd.DataFrame) -> pd.Series:
        """Aggregate news sentiment into a daily score.

        For each day, average the polarity of all articles.

        Args:
            news_df: DataFrame from analyze_news().

        Returns:
            Series indexed by date with daily sentiment score.
        """
        if news_df.empty:
            return pd.Series(dtype=float)

        news_df = news_df.copy()
        news_df["date"] = news_df["published"].dt.date
        daily = news_df.groupby("date")["polarity"].mean()
        daily.index = pd.to_datetime(daily.index)
        return daily