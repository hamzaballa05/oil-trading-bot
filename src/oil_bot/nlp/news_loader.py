"""News loader — fetches oil-related news from RSS feeds."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser

from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)

# RSS feeds gratuits sur le pétrole
RSS_FEEDS = {
    "reuters_commodities": (
        "https://feeds.reuters.com/reuters/businessNews"
    ),
    "yahoo_finance": (
        "https://finance.yahoo.com/rss/headline"
        "?s=CL%3DF&region=US&lang=en-US"
    ),
    "investing_oil": (
        "https://www.investing.com/rss/news_25.rss"
    ),
}

# Mots-clés pour filtrer les news pétrole
OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "opec", "brent", "wti",
    "energy", "barrel", "gasoline", "fuel", "refinery",
    "petrole", "petroleo",
]


@dataclass
class NewsItem:
    """A single news article."""

    title: str
    summary: str
    published: datetime
    source: str
    url: str


class NewsLoader:
    """Fetches oil-related news from RSS feeds.

    No API key required. Uses public RSS feeds.

    Args:
        feeds: Dictionary of {name: url} RSS feeds.
        keywords: Keywords to filter oil-related news.
        max_age_days: Only return news from last N days.
    """

    def __init__(
        self,
        feeds: dict[str, str] | None = None,
        keywords: list[str] | None = None,
        max_age_days: int = 30,
    ) -> None:
        self.feeds = feeds or RSS_FEEDS
        self.keywords = keywords or OIL_KEYWORDS
        self.max_age_days = max_age_days

    def fetch(self) -> list[NewsItem]:
        """Fetch and filter oil-related news from all feeds.

        Returns:
            List of NewsItem sorted by date (newest first).
        """
        all_news = []

        for source, url in self.feeds.items():
            try:
                items = self._fetch_feed(source, url)
                all_news.extend(items)
                logger.info(
                    f"[{source}] Fetched {len(items)} oil-related items."
                )
            except Exception as e:
                logger.warning(f"[{source}] Failed to fetch: {e}")

        # Sort by date, newest first
        all_news.sort(key=lambda x: x.published, reverse=True)
        logger.info(f"Total news fetched: {len(all_news)}")
        return all_news

    def _fetch_feed(self, source: str, url: str) -> list[NewsItem]:
        """Fetch a single RSS feed and filter oil-related items."""
        feed = feedparser.parse(url)
        items = []

        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text = (title + " " + summary).lower()

            # Filter by keywords
            if not any(kw in text for kw in self.keywords):
                continue

            # Parse date
            published = self._parse_date(entry)
            if published is None:
                continue

            items.append(NewsItem(
                title=title,
                summary=summary,
                published=published,
                source=source,
                url=getattr(entry, "link", ""),
            ))

        return items

    def _parse_date(self, entry) -> datetime | None:
        """Parse publication date from RSS entry."""
        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                return datetime(
                    *entry.published_parsed[:6],
                    tzinfo=timezone.utc,
                )
            if hasattr(entry, "updated_parsed") and entry.updated_parsed:
                return datetime(
                    *entry.updated_parsed[:6],
                    tzinfo=timezone.utc,
                )
        except Exception:
            pass
        return None