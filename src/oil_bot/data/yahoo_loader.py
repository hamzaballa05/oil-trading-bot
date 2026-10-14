"""Yahoo Finance data loader with local cache."""

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from oil_bot.data.interfaces import IDataLoader
from oil_bot.data.validators import validate_ohlcv
from oil_bot.exceptions import DataNotAvailableError
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class YahooFinanceLoader(IDataLoader):
    """Loads OHLCV data from Yahoo Finance, with Parquet caching."""

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self, symbol: str, start: date, end: date, timeframe: str = "1d"
    ) -> pd.DataFrame:
        safe = symbol.replace("=", "_").replace("/", "_")
        cache_path = self.cache_dir / f"{safe}_{timeframe}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            s = pd.Timestamp(start, tz="UTC")
            e = pd.Timestamp(end, tz="UTC")
            if df.index[0] <= s and df.index[-1] >= e:
                logger.info(f"[{symbol}] Loaded from cache.")
                return df.loc[s:e].copy()

        logger.info(f"[{symbol}] Downloading from Yahoo Finance...")
        try:
            df = yf.Ticker(symbol).history(
                start=str(start),
                end=str(end),
                interval=timeframe,
                auto_adjust=True,
            )
        except Exception as exc:
            raise DataNotAvailableError(f"Yahoo download failed: {exc}") from exc

        if df.empty:
            raise DataNotAvailableError(f"No data returned for {symbol}.")

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = validate_ohlcv(df, symbol)
        df.to_parquet(cache_path)
        logger.info(f"[{symbol}] Downloaded and cached {len(df)} bars.")
        return df