"""CSV data loader for offline testing."""

from datetime import date
from pathlib import Path

import pandas as pd

from oil_bot.data.interfaces import IDataLoader
from oil_bot.data.validators import validate_ohlcv
from oil_bot.exceptions import DataNotAvailableError


class CsvLoader(IDataLoader):
    """Loads OHLCV data from a local CSV file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(
        self, symbol: str, start: date, end: date, timeframe: str = "1d"
    ) -> pd.DataFrame:
        if not self.file_path.exists():
            raise DataNotAvailableError(f"CSV not found: {self.file_path}")

        df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        df = df.loc[s:e]
        return validate_ohlcv(df, symbol)
