"""OHLCV data validation."""

import pandas as pd

from oil_bot.exceptions import DataNotAvailableError
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED = {"open", "high", "low", "close", "volume"}


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Validate and clean an OHLCV DataFrame."""
    if df.empty:
        raise DataNotAvailableError(f"[{symbol}] Empty DataFrame.")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    missing = REQUIRED - set(df.columns)
    if missing:
        raise DataNotAvailableError(f"[{symbol}] Missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataNotAvailableError(f"[{symbol}] Index must be DatetimeIndex.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    bad = df["high"] < df["low"]
    if bad.any():
        logger.warning(f"[{symbol}] {bad.sum()} bars with high<low, swapping.")
        df.loc[bad, "high"], df.loc[bad, "low"] = (
            df.loc[bad, "low"].values,
            df.loc[bad, "high"].values,
        )

    df = df[list(REQUIRED)]
    logger.info(f"[{symbol}] Validated {len(df)} bars.")
    return df