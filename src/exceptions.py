"""Custom exceptions for oil_bot."""


class OilBotError(Exception):
    """Base exception for all oil_bot errors."""


class DataNotAvailableError(OilBotError):
    """Market data cannot be fetched or is empty."""


class InsufficientDataError(OilBotError):
    """Not enough data to compute an indicator."""


class StrategyNotFoundError(OilBotError):
    """Requested strategy name is unknown."""
