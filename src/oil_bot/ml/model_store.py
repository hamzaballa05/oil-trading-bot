"""Model persistence — save and load trained models."""

from pathlib import Path

import joblib

from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


class ModelStore:
    """Saves and loads trained models using joblib."""

    def __init__(self, store_dir: str | Path = "models") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model,
        feature_columns: list[str],
        name: str = "rf_model",
    ) -> Path:
        """Save a model and its feature list."""
        payload = {
            "model": model,
            "feature_columns": feature_columns,
        }
        path = self.store_dir / f"{name}.joblib"
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, name: str = "rf_model") -> tuple:
        """Load a model and its feature list."""
        path = self.store_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        payload = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return payload["model"], payload["feature_columns"]