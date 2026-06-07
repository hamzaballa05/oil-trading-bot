"""ML Pipeline — training, evaluation, and walk-forward validation."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from oil_bot.ml.feature_engineering import MlFeatureEngine
from oil_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MlTrainResult:
    """Result of training a model."""

    model: RandomForestRegressor
    feature_columns: list[str]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""

    fold_results: list[dict]
    overall_metrics: dict[str, float]
    all_predictions: pd.DataFrame
    all_actuals: pd.Series


class MlPipeline:
    """Handles model training and walk-forward evaluation.

    Args:
        n_estimators: Number of trees in Random Forest.
        max_depth: Maximum tree depth.
        random_state: Seed for reproducibility.
        horizon: Prediction horizon in days.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 10,
        random_state: int = 42,
        horizon: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.horizon = horizon
        self.feature_engine = MlFeatureEngine(horizon=horizon)

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Build features and target, clean infinities, drop NaN rows."""
        features = self.feature_engine.build_features(df)
        target = self.feature_engine.build_target(df)
        feature_cols = self.feature_engine.get_feature_columns(features)

        combined = features[feature_cols].copy()
        combined["target"] = target

        # Remplacer les infinis par NaN
        combined = combined.replace([np.inf, -np.inf], np.nan)

        # Supprimer les lignes avec NaN
        combined = combined.dropna()

        X = combined[feature_cols]
        y = combined["target"]

        logger.info(
            f"Prepared data: {len(X)} rows, {len(feature_cols)} features."
        )
        return X, y, feature_cols

    def train_test_split(
        self, X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Temporal split — NEVER random."""
        split_idx = int(len(X) * train_ratio)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(
            f"Split: train={len(X_train)} rows "
            f"({X_train.index[0].date()} -> {X_train.index[-1].date()}), "
            f"test={len(X_test)} rows "
            f"({X_test.index[0].date()} -> {X_test.index[-1].date()})"
        )
        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> MlTrainResult:
        """Train a RandomForestRegressor and evaluate on test set."""
        logger.info(
            f"Training RandomForest "
            f"(trees={self.n_estimators}, depth={self.max_depth})..."
        )

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = self._compute_metrics(y_train, train_pred)
        test_metrics = self._compute_metrics(y_test, test_pred)

        importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        predictions = pd.DataFrame(
            {
                "actual": y_test,
                "predicted": test_pred,
                "error": y_test.values - test_pred,
            },
            index=y_test.index,
        )

        logger.info(
            f"Train MAE: {train_metrics['mae']:.6f}, "
            f"Test MAE: {test_metrics['mae']:.6f}, "
            f"Test R2: {test_metrics['r2']:.4f}"
        )

        return MlTrainResult(
            model=model,
            feature_columns=list(X_train.columns),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            feature_importance=importance,
            predictions=predictions,
        )

    def walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        train_ratio: float = 0.6,
    ) -> WalkForwardResult:
        """Walk-forward validation."""
        total = len(X)
        test_size = int(total * (1 - train_ratio) / n_folds)

        fold_results = []
        all_preds = []
        all_actuals = []

        for fold in range(n_folds):
            train_end = int(total * train_ratio) + fold * test_size
            test_end = min(train_end + test_size, total)

            if train_end >= total or test_end <= train_end:
                break

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_te = X.iloc[train_end:test_end]
            y_te = y.iloc[train_end:test_end]

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            metrics = self._compute_metrics(y_te, preds)
            metrics["fold"] = fold + 1
            metrics["train_size"] = len(X_tr)
            metrics["test_size"] = len(X_te)
            metrics["test_start"] = str(X_te.index[0].date())
            metrics["test_end"] = str(X_te.index[-1].date())
            fold_results.append(metrics)

            all_preds.extend(preds)
            all_actuals.extend(y_te.values)

            logger.info(
                f"Fold {fold + 1}: "
                f"MAE={metrics['mae']:.6f}, "
                f"R2={metrics['r2']:.4f}, "
                f"Direction={metrics['direction_accuracy']:.1%}"
            )

        overall = self._compute_metrics(
            pd.Series(all_actuals), np.array(all_preds)
        )

        all_pred_df = pd.DataFrame(
            {"actual": all_actuals, "predicted": all_preds}
        )

        logger.info(
            f"Walk-forward overall: "
            f"MAE={overall['mae']:.6f}, "
            f"R2={overall['r2']:.4f}, "
            f"Direction={overall['direction_accuracy']:.1%}"
        )

        return WalkForwardResult(
            fold_results=fold_results,
            overall_metrics=overall,
            all_predictions=all_pred_df,
            all_actuals=pd.Series(all_actuals),
        )

    def _compute_metrics(
        self, actual: pd.Series, predicted: np.ndarray
    ) -> dict[str, float]:
        """Compute regression and trading-relevant metrics."""
        actual_arr = np.array(actual)
        pred_arr = np.array(predicted)

        mae = mean_absolute_error(actual_arr, pred_arr)
        rmse = float(np.sqrt(mean_squared_error(actual_arr, pred_arr)))
        r2 = r2_score(actual_arr, pred_arr) if len(actual_arr) > 1 else 0.0

        actual_sign = np.sign(actual_arr)
        pred_sign = np.sign(pred_arr)
        direction_accuracy = float(np.mean(actual_sign == pred_sign))

        return {
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "direction_accuracy": direction_accuracy,
            "mean_predicted": float(pred_arr.mean()),
            "std_predicted": float(pred_arr.std()),
        }