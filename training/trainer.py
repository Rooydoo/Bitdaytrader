"""Model training with walk-forward validation."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from src.features.calculator import FeatureCalculator


class ModelTrainer:
    """Train LightGBM model with walk-forward validation."""

    def __init__(
        self,
        training_months: int = 12,
        backtest_months: int = 2,
        prediction_horizon: int = 4,
        price_threshold: float = 0.003,
    ) -> None:
        """
        Initialize trainer.

        Args:
            training_months: Months of data for training
            backtest_months: Months of data for backtesting
            prediction_horizon: Prediction horizon in periods
            price_threshold: Price change threshold for positive label (0.3% = 0.003)
        """
        self.training_months = training_months
        self.backtest_months = backtest_months
        self.prediction_horizon = prediction_horizon
        self.price_threshold = price_threshold
        self.feature_calculator = FeatureCalculator()

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Calculate features
        df_features = self.feature_calculator.calculate(df)

        if df_features.empty:
            raise ValueError("Not enough data for feature calculation")

        # Create labels
        labels = self.feature_calculator.create_label(
            df_features,
            horizon=self.prediction_horizon,
            threshold=self.price_threshold,
        )

        # Align features and labels (drop last rows without labels)
        valid_idx = labels.notna()
        features = df_features[FeatureCalculator.FEATURE_NAMES][valid_idx]
        labels = labels[valid_idx]

        return features, labels

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> lgb.LGBMClassifier:
        """
        Train LightGBM model.

        Args:
            features: Feature DataFrame
            labels: Label Series
            params: LightGBM parameters (optional)

        Returns:
            Trained LightGBM model
        """
        if params is None:
            params = self._get_default_params()

        model = lgb.LGBMClassifier(**params)
        model.fit(features, labels)

        return model

    def walk_forward_train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
        """
        Train model using walk-forward validation.

        Args:
            df: Full OHLCV DataFrame
            n_splits: Number of walk-forward splits

        Returns:
            Tuple of (best model, performance metrics)
        """
        # Calculate total periods needed
        periods_per_month = 4 * 24 * 30  # 15min candles per month (approx)
        train_periods = self.training_months * periods_per_month
        test_periods = self.backtest_months * periods_per_month

        # Prepare features and labels
        features, labels = self.prepare_data(df)

        total_periods = len(features)
        if total_periods < train_periods + test_periods:
            raise ValueError(
                f"Not enough data: need {train_periods + test_periods}, got {total_periods}"
            )

        # Walk-forward splits
        step_size = (total_periods - train_periods - test_periods) // n_splits

        results = []
        best_model = None
        best_auc = 0.0

        for i in range(n_splits):
            start_idx = i * step_size
            train_end = start_idx + train_periods
            test_end = train_end + test_periods

            if test_end > total_periods:
                break

            X_train = features.iloc[start_idx:train_end]
            y_train = labels.iloc[start_idx:train_end]
            X_test = features.iloc[train_end:test_end]
            y_test = labels.iloc[train_end:test_end]

            logger.info(
                f"Walk-forward split {i + 1}/{n_splits}: "
                f"train={len(X_train)}, test={len(X_test)}"
            )

            # Train model
            model = self.train(X_train, y_train)

            # Evaluate
            metrics = self._evaluate(model, X_test, y_test)
            metrics["split"] = i + 1
            results.append(metrics)

            logger.info(
                f"Split {i + 1} results: AUC={metrics['auc']:.4f}, "
                f"Accuracy={metrics['accuracy']:.4f}"
            )

            # Track best model
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_model = model

        # Aggregate results
        agg_metrics = self._aggregate_results(results)
        logger.info(f"Average metrics: {agg_metrics}")

        return best_model, agg_metrics

    def _evaluate(
        self,
        model: lgb.LGBMClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
        }

    def _aggregate_results(self, results: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate walk-forward results."""
        if not results:
            return {}

        agg = {}
        keys = ["accuracy", "precision", "recall", "auc"]
        for key in keys:
            values = [r[key] for r in results]
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)

        return agg

    def _get_default_params(self) -> dict[str, Any]:
        """Get default LightGBM parameters."""
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }

    def save_model(
        self,
        model: lgb.LGBMClassifier,
        path: str | Path,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Save trained model.

        Args:
            model: Trained model
            path: Output path
            metrics: Optional metrics to save with model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": model,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "feature_names": FeatureCalculator.FEATURE_NAMES,
        }

        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")

    def backtest(
        self,
        df: pd.DataFrame,
        model: lgb.LGBMClassifier,
        confidence_threshold: float = 0.65,
        initial_capital: float = 1_000_000,
    ) -> dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame
            model: Trained model
            confidence_threshold: Minimum confidence for trading
            initial_capital: Starting capital

        Returns:
            Backtest results
        """
        features, labels = self.prepare_data(df)

        # Get predictions
        probas = model.predict_proba(features)[:, 1]

        # Simulate trades
        capital = initial_capital
        trades = []
        position = None

        for i in range(len(features) - self.prediction_horizon):
            proba = probas[i]
            actual = labels.iloc[i + self.prediction_horizon] if i + self.prediction_horizon < len(labels) else None

            # Entry signal
            if position is None and proba >= confidence_threshold:
                position = {
                    "entry_idx": i,
                    "direction": 1,  # Long
                    "confidence": proba,
                }

            # Exit after prediction_horizon periods
            elif position is not None:
                if i >= position["entry_idx"] + self.prediction_horizon:
                    entry_price = df["close"].iloc[position["entry_idx"]]
                    exit_price = df["close"].iloc[i]

                    pnl_pct = (exit_price - entry_price) / entry_price
                    pnl = capital * 0.02 * pnl_pct  # 2% risk position sizing

                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "confidence": position["confidence"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "direction": "correct" if (pnl_pct > 0) == (position["direction"] == 1) else "wrong",
                    })

                    capital += pnl
                    position = None

        # Calculate results
        if not trades:
            return {"error": "No trades executed"}

        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        returns = [t["pnl"] / initial_capital for t in trades]

        results = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades),
            "total_pnl": total_pnl,
            "return_pct": total_pnl / initial_capital,
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "max_drawdown": self._calculate_max_drawdown([t["pnl"] for t in trades], initial_capital),
            "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            "final_capital": capital,
        }

        return results

    def _calculate_max_drawdown(
        self,
        pnls: list[float],
        initial_capital: float,
    ) -> float:
        """Calculate maximum drawdown from PnL series."""
        capital = initial_capital
        peak = capital
        max_dd = 0.0

        for pnl in pnls:
            capital += pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)

        return max_dd


def train_and_save(
    data_path: str,
    output_path: str = "models/lightgbm_model.joblib",
    training_months: int = 12,
    backtest_months: int = 2,
) -> None:
    """
    Train model and save to file.

    Args:
        data_path: Path to OHLCV CSV file
        output_path: Output model path
        training_months: Training data months
        backtest_months: Backtest data months
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    trainer = ModelTrainer(
        training_months=training_months,
        backtest_months=backtest_months,
    )

    logger.info("Starting walk-forward training...")
    model, metrics = trainer.walk_forward_train(df)

    if model is None:
        raise RuntimeError("Training failed - no model produced")

    logger.info("Running backtest...")
    backtest_results = trainer.backtest(df.tail(10000), model)
    logger.info(f"Backtest results: {backtest_results}")

    trainer.save_model(model, output_path, metrics={"walkforward": metrics, "backtest": backtest_results})
    logger.info(f"Model saved to {output_path}")
