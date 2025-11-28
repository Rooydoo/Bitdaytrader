"""Automatic model retraining with overfitting-aware parameter adjustment."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.database.models import WalkForwardRepository
from training.trainer import ModelTrainer


class AutoRetrainer:
    """
    Automatic model retrainer that adjusts parameters based on overfitting severity.

    This class orchestrates:
    1. Detection of when retraining is needed
    2. Parameter adjustment based on overfitting severity
    3. Walk-forward validation of new model
    4. Model deployment if improved
    """

    def __init__(
        self,
        walkforward_repo: WalkForwardRepository,
        data_path: str,
        model_output_path: str = "models/lightgbm_model.joblib",
        training_months: int = 12,
        backtest_months: int = 2,
    ) -> None:
        """
        Initialize auto retrainer.

        Args:
            walkforward_repo: Repository for walk-forward results
            data_path: Path to OHLCV data CSV
            model_output_path: Path to save trained model
            training_months: Months of data for training
            backtest_months: Months of data for backtesting
        """
        self.walkforward_repo = walkforward_repo
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.training_months = training_months
        self.backtest_months = backtest_months

    def check_retraining_needed(self) -> dict[str, Any]:
        """
        Check if model retraining is needed.

        Returns:
            Dict with 'needed' bool and 'reason' string
        """
        latest = self.walkforward_repo.get_latest()

        if not latest:
            return {
                "needed": True,
                "reason": "No model available",
                "severity": "none",
            }

        # Check if model is overfit
        if latest.is_overfit:
            severity = self._determine_severity(latest.accuracy_gap or 0)
            return {
                "needed": True,
                "reason": f"Model is overfit (gap: {latest.accuracy_gap:.1%})",
                "severity": severity,
            }

        # Check if live accuracy dropped significantly
        if (
            latest.live_predictions
            and latest.live_predictions >= 30
            and latest.live_accuracy is not None
            and latest.test_accuracy_mean is not None
        ):
            live_drop = latest.test_accuracy_mean - latest.live_accuracy
            if live_drop > 0.10:
                return {
                    "needed": True,
                    "reason": f"Live accuracy dropped {live_drop:.1%} below test",
                    "severity": "moderate",
                }

        # Check age of model (retrain every 2 weeks regardless)
        if latest.trained_at:
            age_days = (datetime.utcnow() - latest.trained_at).days
            if age_days >= 14:
                return {
                    "needed": True,
                    "reason": f"Model is {age_days} days old (scheduled retrain)",
                    "severity": "none",
                }

        return {
            "needed": False,
            "reason": "Model performance is acceptable",
            "severity": "none",
        }

    def _determine_severity(self, accuracy_gap: float) -> str:
        """Determine overfitting severity from accuracy gap."""
        if accuracy_gap > 0.15:
            return "severe"
        elif accuracy_gap > 0.10:
            return "moderate"
        elif accuracy_gap > 0.05:
            return "mild"
        return "none"

    def retrain_model(
        self,
        severity: str = "none",
        notify_callback: Any | None = None,
    ) -> dict[str, Any]:
        """
        Retrain model with severity-appropriate parameters.

        Args:
            severity: Overfitting severity level
            notify_callback: Optional async callback for notifications

        Returns:
            Dict with training results
        """
        logger.info(f"Starting model retraining with severity={severity}")

        try:
            # Load data
            df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
            logger.info(f"Loaded {len(df)} rows from {self.data_path}")

            # Create trainer
            trainer = ModelTrainer(
                training_months=self.training_months,
                backtest_months=self.backtest_months,
            )

            # Get parameters based on severity
            if severity != "none":
                params = trainer.get_anti_overfit_params(severity)
                logger.info(f"Using anti-overfit params for severity={severity}")
            else:
                params = trainer._get_default_params()

            # Run walk-forward training
            logger.info("Starting walk-forward training...")
            model, metrics = trainer.walk_forward_train(df)

            if model is None:
                raise RuntimeError("Training failed - no model produced")

            # Run backtest
            logger.info("Running backtest...")
            backtest_results = trainer.backtest(df.tail(10000), model)
            logger.info(f"Backtest results: {backtest_results}")

            # Generate model version
            model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")

            # Save model
            trainer.save_model(
                model,
                self.model_output_path,
                metrics={
                    "walkforward": metrics,
                    "backtest": backtest_results,
                    "version": model_version,
                },
            )

            # Save to walk-forward repository
            # Calculate train metrics for comparison
            train_features, train_labels = trainer.prepare_data(df)
            train_pred = model.predict(train_features)
            train_accuracy = (train_pred == train_labels).mean()

            train_metrics = {
                "accuracy": train_accuracy,
                "auc": metrics.get("auc_mean", 0),
            }

            self.walkforward_repo.save_result(
                model_version=model_version,
                walkforward_metrics=metrics,
                backtest_results=backtest_results,
                train_metrics=train_metrics,
            )

            result = {
                "success": True,
                "model_version": model_version,
                "metrics": metrics,
                "backtest": backtest_results,
                "params_used": params,
            }

            logger.info(f"Retraining completed: {model_version}")
            return result

        except Exception as e:
            logger.exception(f"Retraining failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_recommended_params(self) -> dict[str, Any]:
        """
        Get recommended training parameters based on current model state.

        Returns:
            Dict with recommended params and reason
        """
        check = self.check_retraining_needed()
        trainer = ModelTrainer()

        if check["severity"] == "severe":
            return {
                "params": trainer.get_anti_overfit_params("severe"),
                "reason": "Severe overfitting detected - maximum regularization",
            }
        elif check["severity"] == "moderate":
            return {
                "params": trainer.get_anti_overfit_params("moderate"),
                "reason": "Moderate overfitting detected - increased regularization",
            }
        elif check["severity"] == "mild":
            return {
                "params": trainer.get_anti_overfit_params("mild"),
                "reason": "Mild overfitting detected - slight regularization increase",
            }
        else:
            return {
                "params": trainer._get_default_params(),
                "reason": "No overfitting - using default parameters",
            }


async def run_auto_retrain(
    walkforward_repo: WalkForwardRepository,
    data_path: str,
    model_output_path: str,
    telegram_bot: Any | None = None,
) -> dict[str, Any]:
    """
    Run automatic retraining process.

    Args:
        walkforward_repo: Repository for walk-forward results
        data_path: Path to OHLCV data
        model_output_path: Path to save model
        telegram_bot: Optional TelegramBot for notifications

    Returns:
        Dict with retraining results
    """
    retrainer = AutoRetrainer(
        walkforward_repo=walkforward_repo,
        data_path=data_path,
        model_output_path=model_output_path,
    )

    # Check if retraining is needed
    check = retrainer.check_retraining_needed()

    if not check["needed"]:
        logger.info(f"Retraining not needed: {check['reason']}")
        return {"retrained": False, "reason": check["reason"]}

    logger.info(f"Retraining needed: {check['reason']}")

    # Notify start
    if telegram_bot:
        params = retrainer.get_recommended_params()
        await telegram_bot.notify_retraining_triggered(
            reason=check["reason"],
            params_adjusted=params.get("params"),
        )

    # Run retraining
    result = retrainer.retrain_model(severity=check["severity"])

    # Notify completion
    if telegram_bot and result.get("success"):
        old_result = walkforward_repo.get_latest()
        old_accuracy = old_result.test_accuracy_mean if old_result else None

        await telegram_bot.send_retraining_notification(
            reason=check["reason"],
            old_accuracy=old_accuracy,
            new_accuracy=result.get("metrics", {}).get("accuracy_mean"),
            improvement=(
                result.get("metrics", {}).get("accuracy_mean", 0) - (old_accuracy or 0)
                if old_accuracy
                else None
            ),
        )

    return {
        "retrained": True,
        "reason": check["reason"],
        "result": result,
    }
