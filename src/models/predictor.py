"""LightGBM prediction model for price direction."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger


class Predictor:
    """LightGBM-based price direction predictor."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model file (.joblib)
        """
        self.model: Any = None
        self.model_path = Path(model_path) if model_path else None
        self.feature_names: list[str] = []  # Feature names model was trained with

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    def load(self, model_path: str | Path) -> None:
        """Load trained model from file."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded = joblib.load(model_path)

        # Handle both dict format (from trainer) and raw model format
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.feature_names = loaded.get("feature_names", [])
            logger.info(
                f"Model loaded from {model_path} (dict format, {len(self.feature_names)} features)"
            )
        else:
            self.model = loaded
            self.feature_names = []
            logger.info(f"Model loaded from {model_path}")

        self.model_path = model_path

    def save(self, model_path: str | Path) -> None:
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        self.model_path = model_path
        logger.info(f"Model saved to {model_path}")

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        Predict price direction.

        Args:
            features: Feature array (1D or 2D)

        Returns:
            Tuple of (prediction, confidence)
            - prediction: 1 for up, 0 for down/flat
            - confidence: Probability of the prediction
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get probability predictions
        proba = self.model.predict_proba(features)[0]

        # Get prediction and confidence
        prediction = int(proba[1] >= 0.5)
        confidence = proba[1] if prediction == 1 else proba[0]

        return prediction, float(confidence)

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Get probability of price going up.

        Args:
            features: Feature array

        Returns:
            Probability of positive class (price up)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.model.predict_proba(features)[0]
        return float(proba[1])

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def reload_model(self, model_path: str | Path | None = None) -> None:
        """
        Reload model from file (useful after retraining).

        Args:
            model_path: Path to model file, or use current path if None
        """
        path = Path(model_path) if model_path else self.model_path

        if path is None:
            raise RuntimeError("No model path specified")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Clear current model
        self.model = None
        self.feature_names = []

        # Load new model
        self.load(path)
        logger.info(f"Model reloaded from {path}")

    def get_required_features(self) -> list[str]:
        """Get the list of feature names this model expects."""
        return self.feature_names.copy()

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        """Get default LightGBM parameters optimized for this task."""
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
