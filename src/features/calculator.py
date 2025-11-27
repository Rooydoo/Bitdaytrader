"""Feature calculation module for LightGBM model."""

import numpy as np
import pandas as pd
from loguru import logger


class FeatureCalculator:
    """Calculate 12 technical features for prediction."""

    FEATURE_NAMES = [
        "return_1",
        "return_5",
        "return_15",
        "volatility_20",
        "atr_14",
        "rsi_14",
        "macd_diff",
        "ema_ratio",
        "bb_position",
        "volume_ratio",
        "hour",
        "day_of_week",
    ]

    def __init__(self) -> None:
        """Initialize feature calculator."""
        self.lookback_periods = 50  # Minimum periods needed for calculation

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 12 features from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with feature columns added
        """
        if len(df) < self.lookback_periods:
            logger.warning(
                f"Not enough data for feature calculation. Need {self.lookback_periods}, got {len(df)}"
            )
            return pd.DataFrame()

        result = df.copy()

        # Price returns
        result["return_1"] = result["close"].pct_change(1)
        result["return_5"] = result["close"].pct_change(5)
        result["return_15"] = result["close"].pct_change(15)

        # Volatility (20-period rolling std of returns)
        result["volatility_20"] = result["return_1"].rolling(20).std()

        # ATR (Average True Range)
        result["atr_14"] = self._calculate_atr(result, period=14)

        # RSI
        result["rsi_14"] = self._calculate_rsi(result["close"], period=14)

        # MACD
        result["macd_diff"] = self._calculate_macd_diff(result["close"])

        # EMA ratio (short/long)
        ema_short = result["close"].ewm(span=9, adjust=False).mean()
        ema_long = result["close"].ewm(span=21, adjust=False).mean()
        result["ema_ratio"] = ema_short / ema_long

        # Bollinger Band position
        result["bb_position"] = self._calculate_bb_position(result["close"])

        # Volume ratio (current / 20-period average)
        vol_ma = result["volume"].rolling(20).mean()
        result["volume_ratio"] = result["volume"] / vol_ma

        # Time features
        result["hour"] = result["timestamp"].dt.hour
        result["day_of_week"] = result["timestamp"].dt.dayofweek

        # Drop NaN rows
        result = result.dropna()

        return result

    def get_latest_features(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Get the latest feature vector for prediction.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            1D numpy array of features or None if insufficient data
        """
        result = self.calculate(df)

        if result.empty:
            return None

        features = result[self.FEATURE_NAMES].iloc[-1].values
        return features.astype(np.float32)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return atr

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_diff(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.Series:
        """Calculate MACD - Signal line difference."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()

        return macd - macd_signal

    def _calculate_bb_position(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.Series:
        """
        Calculate Bollinger Band position.
        Returns value between -1 (lower band) and 1 (upper band).
        """
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper = sma + std_dev * std
        lower = sma - std_dev * std

        # Position: -1 at lower band, 0 at middle, 1 at upper band
        bb_width = upper - lower
        position = (prices - lower) / bb_width.replace(0, np.nan) * 2 - 1

        return position.clip(-1, 1)

    def create_label(
        self,
        df: pd.DataFrame,
        horizon: int = 4,
        threshold: float = 0.003,
    ) -> pd.Series:
        """
        Create binary label for training.

        Args:
            df: DataFrame with close prices
            horizon: Prediction horizon (number of periods)
            threshold: Price change threshold (0.3% = 0.003)

        Returns:
            Series with binary labels (1 if price rises by threshold, else 0)
        """
        future_return = df["close"].shift(-horizon) / df["close"] - 1
        label = (future_return >= threshold).astype(int)

        return label
