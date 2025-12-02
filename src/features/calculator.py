"""Feature calculation module for LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.features.registry import FeatureRegistry


class FeatureCalculator:
    """Calculate technical features for prediction with dynamic feature support."""

    # Core features (always calculated for backward compatibility)
    CORE_FEATURE_NAMES = [
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

    # For backward compatibility
    FEATURE_NAMES = CORE_FEATURE_NAMES

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        """
        Initialize feature calculator.

        Args:
            registry: Optional FeatureRegistry for dynamic feature management.
                      If None, only core features are calculated.
        """
        self.registry = registry
        self.lookback_periods = 50  # Minimum periods needed for calculation

    @property
    def active_feature_names(self) -> list[str]:
        """Get list of currently active feature names."""
        if self.registry is None:
            return self.CORE_FEATURE_NAMES.copy()
        return self.registry.get_enabled_features()

    def calculate(self, df: pd.DataFrame, include_extended: bool = True) -> pd.DataFrame:
        """
        Calculate features from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            include_extended: If True and registry is set, also calculate enabled extended features

        Returns:
            DataFrame with feature columns added
        """
        if len(df) < self.lookback_periods:
            logger.warning(
                f"Not enough data for feature calculation. Need {self.lookback_periods}, got {len(df)}"
            )
            return pd.DataFrame()

        result = df.copy()

        # === Core Features (always calculated) ===
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

        # === Extended Features (if registry is set and enabled) ===
        if include_extended and self.registry is not None:
            result = self._calculate_extended_features(result)

        # Drop NaN rows
        result = result.dropna()

        return result

    def _calculate_extended_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate extended features based on registry settings."""
        result = df.copy()
        enabled = self.registry.get_enabled_features() if self.registry else []

        # Trend features
        if "adx_14" in enabled:
            result["adx_14"] = self._calculate_adx(result, period=14)

        if "trend_strength" in enabled:
            result["trend_strength"] = self._calculate_trend_strength(result)

        # Volatility extensions
        if "realized_vol_5" in enabled:
            result["realized_vol_5"] = result["return_1"].rolling(5).std() * np.sqrt(5)

        if "vol_regime" in enabled:
            vol_20 = result["return_1"].rolling(20).std()
            vol_50 = result["return_1"].rolling(50).std()
            result["vol_regime"] = (vol_20 / vol_50.replace(0, np.nan)).fillna(1.0)

        # Momentum extensions
        if "momentum_divergence" in enabled:
            result["momentum_divergence"] = self._calculate_momentum_divergence(result)

        if "price_acceleration" in enabled:
            result["price_acceleration"] = result["return_1"].diff()

        if "roc_5" in enabled:
            result["roc_5"] = (result["close"] / result["close"].shift(5) - 1) * 100

        # Volume extensions
        if "volume_momentum" in enabled:
            result["volume_momentum"] = result["volume"].pct_change(5)

        if "obv_slope" in enabled:
            result["obv_slope"] = self._calculate_obv_slope(result)

        # Time extensions
        if "hour_sin" in enabled:
            result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)

        if "hour_cos" in enabled:
            result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

        if "is_tokyo_session" in enabled:
            # Tokyo session: 9:00-15:00 JST (0:00-6:00 UTC)
            result["is_tokyo_session"] = ((result["hour"] >= 0) & (result["hour"] < 6)).astype(int)

        if "is_london_session" in enabled:
            # London session: 8:00-16:00 UTC
            result["is_london_session"] = ((result["hour"] >= 8) & (result["hour"] < 16)).astype(int)

        if "is_ny_session" in enabled:
            # NY session: 13:00-21:00 UTC
            result["is_ny_session"] = ((result["hour"] >= 13) & (result["hour"] < 21)).astype(int)

        # Market regime
        if "market_regime" in enabled:
            result["market_regime"] = self._calculate_market_regime(result)

        return result

    def get_latest_features(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> np.ndarray | None:
        """
        Get the latest feature vector for prediction.

        Args:
            df: DataFrame with OHLCV data
            feature_names: Specific feature names to extract. If None, uses active features.

        Returns:
            1D numpy array of features or None if insufficient data
        """
        result = self.calculate(df)

        if result.empty:
            return None

        # Use provided feature names or active features
        names = feature_names if feature_names else self.active_feature_names

        # Filter to only existing columns
        available_names = [n for n in names if n in result.columns]
        if len(available_names) != len(names):
            missing = set(names) - set(available_names)
            logger.warning(f"Missing features in data: {missing}")

        features = result[available_names].iloc[-1].values
        return features.astype(np.float32)

    def get_features_for_model(
        self,
        df: pd.DataFrame,
        model_feature_names: list[str],
    ) -> np.ndarray | None:
        """
        Get feature vector matching the model's expected feature order.

        Args:
            df: DataFrame with OHLCV data
            model_feature_names: Feature names the model was trained with

        Returns:
            1D numpy array of features in model's expected order
        """
        return self.get_latest_features(df, feature_names=model_feature_names)

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

    # === Extended Feature Calculation Methods ===

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        Measures trend strength (0-100, >25 indicates strong trend).
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength indicator.
        Combines ADX with EMA alignment for comprehensive trend measurement.
        Returns normalized value 0-1.
        """
        # EMA alignment score
        ema_9 = df["close"].ewm(span=9, adjust=False).mean()
        ema_21 = df["close"].ewm(span=21, adjust=False).mean()
        ema_50 = df["close"].ewm(span=50, adjust=False).mean()

        # Count aligned EMAs (bullish: 9 > 21 > 50 or bearish: 9 < 21 < 50)
        bullish_aligned = ((ema_9 > ema_21) & (ema_21 > ema_50)).astype(float)
        bearish_aligned = ((ema_9 < ema_21) & (ema_21 < ema_50)).astype(float)
        alignment_score = bullish_aligned + bearish_aligned

        # Combine with ADX (normalized)
        adx = self._calculate_adx(df, period=14)
        adx_normalized = (adx / 100).clip(0, 1)

        # Combined trend strength
        trend_strength = (alignment_score * 0.5 + adx_normalized * 0.5)

        return trend_strength

    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum divergence.
        Detects divergence between price and RSI.
        Positive: bullish divergence, Negative: bearish divergence.
        """
        close = df["close"]
        rsi = self._calculate_rsi(close, period=14)

        # 20-period price and RSI slopes
        period = 20
        price_change = close.pct_change(period)
        rsi_change = rsi.diff(period)

        # Divergence: price down but RSI up (bullish) or price up but RSI down (bearish)
        # Normalized to -1 to 1
        divergence = (rsi_change / 100) - price_change
        divergence = divergence.clip(-1, 1)

        return divergence

    def _calculate_obv_slope(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV) slope.
        Measures volume flow trend.
        """
        close = df["close"]
        volume = df["volume"]

        # OBV calculation
        obv_direction = np.sign(close.diff()).fillna(0)
        obv = (volume * obv_direction).cumsum()

        # Normalize OBV to make slope comparable across different volume scales
        obv_normalized = (obv - obv.rolling(50).mean()) / obv.rolling(50).std().replace(0, 1)

        # Calculate slope (rate of change)
        obv_slope = obv_normalized.diff(period) / period

        return obv_slope.clip(-2, 2)

    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market regime indicator.
        0: Range-bound, 1: Trending
        Based on ADX and volatility clustering.
        """
        # ADX-based trend detection
        adx = self._calculate_adx(df, period=14)
        is_trending = (adx > 25).astype(float)

        # Volatility regime (high vol often means trending)
        returns = df["close"].pct_change()
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(50).std()
        vol_ratio = (vol_short / vol_long.replace(0, np.nan)).fillna(1)
        high_vol = (vol_ratio > 1.2).astype(float)

        # Combined regime score
        regime = (is_trending * 0.6 + high_vol * 0.4)

        return regime
