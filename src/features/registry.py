"""Feature registry for dynamic feature management."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.timezone import now_jst


@dataclass
class FeatureConfig:
    """Configuration for a single feature."""

    name: str
    category: str
    enabled: bool = True
    importance_score: float = 0.0
    description: str = ""
    parameters: dict = field(default_factory=dict)
    last_updated: datetime = field(default_factory=now_jst)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "enabled": self.enabled,
            "importance_score": self.importance_score,
            "description": self.description,
            "parameters": self.parameters,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        return cls(
            name=data["name"],
            category=data["category"],
            enabled=data.get("enabled", True),
            importance_score=data.get("importance_score", 0.0),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else now_jst(),
        )


class FeatureRegistry:
    """
    Registry for managing features dynamically.
    Allows enabling/disabling features and tracking importance.
    """

    # Core features (always available)
    CORE_FEATURES = {
        "return_1": {"category": "return", "description": "1期間リターン"},
        "return_5": {"category": "return", "description": "5期間リターン"},
        "return_15": {"category": "return", "description": "15期間リターン"},
        "volatility_20": {"category": "volatility", "description": "20期間ボラティリティ"},
        "atr_14": {"category": "volatility", "description": "14期間ATR"},
        "rsi_14": {"category": "momentum", "description": "14期間RSI"},
        "macd_diff": {"category": "momentum", "description": "MACD差分"},
        "ema_ratio": {"category": "trend", "description": "EMA比率 (9/21)"},
        "bb_position": {"category": "trend", "description": "ボリンジャーバンド位置"},
        "volume_ratio": {"category": "volume", "description": "出来高比率"},
        "hour": {"category": "time", "description": "時間 (0-23)"},
        "day_of_week": {"category": "time", "description": "曜日 (0-6)"},
    }

    # Extended features (can be enabled/disabled)
    EXTENDED_FEATURES = {
        # Trend strength
        "adx_14": {"category": "trend", "description": "14期間ADX（トレンド強度）"},
        "trend_strength": {"category": "trend", "description": "トレンド強度指標"},

        # Volatility extensions
        "realized_vol_5": {"category": "volatility", "description": "5期間実現ボラティリティ"},
        "vol_regime": {"category": "volatility", "description": "ボラティリティレジーム"},

        # Momentum extensions
        "momentum_divergence": {"category": "momentum", "description": "モメンタム乖離"},
        "price_acceleration": {"category": "momentum", "description": "価格加速度"},
        "roc_5": {"category": "momentum", "description": "5期間ROC"},

        # Volume extensions
        "volume_momentum": {"category": "volume", "description": "出来高モメンタム"},
        "obv_slope": {"category": "volume", "description": "OBV傾き"},

        # Time extensions
        "hour_sin": {"category": "time", "description": "時間（サイン変換）"},
        "hour_cos": {"category": "time", "description": "時間（コサイン変換）"},
        "is_tokyo_session": {"category": "time", "description": "東京セッション"},
        "is_london_session": {"category": "time", "description": "ロンドンセッション"},
        "is_ny_session": {"category": "time", "description": "NYセッション"},

        # Market regime
        "market_regime": {"category": "regime", "description": "市場レジーム（トレンド/レンジ）"},

        # Sentiment (if available)
        "funding_rate": {"category": "sentiment", "description": "ファンディングレート"},

        # Correlation
        "btc_eth_correlation_20": {"category": "correlation", "description": "BTC-ETH相関"},

        # Orderbook features (real-time only)
        "bid_ask_spread": {"category": "orderbook", "description": "スプレッド（%）"},
        "orderbook_imbalance": {"category": "orderbook", "description": "板の偏り（-1〜1）"},
    }

    def __init__(self, config_path: str = "data/feature_registry.json") -> None:
        """
        Initialize feature registry.

        Args:
            config_path: Path to feature configuration file
        """
        self.config_path = Path(config_path)
        self.features: dict[str, FeatureConfig] = {}

        self._init_features()
        self._load_config()

        logger.info(f"Feature registry initialized: {len(self.features)} features")

    def _init_features(self) -> None:
        """Initialize features from definitions."""
        # Add core features (always enabled)
        for name, info in self.CORE_FEATURES.items():
            self.features[name] = FeatureConfig(
                name=name,
                category=info["category"],
                enabled=True,  # Core features always enabled
                description=info["description"],
            )

        # Add extended features (disabled by default)
        for name, info in self.EXTENDED_FEATURES.items():
            self.features[name] = FeatureConfig(
                name=name,
                category=info["category"],
                enabled=False,  # Extended features disabled by default
                description=info["description"],
            )

    def _load_config(self) -> None:
        """Load configuration from file if exists."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            for name, config_data in data.get("features", {}).items():
                if name in self.features:
                    # Update existing feature config
                    self.features[name].enabled = config_data.get("enabled", self.features[name].enabled)
                    self.features[name].importance_score = config_data.get("importance_score", 0.0)
                    if config_data.get("last_updated"):
                        self.features[name].last_updated = datetime.fromisoformat(config_data["last_updated"])

            logger.info(f"Loaded feature config from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load feature config: {e}")

    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "features": {name: f.to_dict() for name, f in self.features.items()},
            "saved_at": now_jst().isoformat(),
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Feature config saved to {self.config_path}")

    def get_enabled_features(self) -> list[str]:
        """Get list of enabled feature names."""
        return [name for name, f in self.features.items() if f.enabled]

    def get_core_features(self) -> list[str]:
        """Get list of core feature names."""
        return list(self.CORE_FEATURES.keys())

    def get_extended_features(self) -> list[str]:
        """Get list of extended feature names."""
        return list(self.EXTENDED_FEATURES.keys())

    def is_core_feature(self, name: str) -> bool:
        """Check if a feature is a core feature."""
        return name in self.CORE_FEATURES

    def enable_feature(self, name: str) -> bool:
        """
        Enable a feature.

        Args:
            name: Feature name

        Returns:
            True if successful
        """
        if name not in self.features:
            logger.warning(f"Unknown feature: {name}")
            return False

        self.features[name].enabled = True
        self.features[name].last_updated = now_jst()
        self.save_config()

        logger.info(f"Feature enabled: {name}")
        return True

    def disable_feature(self, name: str) -> bool:
        """
        Disable a feature.

        Args:
            name: Feature name

        Returns:
            True if successful
        """
        if name not in self.features:
            logger.warning(f"Unknown feature: {name}")
            return False

        # Don't disable core features
        if self.is_core_feature(name):
            logger.warning(f"Cannot disable core feature: {name}")
            return False

        self.features[name].enabled = False
        self.features[name].last_updated = now_jst()
        self.save_config()

        logger.info(f"Feature disabled: {name}")
        return True

    def toggle_feature(self, name: str) -> bool:
        """Toggle a feature on/off."""
        if name not in self.features:
            return False

        if self.features[name].enabled:
            return self.disable_feature(name)
        else:
            return self.enable_feature(name)

    def update_importance(self, name: str, score: float) -> bool:
        """
        Update importance score for a feature.

        Args:
            name: Feature name
            score: Importance score (0.0 to 1.0)

        Returns:
            True if successful
        """
        if name not in self.features:
            logger.warning(f"Unknown feature: {name}")
            return False

        self.features[name].importance_score = max(0.0, min(1.0, score))
        self.features[name].last_updated = now_jst()
        self.save_config()

        return True

    def update_importance_batch(self, scores: dict[str, float]) -> int:
        """
        Update importance scores for multiple features.

        Args:
            scores: Dict of feature_name -> importance_score

        Returns:
            Number of features updated
        """
        updated = 0
        for name, score in scores.items():
            if name in self.features:
                self.features[name].importance_score = max(0.0, min(1.0, score))
                self.features[name].last_updated = now_jst()
                updated += 1

        if updated > 0:
            self.save_config()

        return updated

    def get_feature_info(self, name: str) -> FeatureConfig | None:
        """Get feature configuration."""
        return self.features.get(name)

    def get_features_by_category(self, category: str) -> list[FeatureConfig]:
        """Get all features in a category."""
        return [f for f in self.features.values() if f.category == category]

    def get_top_features(self, n: int = 10, enabled_only: bool = True) -> list[FeatureConfig]:
        """Get top N features by importance score."""
        features = self.features.values()
        if enabled_only:
            features = [f for f in features if f.enabled]

        return sorted(features, key=lambda f: f.importance_score, reverse=True)[:n]

    def get_summary(self) -> dict:
        """Get summary of feature registry."""
        enabled = [f for f in self.features.values() if f.enabled]
        disabled = [f for f in self.features.values() if not f.enabled]

        categories = {}
        for f in self.features.values():
            if f.category not in categories:
                categories[f.category] = {"total": 0, "enabled": 0}
            categories[f.category]["total"] += 1
            if f.enabled:
                categories[f.category]["enabled"] += 1

        return {
            "total_features": len(self.features),
            "enabled_features": len(enabled),
            "disabled_features": len(disabled),
            "core_features": len(self.CORE_FEATURES),
            "extended_features": len(self.EXTENDED_FEATURES),
            "categories": categories,
            "top_by_importance": [
                {"name": f.name, "score": f.importance_score}
                for f in self.get_top_features(5)
            ],
        }

    def suggest_features_to_enable(self) -> list[str]:
        """
        Suggest features to enable based on importance scores.
        Returns disabled features with high importance.
        """
        disabled = [f for f in self.features.values() if not f.enabled and not self.is_core_feature(f.name)]
        high_importance = [f for f in disabled if f.importance_score > 0.5]

        return [f.name for f in sorted(high_importance, key=lambda f: f.importance_score, reverse=True)]

    def suggest_features_to_disable(self) -> list[str]:
        """
        Suggest features to disable based on importance scores.
        Returns enabled extended features with low importance.
        """
        enabled_extended = [
            f for f in self.features.values()
            if f.enabled and not self.is_core_feature(f.name)
        ]
        low_importance = [f for f in enabled_extended if f.importance_score < 0.1]

        return [f.name for f in sorted(low_importance, key=lambda f: f.importance_score)]
