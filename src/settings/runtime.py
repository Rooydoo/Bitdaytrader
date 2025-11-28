"""Runtime settings manager for dynamic configuration via Telegram."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class RuntimeSettings:
    """
    Manages runtime settings that can be changed via Telegram.

    Overrides .env settings without modifying the file.
    Settings are persisted to a JSON file for restart recovery.
    """

    # Settings that can be modified at runtime
    MODIFIABLE_SETTINGS = {
        # Portfolio allocation
        "symbols_config": str,
        "total_capital_utilization": float,
        "long_allocation_ratio": float,
        "short_allocation_ratio": float,
        # Risk management - LONG
        "long_risk_per_trade": float,
        "long_max_position_size": float,
        "long_max_daily_trades": int,
        "long_confidence_threshold": float,
        # Risk management - SHORT
        "short_risk_per_trade": float,
        "short_max_position_size": float,
        "short_max_daily_trades": int,
        "short_confidence_threshold": float,
        # Global
        "daily_loss_limit": float,
        "max_daily_trades": int,
        # Mode
        "mode": str,  # paper or live
    }

    def __init__(self, persist_path: str = "data/runtime_settings.json") -> None:
        """
        Initialize runtime settings manager.

        Args:
            persist_path: Path to JSON file for persistence
        """
        self.persist_path = Path(persist_path)
        self._overrides: dict[str, Any] = {}
        self._load_persisted()

    def _load_persisted(self) -> None:
        """Load persisted settings from JSON file."""
        if self.persist_path.exists():
            try:
                with open(self.persist_path) as f:
                    self._overrides = json.load(f)
                logger.info(f"Loaded {len(self._overrides)} runtime settings overrides")
            except Exception as e:
                logger.error(f"Failed to load runtime settings: {e}")
                self._overrides = {}

    def _save_persisted(self) -> None:
        """Save current overrides to JSON file."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(self._overrides, f, indent=2)
            logger.debug(f"Saved runtime settings to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save runtime settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a runtime setting override.

        Args:
            key: Setting key
            default: Default value if not overridden

        Returns:
            Overridden value or default
        """
        return self._overrides.get(key, default)

    def set(self, key: str, value: Any) -> tuple[bool, str]:
        """
        Set a runtime setting override.

        Args:
            key: Setting key
            value: New value

        Returns:
            Tuple of (success, message)
        """
        if key not in self.MODIFIABLE_SETTINGS:
            return False, f"Setting '{key}' is not modifiable at runtime"

        expected_type = self.MODIFIABLE_SETTINGS[key]

        try:
            # Type conversion
            if expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(value)
            elif expected_type == str:
                value = str(value)

            # Validation
            valid, msg = self._validate(key, value)
            if not valid:
                return False, msg

            old_value = self._overrides.get(key)
            self._overrides[key] = value
            self._save_persisted()

            if old_value is not None:
                return True, f"Updated {key}: {old_value} → {value}"
            else:
                return True, f"Set {key} = {value}"

        except ValueError as e:
            return False, f"Invalid value for {key}: {e}"

    def _validate(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate a setting value."""
        # Percentage values (0-1)
        if key in [
            "total_capital_utilization",
            "long_allocation_ratio",
            "short_allocation_ratio",
            "long_risk_per_trade",
            "short_risk_per_trade",
            "long_max_position_size",
            "short_max_position_size",
            "daily_loss_limit",
            "long_confidence_threshold",
            "short_confidence_threshold",
        ]:
            if not 0 <= value <= 1:
                return False, f"{key} must be between 0 and 1 (got {value})"

        # LONG/SHORT ratio should sum to 1
        if key == "long_allocation_ratio":
            short_ratio = self._overrides.get("short_allocation_ratio", 0.4)
            if value + short_ratio > 1:
                return False, f"LONG + SHORT ratios exceed 100% ({value} + {short_ratio})"

        if key == "short_allocation_ratio":
            long_ratio = self._overrides.get("long_allocation_ratio", 0.6)
            if value + long_ratio > 1:
                return False, f"LONG + SHORT ratios exceed 100% ({long_ratio} + {value})"

        # Integer limits
        if key in ["long_max_daily_trades", "short_max_daily_trades", "max_daily_trades"]:
            if value < 0 or value > 20:
                return False, f"{key} must be between 0 and 20 (got {value})"

        # Mode validation
        if key == "mode":
            if value not in ["paper", "live", "backtest"]:
                return False, f"mode must be 'paper', 'live', or 'backtest' (got {value})"

        # Symbols config validation
        if key == "symbols_config":
            try:
                total = 0
                for item in value.split(","):
                    item = item.strip()
                    if ":" in item:
                        _, alloc = item.split(":")
                        total += float(alloc.strip())
                if total > 1:
                    return False, f"Symbol allocations exceed 100% (total: {total:.0%})"
            except Exception as e:
                return False, f"Invalid symbols_config format: {e}"

        return True, "OK"

    def delete(self, key: str) -> tuple[bool, str]:
        """
        Remove a runtime override (revert to .env value).

        Args:
            key: Setting key

        Returns:
            Tuple of (success, message)
        """
        if key in self._overrides:
            del self._overrides[key]
            self._save_persisted()
            return True, f"Removed override for {key} (reverted to .env)"
        return False, f"No override exists for {key}"

    def get_all_overrides(self) -> dict[str, Any]:
        """Get all current overrides."""
        return self._overrides.copy()

    def clear_all(self) -> None:
        """Clear all overrides."""
        self._overrides = {}
        self._save_persisted()
        logger.info("Cleared all runtime setting overrides")

    def get_display_summary(self) -> str:
        """Get a formatted summary for Telegram display."""
        if not self._overrides:
            return "No runtime overrides (using .env defaults)"

        lines = ["<b>Runtime Overrides:</b>"]
        for key, value in sorted(self._overrides.items()):
            if isinstance(value, float):
                if key.endswith("_ratio") or key.endswith("_threshold") or "limit" in key or "utilization" in key:
                    lines.append(f"• {key}: {value:.0%}")
                else:
                    lines.append(f"• {key}: {value:.4f}")
            else:
                lines.append(f"• {key}: {value}")

        return "\n".join(lines)


# Global instance
_runtime_settings: RuntimeSettings | None = None


def get_runtime_settings() -> RuntimeSettings:
    """Get global runtime settings instance."""
    global _runtime_settings
    if _runtime_settings is None:
        _runtime_settings = RuntimeSettings()
    return _runtime_settings
