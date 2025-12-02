"""Risk management module with direction-specific settings."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    from src.agent.long_term_memory import LongTermMemory


@dataclass
class MemoryBasedAdjustment:
    """Adjustment suggested by long-term memory rules."""

    parameter: str
    adjustment_type: str  # "multiply", "add", "set"
    value: float
    reason: str
    rule_id: str | None = None
    confidence: str = "medium"  # "high", "medium", "low"


@dataclass
class DynamicLeverageConfig:
    """Configuration for dynamic leverage adjustment."""

    # Base leverage from settings
    base_leverage: float = 2.0
    min_leverage: float = 1.0  # Floor - never go below this
    max_leverage: float = 4.0  # Ceiling - never exceed this

    # Confidence thresholds and multipliers
    high_confidence_threshold: float = 0.85  # >= 85% = aggressive
    medium_confidence_threshold: float = 0.75  # >= 75% = normal
    low_confidence_threshold: float = 0.65  # >= 65% = conservative

    high_confidence_multiplier: float = 1.3  # Up to 30% more leverage
    medium_confidence_multiplier: float = 1.0  # Normal leverage
    low_confidence_multiplier: float = 0.7  # 30% less leverage
    very_low_confidence_multiplier: float = 0.5  # 50% less leverage

    # Volatility thresholds (ATR as % of price)
    high_volatility_threshold: float = 0.03  # 3%+ = high vol
    low_volatility_threshold: float = 0.01  # <1% = low vol

    high_volatility_multiplier: float = 0.6  # Reduce leverage in high vol
    normal_volatility_multiplier: float = 1.0
    low_volatility_multiplier: float = 1.1  # Slightly more in low vol


@dataclass
class LeverageCalculation:
    """Result of dynamic leverage calculation."""

    base_leverage: float
    adjusted_leverage: float
    confidence_factor: float
    volatility_factor: float
    memory_factor: float
    conservative_factor: float
    reasons: list[str]

    @property
    def total_adjustment(self) -> float:
        """Total adjustment ratio from base."""
        if self.base_leverage == 0:
            return 0.0
        return self.adjusted_leverage / self.base_leverage


@dataclass
class LosingStreakConfig:
    """Configuration for losing streak risk reduction."""

    # At how many consecutive losses to start reducing risk
    start_reduction_at: int = 2
    # Reduction per additional loss (e.g., 0.20 = 20% reduction per loss)
    reduction_per_loss: float = 0.20
    # Minimum risk multiplier (floor)
    min_risk_multiplier: float = 0.30
    # After how many consecutive wins to fully recover
    wins_to_recover: int = 2


@dataclass
class DrawdownState:
    """Tracks drawdown across different time periods."""

    # Peak capital tracking
    peak_capital: float = 0.0
    current_capital: float = 0.0

    # Weekly tracking
    weekly_start_capital: float = 0.0
    weekly_pnl: float = 0.0
    week_start_date: date | None = None

    # Monthly tracking
    monthly_start_capital: float = 0.0
    monthly_pnl: float = 0.0
    month_start_date: date | None = None

    # Losing streak tracking
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    consecutive_wins: int = 0  # Track wins after a losing streak

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak (0.0 to 1.0)."""
        if self.peak_capital <= 0:
            return 0.0
        return max(0.0, (self.peak_capital - self.current_capital) / self.peak_capital)

    @property
    def weekly_drawdown(self) -> float:
        """Calculate weekly drawdown (0.0 to 1.0)."""
        if self.weekly_start_capital <= 0:
            return 0.0
        if self.weekly_pnl >= 0:
            return 0.0
        return abs(self.weekly_pnl) / self.weekly_start_capital

    @property
    def monthly_drawdown(self) -> float:
        """Calculate monthly drawdown (0.0 to 1.0)."""
        if self.monthly_start_capital <= 0:
            return 0.0
        if self.monthly_pnl >= 0:
            return 0.0
        return abs(self.monthly_pnl) / self.monthly_start_capital


@dataclass
class DirectionConfig:
    """Configuration for a specific direction (LONG or SHORT)."""

    risk_per_trade: float
    max_position_size: float
    max_daily_trades: int
    confidence_threshold: float
    sl_atr_multiple: float
    tp_levels: list[tuple[float, float]]  # (R-multiple, ratio)


@dataclass
class PositionSize:
    """Position size calculation result."""

    size: float
    stop_loss: float
    risk_amount: float
    position_value: float


@dataclass
class RiskCheck:
    """Risk check result."""

    allowed: bool
    reason: str = ""


@dataclass
class DirectionStats:
    """Statistics for a specific direction."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0


class RiskManager:
    """Risk management with separate settings for LONG and SHORT positions."""

    def __init__(
        self,
        # LONG settings - 確信度閾値を75%に引き上げ
        long_risk_per_trade: float = 0.02,
        long_max_position_size: float = 0.10,
        long_max_daily_trades: int = 2,  # 3→2に削減（税金効率化）
        long_confidence_threshold: float = 0.75,  # 65%→75%に引き上げ
        long_sl_atr_multiple: float = 2.0,
        long_tp_levels: list[tuple[float, float]] | None = None,
        # SHORT settings (stricter by default) - 確信度閾値を80%に引き上げ
        short_risk_per_trade: float = 0.015,
        short_max_position_size: float = 0.07,
        short_max_daily_trades: int = 1,  # 2→1に削減（税金効率化）
        short_confidence_threshold: float = 0.80,  # 70%→80%に引き上げ
        short_sl_atr_multiple: float = 1.5,
        short_tp_levels: list[tuple[float, float]] | None = None,
        # Global settings - 取引頻度を制限
        daily_loss_limit: float = 0.03,
        weekly_loss_limit: float = 0.07,
        monthly_loss_limit: float = 0.12,
        max_drawdown_limit: float = 0.15,
        max_daily_trades: int = 3,  # 5→3に削減（税金効率化）
        max_consecutive_losses: int = 5,
        # Losing streak risk reduction settings
        losing_streak_config: LosingStreakConfig | None = None,
    ) -> None:
        """
        Initialize risk manager with direction-specific settings.

        Args:
            long_*: Settings for LONG positions
            short_*: Settings for SHORT positions (stricter defaults)
            daily_loss_limit: Max daily loss as fraction of capital (default: 3%)
            weekly_loss_limit: Max weekly loss as fraction of capital (default: 7%)
            monthly_loss_limit: Max monthly loss as fraction of capital (default: 12%)
            max_drawdown_limit: Max drawdown from peak capital (default: 15%)
            max_daily_trades: Max total trades per day
            max_consecutive_losses: Stop after N consecutive losses (default: 5)
            losing_streak_config: Configuration for losing streak risk reduction
        """
        # LONG configuration - 利益を伸ばすためTP引き上げ
        self.long_config = DirectionConfig(
            risk_per_trade=long_risk_per_trade,
            max_position_size=long_max_position_size,
            max_daily_trades=long_max_daily_trades,
            confidence_threshold=long_confidence_threshold,
            sl_atr_multiple=long_sl_atr_multiple,
            # 期待R比: 0.33×2.0 + 0.33×3.0 + 0.34×5.0 = 3.35R（全て到達時）
            tp_levels=long_tp_levels or [(2.0, 0.33), (3.0, 0.33), (5.0, 0.34)],
        )

        # SHORT configuration (stricter) - 早めに利確だがTP1は1.5R確保
        self.short_config = DirectionConfig(
            risk_per_trade=short_risk_per_trade,
            max_position_size=short_max_position_size,
            max_daily_trades=short_max_daily_trades,
            confidence_threshold=short_confidence_threshold,
            sl_atr_multiple=short_sl_atr_multiple,
            # 期待R比: 0.40×1.5 + 0.35×2.0 + 0.25×3.0 = 2.05R（全て到達時）
            tp_levels=short_tp_levels or [(1.5, 0.40), (2.0, 0.35), (3.0, 0.25)],
        )

        # Global limits
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.monthly_loss_limit = monthly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_daily_trades = max_daily_trades
        self.max_consecutive_losses = max_consecutive_losses

        # Daily tracking - total
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date | None = None

        # Daily tracking - by direction
        self._long_stats = DirectionStats()
        self._short_stats = DirectionStats()

        # Drawdown tracking
        self._drawdown = DrawdownState()

        # Losing streak risk reduction
        self.losing_streak_config = losing_streak_config or LosingStreakConfig()

        # Conservative mode (activated when overfitting detected)
        self._conservative_mode = False
        self._conservative_multiplier = 0.5  # Reduce risk by 50%
        self._original_long_config: DirectionConfig | None = None
        self._original_short_config: DirectionConfig | None = None

        # Portfolio allocation settings (set from Settings)
        self._symbol_allocations: dict[str, float] = {}  # symbol -> allocation %
        self._total_capital_utilization: float = 1.0  # How much of total capital to use
        self._long_allocation_ratio: float = 0.6  # LONG portion of allocated capital
        self._short_allocation_ratio: float = 0.4  # SHORT portion of allocated capital

        # Position tracking per symbol
        self._symbol_positions: dict[str, dict[str, float]] = {}  # symbol -> {long: value, short: value}

        # Long-term memory reference (optional)
        self._long_term_memory: "LongTermMemory | None" = None
        self._memory_adjustments_enabled: bool = True
        self._last_memory_check: datetime | None = None
        self._cached_adjustments: list[MemoryBasedAdjustment] = []

        # Dynamic leverage configuration
        self._dynamic_leverage_config: DynamicLeverageConfig | None = None
        self._dynamic_leverage_enabled: bool = False

    def set_long_term_memory(
        self,
        memory: "LongTermMemory",
        enable_adjustments: bool = True,
    ) -> None:
        """
        Set long-term memory reference for rule-based adjustments.

        Args:
            memory: LongTermMemory instance
            enable_adjustments: Whether to apply memory-based adjustments
        """
        self._long_term_memory = memory
        self._memory_adjustments_enabled = enable_adjustments
        logger.info(
            f"Long-term memory connected to RiskManager "
            f"(adjustments: {'enabled' if enable_adjustments else 'disabled'})"
        )

    def enable_memory_adjustments(self, enabled: bool = True) -> None:
        """Enable or disable memory-based adjustments."""
        self._memory_adjustments_enabled = enabled
        logger.info(f"Memory-based adjustments: {'enabled' if enabled else 'disabled'}")

    def get_memory_adjustments(
        self,
        side: str,
        market_conditions: dict | None = None,
    ) -> list[MemoryBasedAdjustment]:
        """
        Get risk adjustments based on long-term memory rules.

        Args:
            side: "BUY" (LONG) or "SELL" (SHORT)
            market_conditions: Current market conditions (volatility, trend, etc.)

        Returns:
            List of suggested adjustments
        """
        if not self._long_term_memory or not self._memory_adjustments_enabled:
            return []

        adjustments = []
        direction = "LONG" if side == "BUY" else "SHORT"

        # Get active rules from memory
        rules = self._long_term_memory.get_active_rules()

        for rule in rules:
            # Skip low confidence rules
            if rule.confidence.value == "low":
                continue

            # Parse rule content for risk-related keywords
            content_lower = rule.content.lower()

            # Check for position size adjustments
            if "ポジション" in content_lower and ("縮小" in content_lower or "減らす" in content_lower):
                if self._rule_applies_to_conditions(rule, market_conditions):
                    adjustments.append(MemoryBasedAdjustment(
                        parameter="position_size_multiplier",
                        adjustment_type="multiply",
                        value=0.7 if rule.confidence.value == "high" else 0.8,
                        reason=rule.content[:100],
                        rule_id=rule.id,
                        confidence=rule.confidence.value,
                    ))

            # Check for confidence threshold adjustments
            if "閾値" in content_lower and ("上げる" in content_lower or "厳しく" in content_lower):
                if self._rule_applies_to_conditions(rule, market_conditions):
                    adjustments.append(MemoryBasedAdjustment(
                        parameter="confidence_threshold_add",
                        adjustment_type="add",
                        value=0.05 if rule.confidence.value == "high" else 0.03,
                        reason=rule.content[:100],
                        rule_id=rule.id,
                        confidence=rule.confidence.value,
                    ))

            # Check for risk per trade adjustments
            if "リスク" in content_lower and ("下げる" in content_lower or "減らす" in content_lower):
                if self._rule_applies_to_conditions(rule, market_conditions):
                    adjustments.append(MemoryBasedAdjustment(
                        parameter="risk_per_trade_multiplier",
                        adjustment_type="multiply",
                        value=0.7 if rule.confidence.value == "high" else 0.85,
                        reason=rule.content[:100],
                        rule_id=rule.id,
                        confidence=rule.confidence.value,
                    ))

            # Check for direction-specific rules
            if direction.lower() in content_lower or direction in content_lower:
                if "控える" in content_lower or "避ける" in content_lower or "停止" in content_lower:
                    adjustments.append(MemoryBasedAdjustment(
                        parameter=f"{direction.lower()}_should_skip",
                        adjustment_type="set",
                        value=1.0,
                        reason=rule.content[:100],
                        rule_id=rule.id,
                        confidence=rule.confidence.value,
                    ))

        self._cached_adjustments = adjustments
        self._last_memory_check = datetime.now()

        if adjustments:
            logger.info(
                f"Memory-based adjustments for {direction}: {len(adjustments)} rules applied"
            )

        return adjustments

    def _rule_applies_to_conditions(
        self,
        rule,
        market_conditions: dict | None,
    ) -> bool:
        """Check if a rule applies to current market conditions."""
        if not market_conditions:
            return True  # Apply by default if no conditions provided

        content_lower = rule.content.lower()

        # Check volatility conditions
        if "ボラティリティ" in content_lower or "変動" in content_lower:
            volatility = market_conditions.get("volatility", "normal")
            if "高" in content_lower and volatility != "high":
                return False
            if "低" in content_lower and volatility != "low":
                return False

        # Check trend conditions
        if "トレンド" in content_lower:
            trend = market_conditions.get("trend", "neutral")
            if "上昇" in content_lower and trend != "up":
                return False
            if "下降" in content_lower and trend != "down":
                return False

        # Check consecutive losses condition
        if "連続損失" in content_lower or "連敗" in content_lower:
            consecutive_losses = self._drawdown.consecutive_losses
            if consecutive_losses < 2:  # Only apply if actually in a losing streak
                return False

        return True

    def apply_memory_adjustments(
        self,
        base_value: float,
        parameter: str,
        adjustments: list[MemoryBasedAdjustment],
    ) -> tuple[float, list[str]]:
        """
        Apply memory-based adjustments to a parameter.

        Args:
            base_value: Original parameter value
            parameter: Parameter name to adjust
            adjustments: List of adjustments to consider

        Returns:
            Tuple of (adjusted_value, list of reasons)
        """
        adjusted = base_value
        reasons = []

        # Filter relevant adjustments
        relevant = [a for a in adjustments if parameter in a.parameter]

        for adj in relevant:
            if adj.adjustment_type == "multiply":
                adjusted *= adj.value
                reasons.append(f"{adj.reason} (×{adj.value:.2f})")
            elif adj.adjustment_type == "add":
                adjusted += adj.value
                reasons.append(f"{adj.reason} (+{adj.value:.2f})")
            elif adj.adjustment_type == "set":
                adjusted = adj.value
                reasons.append(f"{adj.reason} (={adj.value})")

        return adjusted, reasons

    def should_skip_trade(
        self,
        side: str,
        market_conditions: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Check if trade should be skipped based on memory rules.

        Args:
            side: "BUY" (LONG) or "SELL" (SHORT)
            market_conditions: Current market conditions

        Returns:
            Tuple of (should_skip, reason)
        """
        adjustments = self.get_memory_adjustments(side, market_conditions)
        direction = "LONG" if side == "BUY" else "SHORT"

        for adj in adjustments:
            if adj.parameter == f"{direction.lower()}_should_skip" and adj.value > 0:
                return True, adj.reason

        return False, ""

    def get_memory_status(self) -> dict:
        """Get status of memory-based adjustments."""
        return {
            "memory_connected": self._long_term_memory is not None,
            "adjustments_enabled": self._memory_adjustments_enabled,
            "last_check": self._last_memory_check.isoformat() if self._last_memory_check else None,
            "cached_adjustments": len(self._cached_adjustments),
            "active_rules_count": (
                len(self._long_term_memory.get_active_rules())
                if self._long_term_memory else 0
            ),
        }

    # ========== Dynamic Leverage Methods ==========

    def enable_dynamic_leverage(
        self,
        config: DynamicLeverageConfig | None = None,
    ) -> None:
        """
        Enable dynamic leverage adjustment based on prediction confidence and market conditions.

        Args:
            config: Configuration for dynamic leverage, or use defaults
        """
        self._dynamic_leverage_config = config or DynamicLeverageConfig()
        self._dynamic_leverage_enabled = True
        logger.info(
            f"Dynamic leverage enabled: base={self._dynamic_leverage_config.base_leverage}x, "
            f"range=[{self._dynamic_leverage_config.min_leverage}x, {self._dynamic_leverage_config.max_leverage}x]"
        )

    def disable_dynamic_leverage(self) -> None:
        """Disable dynamic leverage adjustment."""
        self._dynamic_leverage_enabled = False
        logger.info("Dynamic leverage disabled")

    def calculate_dynamic_leverage(
        self,
        confidence: float,
        side: str,
        current_price: float | None = None,
        atr: float | None = None,
        market_conditions: dict | None = None,
    ) -> LeverageCalculation:
        """
        Calculate dynamic leverage based on prediction confidence and market conditions.

        Higher confidence predictions get more leverage to maximize profits.
        Lower confidence or risky conditions get less leverage to minimize losses.

        Args:
            confidence: Prediction confidence (0.0 to 1.0)
            side: "BUY" (LONG) or "SELL" (SHORT)
            current_price: Current market price (for volatility calculation)
            atr: Current ATR value (for volatility calculation)
            market_conditions: Additional market context

        Returns:
            LeverageCalculation with adjusted leverage and breakdown
        """
        if not self._dynamic_leverage_enabled or not self._dynamic_leverage_config:
            # Return base leverage without adjustment
            base = getattr(self._dynamic_leverage_config, 'base_leverage', 2.0) if self._dynamic_leverage_config else 2.0
            return LeverageCalculation(
                base_leverage=base,
                adjusted_leverage=base,
                confidence_factor=1.0,
                volatility_factor=1.0,
                memory_factor=1.0,
                conservative_factor=1.0,
                reasons=["Dynamic leverage disabled"],
            )

        config = self._dynamic_leverage_config
        reasons = []

        # 1. Base leverage
        leverage = config.base_leverage

        # 2. Confidence factor
        confidence_factor = self._calculate_confidence_factor(confidence, config)
        reasons.append(f"Confidence {confidence:.1%} → factor {confidence_factor:.2f}")

        # 3. Volatility factor
        volatility_factor = self._calculate_volatility_factor(
            current_price, atr, config
        )
        if volatility_factor != 1.0:
            vol_pct = (atr / current_price * 100) if current_price and atr else 0
            reasons.append(f"Volatility {vol_pct:.1f}% → factor {volatility_factor:.2f}")

        # 4. Memory-based factor (from learned rules)
        memory_factor = self._calculate_memory_leverage_factor(side, market_conditions)
        if memory_factor != 1.0:
            reasons.append(f"Memory rules → factor {memory_factor:.2f}")

        # 5. Conservative mode factor
        conservative_factor = 1.0
        if self._conservative_mode:
            conservative_factor = self._conservative_multiplier
            reasons.append(f"Conservative mode → factor {conservative_factor:.2f}")

        # NOTE: Losing streak factor is NOT applied here to avoid double-application.
        # It is already applied in calculate_position_size() via get_losing_streak_multiplier().
        # Applying it here would cause excessive risk reduction during losing streaks.

        # Apply all factors
        adjusted_leverage = (
            leverage
            * confidence_factor
            * volatility_factor
            * memory_factor
            * conservative_factor
        )

        # Clamp to configured range
        adjusted_leverage = max(config.min_leverage, min(config.max_leverage, adjusted_leverage))

        # Log significant adjustments
        if abs(adjusted_leverage - leverage) > 0.1:
            logger.info(
                f"Dynamic leverage: {leverage:.1f}x → {adjusted_leverage:.1f}x "
                f"(conf={confidence_factor:.2f}, vol={volatility_factor:.2f}, "
                f"mem={memory_factor:.2f}, cons={conservative_factor:.2f})"
            )

        return LeverageCalculation(
            base_leverage=leverage,
            adjusted_leverage=adjusted_leverage,
            confidence_factor=confidence_factor,
            volatility_factor=volatility_factor,
            memory_factor=memory_factor,
            conservative_factor=conservative_factor,
            reasons=reasons,
        )

    def _calculate_confidence_factor(
        self,
        confidence: float,
        config: DynamicLeverageConfig,
    ) -> float:
        """Calculate leverage factor based on prediction confidence."""
        if confidence >= config.high_confidence_threshold:
            # High confidence: increase leverage
            return config.high_confidence_multiplier
        elif confidence >= config.medium_confidence_threshold:
            # Medium confidence: normal leverage
            return config.medium_confidence_multiplier
        elif confidence >= config.low_confidence_threshold:
            # Low confidence: reduce leverage
            return config.low_confidence_multiplier
        else:
            # Very low confidence: significantly reduce
            return config.very_low_confidence_multiplier

    def _calculate_volatility_factor(
        self,
        current_price: float | None,
        atr: float | None,
        config: DynamicLeverageConfig,
    ) -> float:
        """Calculate leverage factor based on market volatility."""
        if current_price is None or atr is None or current_price <= 0:
            return config.normal_volatility_multiplier

        # Calculate volatility as ATR / price
        volatility_pct = atr / current_price

        if volatility_pct >= config.high_volatility_threshold:
            # High volatility: reduce leverage to manage risk
            return config.high_volatility_multiplier
        elif volatility_pct <= config.low_volatility_threshold:
            # Low volatility: can use slightly more leverage
            return config.low_volatility_multiplier
        else:
            return config.normal_volatility_multiplier

    def _calculate_memory_leverage_factor(
        self,
        side: str,
        market_conditions: dict | None = None,
    ) -> float:
        """Calculate leverage factor based on long-term memory rules."""
        if not self._long_term_memory or not self._memory_adjustments_enabled:
            return 1.0

        factor = 1.0
        direction = "LONG" if side == "BUY" else "SHORT"

        # Get active rules from memory
        rules = self._long_term_memory.get_active_rules()

        for rule in rules:
            if rule.confidence.value == "low":
                continue

            content_lower = rule.content.lower()

            # Check for leverage-related rules
            if "レバレッジ" in content_lower or "leverage" in content_lower.lower():
                if self._rule_applies_to_conditions(rule, market_conditions):
                    # Check if rule suggests reducing leverage
                    if any(word in content_lower for word in ["下げる", "減らす", "控える", "reduce", "lower"]):
                        reduction = 0.7 if rule.confidence.value == "high" else 0.85
                        factor *= reduction
                        logger.debug(f"Memory rule reduces leverage: {rule.content[:50]}...")

                    # Check if rule suggests increasing leverage
                    elif any(word in content_lower for word in ["上げる", "増やす", "積極", "increase"]):
                        increase = 1.2 if rule.confidence.value == "high" else 1.1
                        factor *= increase
                        logger.debug(f"Memory rule increases leverage: {rule.content[:50]}...")

            # Check for direction-specific caution rules
            if direction.lower() in content_lower or direction in content_lower:
                if any(word in content_lower for word in ["注意", "慎重", "careful", "caution"]):
                    if self._rule_applies_to_conditions(rule, market_conditions):
                        factor *= 0.8
                        logger.debug(f"Memory rule advises caution for {direction}")

        # Clamp factor to reasonable range
        return max(0.5, min(1.5, factor))

    def get_dynamic_leverage_status(self) -> dict:
        """Get status of dynamic leverage configuration."""
        config = self._dynamic_leverage_config
        return {
            "enabled": self._dynamic_leverage_enabled,
            "base_leverage": config.base_leverage if config else None,
            "min_leverage": config.min_leverage if config else None,
            "max_leverage": config.max_leverage if config else None,
            "high_confidence_threshold": config.high_confidence_threshold if config else None,
            "current_streak_factor": self.get_losing_streak_multiplier(),
            "consecutive_losses": self._drawdown.consecutive_losses,
        }

    def enable_conservative_mode(self, multiplier: float = 0.5) -> None:
        """
        Enable conservative mode - reduces all risk parameters.

        Called when overfitting is detected to protect capital.

        Args:
            multiplier: Risk reduction factor (0.5 = 50% of normal risk)
        """
        if self._conservative_mode:
            logger.warning("Conservative mode already enabled")
            return

        self._conservative_multiplier = multiplier

        # Store original configs
        self._original_long_config = DirectionConfig(
            risk_per_trade=self.long_config.risk_per_trade,
            max_position_size=self.long_config.max_position_size,
            max_daily_trades=self.long_config.max_daily_trades,
            confidence_threshold=self.long_config.confidence_threshold,
            sl_atr_multiple=self.long_config.sl_atr_multiple,
            tp_levels=self.long_config.tp_levels.copy(),
        )
        self._original_short_config = DirectionConfig(
            risk_per_trade=self.short_config.risk_per_trade,
            max_position_size=self.short_config.max_position_size,
            max_daily_trades=self.short_config.max_daily_trades,
            confidence_threshold=self.short_config.confidence_threshold,
            sl_atr_multiple=self.short_config.sl_atr_multiple,
            tp_levels=self.short_config.tp_levels.copy(),
        )

        # Apply conservative adjustments
        for config in [self.long_config, self.short_config]:
            config.risk_per_trade *= multiplier
            config.max_position_size *= multiplier
            config.max_daily_trades = max(1, int(config.max_daily_trades * multiplier))
            # Increase confidence threshold (be more selective)
            config.confidence_threshold = min(0.85, config.confidence_threshold + 0.10)
            # Tighter stop loss
            config.sl_atr_multiple *= 0.75
            # Take profits earlier (reduce R-multiples)
            config.tp_levels = [
                (level * 0.75, ratio) for level, ratio in config.tp_levels
            ]

        self._conservative_mode = True
        logger.warning(
            f"CONSERVATIVE MODE ENABLED: Risk reduced to {multiplier:.0%}, "
            f"confidence threshold increased by 10%"
        )

    def disable_conservative_mode(self) -> None:
        """Disable conservative mode and restore original settings."""
        if not self._conservative_mode:
            logger.warning("Conservative mode not enabled")
            return

        if self._original_long_config:
            self.long_config = self._original_long_config
            self._original_long_config = None
        if self._original_short_config:
            self.short_config = self._original_short_config
            self._original_short_config = None

        self._conservative_mode = False
        logger.info("Conservative mode disabled, original settings restored")

    @property
    def is_conservative_mode(self) -> bool:
        """Check if conservative mode is active."""
        return self._conservative_mode

    def configure_allocation(
        self,
        symbol_allocations: dict[str, float],
        total_capital_utilization: float = 0.80,
        long_allocation_ratio: float = 0.60,
        short_allocation_ratio: float = 0.40,
    ) -> None:
        """
        Configure portfolio allocation settings.

        Args:
            symbol_allocations: Dict of symbol -> allocation percentage
            total_capital_utilization: How much of total capital to use (0.8 = 80%)
            long_allocation_ratio: Portion for LONG positions (0.6 = 60%)
            short_allocation_ratio: Portion for SHORT positions (0.4 = 40%)
        """
        self._symbol_allocations = symbol_allocations
        self._total_capital_utilization = total_capital_utilization
        self._long_allocation_ratio = long_allocation_ratio
        self._short_allocation_ratio = short_allocation_ratio

        # Initialize position tracking for each symbol
        for symbol in symbol_allocations:
            if symbol not in self._symbol_positions:
                self._symbol_positions[symbol] = {"long": 0.0, "short": 0.0}

        logger.info(
            f"Portfolio allocation configured: {len(symbol_allocations)} symbols, "
            f"utilization={total_capital_utilization:.0%}, "
            f"long/short={long_allocation_ratio:.0%}/{short_allocation_ratio:.0%}"
        )

    def get_allocated_capital(
        self,
        symbol: str,
        direction: str,
        total_capital: float,
    ) -> float:
        """
        Calculate allocated capital for a symbol and direction.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            total_capital: Total available capital

        Returns:
            Allocated capital for this symbol/direction
        """
        # Get symbol allocation (default to 100% if not configured)
        symbol_pct = self._symbol_allocations.get(symbol, 1.0)

        # Calculate: total * utilization * symbol_pct * direction_ratio
        usable_capital = total_capital * self._total_capital_utilization
        symbol_capital = usable_capital * symbol_pct

        if direction == "LONG":
            return symbol_capital * self._long_allocation_ratio
        else:  # SHORT
            return symbol_capital * self._short_allocation_ratio

    def get_remaining_capital(
        self,
        symbol: str,
        direction: str,
        total_capital: float,
    ) -> float:
        """
        Calculate remaining capital for a symbol/direction after open positions.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            total_capital: Total available capital

        Returns:
            Remaining allocable capital
        """
        allocated = self.get_allocated_capital(symbol, direction, total_capital)
        used = self._symbol_positions.get(symbol, {}).get(direction.lower(), 0.0)
        return max(0.0, allocated - used)

    def update_position(
        self,
        symbol: str,
        direction: str,
        position_value: float,
        is_close: bool = False,
    ) -> None:
        """
        Update position tracking when opening/closing positions.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            position_value: Position value in JPY
            is_close: True if closing a position
        """
        if symbol not in self._symbol_positions:
            self._symbol_positions[symbol] = {"long": 0.0, "short": 0.0}

        key = direction.lower()
        if is_close:
            self._symbol_positions[symbol][key] = max(
                0.0, self._symbol_positions[symbol][key] - position_value
            )
        else:
            self._symbol_positions[symbol][key] += position_value

        logger.debug(
            f"Position updated: {symbol} {direction} = ¥{self._symbol_positions[symbol][key]:,.0f}"
        )

    def get_allocation_summary(self, total_capital: float) -> dict:
        """
        Get summary of current allocation status.

        Args:
            total_capital: Total available capital

        Returns:
            Dict with allocation details per symbol
        """
        summary = {
            "total_capital": total_capital,
            "utilization_rate": self._total_capital_utilization,
            "usable_capital": total_capital * self._total_capital_utilization,
            "long_ratio": self._long_allocation_ratio,
            "short_ratio": self._short_allocation_ratio,
            "symbols": {},
        }

        for symbol, alloc in self._symbol_allocations.items():
            symbol_capital = total_capital * self._total_capital_utilization * alloc
            positions = self._symbol_positions.get(symbol, {"long": 0.0, "short": 0.0})

            summary["symbols"][symbol] = {
                "allocation_pct": alloc,
                "allocated_capital": symbol_capital,
                "long_allocated": symbol_capital * self._long_allocation_ratio,
                "long_used": positions["long"],
                "long_remaining": max(0, symbol_capital * self._long_allocation_ratio - positions["long"]),
                "short_allocated": symbol_capital * self._short_allocation_ratio,
                "short_used": positions["short"],
                "short_remaining": max(0, symbol_capital * self._short_allocation_ratio - positions["short"]),
            }

        return summary

    def get_config(self, side: str) -> DirectionConfig:
        """Get configuration for the specified direction."""
        if side == "BUY":
            return self.long_config
        else:
            return self.short_config

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._long_stats = DirectionStats()
        self._short_stats = DirectionStats()
        self._last_reset_date = date.today()
        logger.info("Daily risk counters reset")

    def _check_and_reset_daily(self) -> None:
        """Check if we need to reset daily counters."""
        today = date.today()
        if self._last_reset_date != today:
            self.reset_daily()

    def _check_and_reset_weekly(self, capital: float) -> None:
        """Check if we need to reset weekly counters (Monday)."""
        today = date.today()
        # Reset on Monday (weekday() == 0)
        if self._drawdown.week_start_date is None:
            self._drawdown.week_start_date = today
            self._drawdown.weekly_start_capital = capital
            self._drawdown.weekly_pnl = 0.0
        elif today.isocalendar()[1] != self._drawdown.week_start_date.isocalendar()[1]:
            # New week
            logger.info(
                f"Weekly reset: Previous week PnL = ¥{self._drawdown.weekly_pnl:,.0f} "
                f"({self._drawdown.weekly_drawdown:.2%} drawdown)"
            )
            self._drawdown.week_start_date = today
            self._drawdown.weekly_start_capital = capital
            self._drawdown.weekly_pnl = 0.0

    def _check_and_reset_monthly(self, capital: float) -> None:
        """Check if we need to reset monthly counters."""
        today = date.today()
        if self._drawdown.month_start_date is None:
            self._drawdown.month_start_date = today
            self._drawdown.monthly_start_capital = capital
            self._drawdown.monthly_pnl = 0.0
        elif today.month != self._drawdown.month_start_date.month:
            # New month
            logger.info(
                f"Monthly reset: Previous month PnL = ¥{self._drawdown.monthly_pnl:,.0f} "
                f"({self._drawdown.monthly_drawdown:.2%} drawdown)"
            )
            self._drawdown.month_start_date = today
            self._drawdown.monthly_start_capital = capital
            self._drawdown.monthly_pnl = 0.0

    def update_capital(self, capital: float) -> None:
        """
        Update current capital and track peak for drawdown calculation.

        Should be called after each trade or periodically with current account balance.

        Args:
            capital: Current account capital
        """
        self._drawdown.current_capital = capital

        # Update peak if we have new high
        if capital > self._drawdown.peak_capital:
            self._drawdown.peak_capital = capital
            logger.debug(f"New peak capital: ¥{capital:,.0f}")

        # Check weekly/monthly resets
        self._check_and_reset_weekly(capital)
        self._check_and_reset_monthly(capital)

    def get_losing_streak_multiplier(self) -> float:
        """
        Calculate risk multiplier based on losing streak.

        The multiplier reduces position sizes during losing streaks
        and recovers after winning trades.

        Returns:
            Multiplier between min_risk_multiplier and 1.0
        """
        config = self.losing_streak_config
        losses = self._drawdown.consecutive_losses
        wins = self._drawdown.consecutive_wins

        # If we've had enough wins after a streak, full recovery
        if wins >= config.wins_to_recover:
            return 1.0

        # If not enough consecutive losses, no reduction
        if losses < config.start_reduction_at:
            return 1.0

        # Calculate reduction based on losses beyond threshold
        excess_losses = losses - config.start_reduction_at + 1
        reduction = excess_losses * config.reduction_per_loss
        multiplier = 1.0 - reduction

        # Apply floor
        multiplier = max(multiplier, config.min_risk_multiplier)

        if multiplier < 1.0:
            logger.info(
                f"Losing streak risk reduction: {losses} consecutive losses, "
                f"risk multiplier = {multiplier:.1%}"
            )

        return multiplier

    def add_trade_result(self, pnl: float, side: str, capital: float | None = None) -> None:
        """
        Record a trade result.

        Args:
            pnl: Profit/loss from the trade
            side: "BUY" (LONG) or "SELL" (SHORT)
            capital: Current capital after trade (for drawdown tracking)
        """
        self._check_and_reset_daily()

        # Update global stats
        self._daily_pnl += pnl
        self._daily_trades += 1

        # Update direction-specific stats
        if side == "BUY":
            stats = self._long_stats
            direction = "LONG"
        else:
            stats = self._short_stats
            direction = "SHORT"

        stats.trades += 1
        stats.pnl += pnl
        if pnl > 0:
            stats.wins += 1
            # Track consecutive wins for recovery
            self._drawdown.consecutive_wins += 1
            # Reset consecutive losses on win
            if self._drawdown.consecutive_losses > 0:
                logger.info(
                    f"Winning trade after {self._drawdown.consecutive_losses} losses, "
                    f"consecutive wins: {self._drawdown.consecutive_wins}"
                )
            self._drawdown.consecutive_losses = 0
        else:
            stats.losses += 1
            # Reset consecutive wins on loss
            self._drawdown.consecutive_wins = 0
            # Track consecutive losses
            self._drawdown.consecutive_losses += 1
            self._drawdown.max_consecutive_losses = max(
                self._drawdown.max_consecutive_losses,
                self._drawdown.consecutive_losses
            )

        # Update weekly/monthly PnL
        self._drawdown.weekly_pnl += pnl
        self._drawdown.monthly_pnl += pnl

        # Update capital tracking if provided
        if capital is not None:
            self.update_capital(capital)

        logger.info(
            f"{direction} trade recorded: PnL=¥{pnl:,.0f}, "
            f"{direction} stats: trades={stats.trades}, win_rate={stats.win_rate:.1%}, pnl=¥{stats.pnl:,.0f}"
        )

        # Log drawdown status if significant
        if self._drawdown.current_drawdown > 0.05:
            logger.warning(
                f"Drawdown alert: {self._drawdown.current_drawdown:.1%} from peak, "
                f"weekly: {self._drawdown.weekly_drawdown:.1%}, "
                f"monthly: {self._drawdown.monthly_drawdown:.1%}, "
                f"consecutive losses: {self._drawdown.consecutive_losses}"
            )

    def check_can_trade(self, capital: float, side: str) -> RiskCheck:
        """
        Check if trading is allowed for a specific direction.

        Args:
            capital: Current capital
            side: "BUY" (LONG) or "SELL" (SHORT)

        Returns:
            RiskCheck with allowed status and reason if not allowed
        """
        self._check_and_reset_daily()

        config = self.get_config(side)
        direction = "LONG" if side == "BUY" else "SHORT"
        stats = self._long_stats if side == "BUY" else self._short_stats

        # Check consecutive losses limit
        if self._drawdown.consecutive_losses >= self.max_consecutive_losses:
            return RiskCheck(
                allowed=False,
                reason=f"Max consecutive losses reached: {self._drawdown.consecutive_losses} >= {self.max_consecutive_losses}",
            )

        # Check max drawdown from peak
        if self._drawdown.current_drawdown >= self.max_drawdown_limit:
            return RiskCheck(
                allowed=False,
                reason=f"Max drawdown limit reached: {self._drawdown.current_drawdown:.2%} >= {self.max_drawdown_limit:.2%}",
            )

        # Check monthly loss limit
        if self._drawdown.monthly_drawdown >= self.monthly_loss_limit:
            return RiskCheck(
                allowed=False,
                reason=f"Monthly loss limit reached: {self._drawdown.monthly_drawdown:.2%} >= {self.monthly_loss_limit:.2%}",
            )

        # Check weekly loss limit
        if self._drawdown.weekly_drawdown >= self.weekly_loss_limit:
            return RiskCheck(
                allowed=False,
                reason=f"Weekly loss limit reached: {self._drawdown.weekly_drawdown:.2%} >= {self.weekly_loss_limit:.2%}",
            )

        # Check global daily loss limit
        daily_loss_ratio = abs(self._daily_pnl) / capital if capital > 0 else 0
        if self._daily_pnl < 0 and daily_loss_ratio >= self.daily_loss_limit:
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit reached: {daily_loss_ratio:.2%} >= {self.daily_loss_limit:.2%}",
            )

        # Check global max daily trades
        if self._daily_trades >= self.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Max daily trades reached: {self._daily_trades} >= {self.max_daily_trades}",
            )

        # Check direction-specific max trades
        if stats.trades >= config.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Max {direction} trades reached: {stats.trades} >= {config.max_daily_trades}",
            )

        return RiskCheck(allowed=True)

    def check_confidence(self, confidence: float, side: str) -> RiskCheck:
        """
        Check if confidence meets threshold for direction.

        Args:
            confidence: Model confidence
            side: "BUY" (LONG) or "SELL" (SHORT)

        Returns:
            RiskCheck with result
        """
        config = self.get_config(side)
        direction = "LONG" if side == "BUY" else "SHORT"

        if confidence < config.confidence_threshold:
            return RiskCheck(
                allowed=False,
                reason=f"{direction} confidence {confidence:.2%} below threshold {config.confidence_threshold:.2%}",
            )

        return RiskCheck(allowed=True)

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        atr: float,
        side: str,
        symbol: str | None = None,
    ) -> PositionSize:
        """
        Calculate position size based on direction-specific settings and allocation.

        Args:
            capital: Total available capital
            entry_price: Expected entry price
            atr: Current ATR value
            side: "BUY" (LONG) or "SELL" (SHORT)
            symbol: Trading symbol (for allocation-based sizing)

        Returns:
            PositionSize with size, stop loss, and risk details
        """
        config = self.get_config(side)
        direction = "LONG" if side == "BUY" else "SHORT"

        # Get allocated capital for this symbol/direction
        if symbol and self._symbol_allocations:
            allocated_capital = self.get_remaining_capital(symbol, direction, capital)
            if allocated_capital <= 0:
                logger.warning(
                    f"No remaining capital for {symbol} {direction} "
                    f"(allocation exhausted)"
                )
                return PositionSize(
                    size=0.0,
                    stop_loss=entry_price,
                    risk_amount=0.0,
                    position_value=0.0,
                )
        else:
            # Fallback to total capital if no allocation configured
            allocated_capital = capital

        # Calculate stop loss price using direction-specific ATR multiple
        stop_distance = atr * config.sl_atr_multiple

        if side == "BUY":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Get losing streak multiplier (reduces risk during losing streaks)
        streak_multiplier = self.get_losing_streak_multiplier()

        # Calculate risk amount using direction-specific risk per trade
        # Apply losing streak multiplier for dynamic risk reduction
        base_risk_per_trade = config.risk_per_trade
        adjusted_risk_per_trade = base_risk_per_trade * streak_multiplier
        risk_amount = allocated_capital * adjusted_risk_per_trade

        # Calculate position size based on risk
        size_by_risk = risk_amount / stop_distance if stop_distance > 0 else 0

        # Check direction-specific max position size limit (also apply streak multiplier)
        max_position_value = allocated_capital * config.max_position_size * streak_multiplier
        max_size = max_position_value / entry_price if entry_price > 0 else 0

        # Take the smaller of the two
        final_size = min(size_by_risk, max_size)
        position_value = final_size * entry_price

        logger.debug(
            f"{direction} position size: allocated_capital={allocated_capital:.0f}, "
            f"entry={entry_price:.0f}, atr={atr:.0f}, "
            f"risk_per_trade={adjusted_risk_per_trade:.2%} (base: {base_risk_per_trade:.2%}, streak mult: {streak_multiplier:.1%}), "
            f"sl_atr_mult={config.sl_atr_multiple}, "
            f"size={final_size:.6f}"
        )

        return PositionSize(
            size=final_size,
            stop_loss=stop_loss,
            risk_amount=min(risk_amount, stop_distance * final_size),
            position_value=position_value,
        )

    def get_take_profit_levels(self, side: str) -> list[tuple[float, float]]:
        """Get take profit levels for a direction."""
        config = self.get_config(side)
        return config.tp_levels

    def calculate_take_profit_prices(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
    ) -> list[tuple[float, float]]:
        """
        Calculate take profit price levels for a direction.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: "BUY" (LONG) or "SELL" (SHORT)

        Returns:
            List of (price, ratio) tuples for take profit levels
        """
        config = self.get_config(side)

        # Calculate 1R (risk in price terms)
        one_r = abs(entry_price - stop_loss)

        result = []
        for r_multiple, ratio in config.tp_levels:
            if side == "BUY":
                tp_price = entry_price + (one_r * r_multiple)
            else:
                tp_price = entry_price - (one_r * r_multiple)

            result.append((tp_price, ratio))

        return result

    def get_drawdown_state(self) -> dict:
        """Get current drawdown tracking state."""
        return {
            "peak_capital": self._drawdown.peak_capital,
            "current_capital": self._drawdown.current_capital,
            "current_drawdown": self._drawdown.current_drawdown,
            "weekly_pnl": self._drawdown.weekly_pnl,
            "weekly_drawdown": self._drawdown.weekly_drawdown,
            "weekly_limit": self.weekly_loss_limit,
            "monthly_pnl": self._drawdown.monthly_pnl,
            "monthly_drawdown": self._drawdown.monthly_drawdown,
            "monthly_limit": self.monthly_loss_limit,
            "max_drawdown_limit": self.max_drawdown_limit,
            "consecutive_losses": self._drawdown.consecutive_losses,
            "consecutive_wins": self._drawdown.consecutive_wins,
            "max_consecutive_losses_limit": self.max_consecutive_losses,
            "losing_streak_multiplier": self.get_losing_streak_multiplier(),
            "week_start_date": self._drawdown.week_start_date.isoformat() if self._drawdown.week_start_date else None,
            "month_start_date": self._drawdown.month_start_date.isoformat() if self._drawdown.month_start_date else None,
        }

    def get_daily_stats(self) -> dict:
        """Get current daily statistics including direction breakdown."""
        self._check_and_reset_daily()
        return {
            "date": self._last_reset_date.isoformat() if self._last_reset_date else "",
            "total": {
                "trades": self._daily_trades,
                "pnl": self._daily_pnl,
            },
            "long": {
                "trades": self._long_stats.trades,
                "wins": self._long_stats.wins,
                "losses": self._long_stats.losses,
                "pnl": self._long_stats.pnl,
                "win_rate": self._long_stats.win_rate,
            },
            "short": {
                "trades": self._short_stats.trades,
                "wins": self._short_stats.wins,
                "losses": self._short_stats.losses,
                "pnl": self._short_stats.pnl,
                "win_rate": self._short_stats.win_rate,
            },
            "drawdown": self.get_drawdown_state(),
        }

    def get_direction_performance_summary(self) -> str:
        """Get a formatted summary of direction performance."""
        stats = self.get_daily_stats()

        long = stats["long"]
        short = stats["short"]

        summary = f"""
Direction Performance (Today):
━━━━━━━━━━━━━━━━━━━━━━━━━━

LONG:
  Trades: {long['trades']}
  Win Rate: {long['win_rate']:.1%}
  PnL: ¥{long['pnl']:,.0f}

SHORT:
  Trades: {short['trades']}
  Win Rate: {short['win_rate']:.1%}
  PnL: ¥{short['pnl']:,.0f}

Total PnL: ¥{stats['total']['pnl']:,.0f}
"""
        return summary.strip()

    def validate_order(
        self,
        capital: float,
        price: float,
        size: float,
        side: str,
    ) -> RiskCheck:
        """
        Validate an order against direction-specific risk limits.

        Args:
            capital: Current capital
            price: Order price
            size: Order size
            side: "BUY" (LONG) or "SELL" (SHORT)

        Returns:
            RiskCheck with validation result
        """
        config = self.get_config(side)
        direction = "LONG" if side == "BUY" else "SHORT"

        # Check position size limit
        position_value = price * size
        position_ratio = position_value / capital if capital > 0 else float("inf")

        if position_ratio > config.max_position_size:
            return RiskCheck(
                allowed=False,
                reason=f"{direction} position size exceeds limit: {position_ratio:.2%} > {config.max_position_size:.2%}",
            )

        return RiskCheck(allowed=True)

    def update_runtime_settings(
        self,
        long_confidence_threshold: float | None = None,
        short_confidence_threshold: float | None = None,
        long_risk_per_trade: float | None = None,
        short_risk_per_trade: float | None = None,
        long_max_position_size: float | None = None,
        short_max_position_size: float | None = None,
        long_max_daily_trades: int | None = None,
        short_max_daily_trades: int | None = None,
        max_daily_trades: int | None = None,
        daily_loss_limit: float | None = None,
    ) -> dict[str, str]:
        """
        Update risk parameters at runtime.

        This method allows dynamic adjustment of risk parameters without restarting
        the trading engine. Used by the API and Meta AI Agent.

        Args:
            long_confidence_threshold: Confidence threshold for LONG trades
            short_confidence_threshold: Confidence threshold for SHORT trades
            long_risk_per_trade: Risk per trade for LONG positions
            short_risk_per_trade: Risk per trade for SHORT positions
            long_max_position_size: Max position size for LONG
            short_max_position_size: Max position size for SHORT
            long_max_daily_trades: Max daily trades for LONG
            short_max_daily_trades: Max daily trades for SHORT
            max_daily_trades: Global max daily trades
            daily_loss_limit: Daily loss limit

        Returns:
            Dict of parameter names and their update status
        """
        updates = {}

        # LONG config updates
        if long_confidence_threshold is not None:
            old = self.long_config.confidence_threshold
            self.long_config.confidence_threshold = long_confidence_threshold
            updates["long_confidence_threshold"] = f"{old:.2%} → {long_confidence_threshold:.2%}"

        if long_risk_per_trade is not None:
            old = self.long_config.risk_per_trade
            self.long_config.risk_per_trade = long_risk_per_trade
            updates["long_risk_per_trade"] = f"{old:.2%} → {long_risk_per_trade:.2%}"

        if long_max_position_size is not None:
            old = self.long_config.max_position_size
            self.long_config.max_position_size = long_max_position_size
            updates["long_max_position_size"] = f"{old:.2%} → {long_max_position_size:.2%}"

        if long_max_daily_trades is not None:
            old = self.long_config.max_daily_trades
            self.long_config.max_daily_trades = long_max_daily_trades
            updates["long_max_daily_trades"] = f"{old} → {long_max_daily_trades}"

        # SHORT config updates
        if short_confidence_threshold is not None:
            old = self.short_config.confidence_threshold
            self.short_config.confidence_threshold = short_confidence_threshold
            updates["short_confidence_threshold"] = f"{old:.2%} → {short_confidence_threshold:.2%}"

        if short_risk_per_trade is not None:
            old = self.short_config.risk_per_trade
            self.short_config.risk_per_trade = short_risk_per_trade
            updates["short_risk_per_trade"] = f"{old:.2%} → {short_risk_per_trade:.2%}"

        if short_max_position_size is not None:
            old = self.short_config.max_position_size
            self.short_config.max_position_size = short_max_position_size
            updates["short_max_position_size"] = f"{old:.2%} → {short_max_position_size:.2%}"

        if short_max_daily_trades is not None:
            old = self.short_config.max_daily_trades
            self.short_config.max_daily_trades = short_max_daily_trades
            updates["short_max_daily_trades"] = f"{old} → {short_max_daily_trades}"

        # Global updates
        if max_daily_trades is not None:
            old = self.max_daily_trades
            self.max_daily_trades = max_daily_trades
            updates["max_daily_trades"] = f"{old} → {max_daily_trades}"

        if daily_loss_limit is not None:
            old = self.daily_loss_limit
            self.daily_loss_limit = daily_loss_limit
            updates["daily_loss_limit"] = f"{old:.2%} → {daily_loss_limit:.2%}"

        if updates:
            logger.info(f"Runtime settings updated: {updates}")

        return updates
