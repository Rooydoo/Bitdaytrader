"""Risk management module with direction-specific settings."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from loguru import logger


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
        # LONG settings
        long_risk_per_trade: float = 0.02,
        long_max_position_size: float = 0.10,
        long_max_daily_trades: int = 3,
        long_confidence_threshold: float = 0.65,
        long_sl_atr_multiple: float = 2.0,
        long_tp_levels: list[tuple[float, float]] | None = None,
        # SHORT settings (stricter by default)
        short_risk_per_trade: float = 0.015,
        short_max_position_size: float = 0.07,
        short_max_daily_trades: int = 2,
        short_confidence_threshold: float = 0.70,
        short_sl_atr_multiple: float = 1.5,
        short_tp_levels: list[tuple[float, float]] | None = None,
        # Global settings
        daily_loss_limit: float = 0.03,
        max_daily_trades: int = 5,
    ) -> None:
        """
        Initialize risk manager with direction-specific settings.

        Args:
            long_*: Settings for LONG positions
            short_*: Settings for SHORT positions (stricter defaults)
            daily_loss_limit: Max daily loss as fraction of capital
            max_daily_trades: Max total trades per day
        """
        # LONG configuration
        self.long_config = DirectionConfig(
            risk_per_trade=long_risk_per_trade,
            max_position_size=long_max_position_size,
            max_daily_trades=long_max_daily_trades,
            confidence_threshold=long_confidence_threshold,
            sl_atr_multiple=long_sl_atr_multiple,
            tp_levels=long_tp_levels or [(1.5, 0.5), (2.5, 0.3), (4.0, 0.2)],
        )

        # SHORT configuration (stricter)
        self.short_config = DirectionConfig(
            risk_per_trade=short_risk_per_trade,
            max_position_size=short_max_position_size,
            max_daily_trades=short_max_daily_trades,
            confidence_threshold=short_confidence_threshold,
            sl_atr_multiple=short_sl_atr_multiple,
            tp_levels=short_tp_levels or [(1.0, 0.5), (1.5, 0.3), (2.5, 0.2)],
        )

        # Global limits
        self.daily_loss_limit = daily_loss_limit
        self.max_daily_trades = max_daily_trades

        # Daily tracking - total
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date | None = None

        # Daily tracking - by direction
        self._long_stats = DirectionStats()
        self._short_stats = DirectionStats()

        # Conservative mode (activated when overfitting detected)
        self._conservative_mode = False
        self._conservative_multiplier = 0.5  # Reduce risk by 50%
        self._original_long_config: DirectionConfig | None = None
        self._original_short_config: DirectionConfig | None = None

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

    def add_trade_result(self, pnl: float, side: str) -> None:
        """
        Record a trade result.

        Args:
            pnl: Profit/loss from the trade
            side: "BUY" (LONG) or "SELL" (SHORT)
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
        else:
            stats.losses += 1

        logger.info(
            f"{direction} trade recorded: PnL={pnl:.2f}, "
            f"{direction} stats: trades={stats.trades}, win_rate={stats.win_rate:.1%}, pnl={stats.pnl:.2f}"
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
    ) -> PositionSize:
        """
        Calculate position size based on direction-specific settings.

        Args:
            capital: Available capital
            entry_price: Expected entry price
            atr: Current ATR value
            side: "BUY" (LONG) or "SELL" (SHORT)

        Returns:
            PositionSize with size, stop loss, and risk details
        """
        config = self.get_config(side)
        direction = "LONG" if side == "BUY" else "SHORT"

        # Calculate stop loss price using direction-specific ATR multiple
        stop_distance = atr * config.sl_atr_multiple

        if side == "BUY":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Calculate risk amount using direction-specific risk per trade
        risk_amount = capital * config.risk_per_trade

        # Calculate position size based on risk
        size_by_risk = risk_amount / stop_distance

        # Check direction-specific max position size limit
        max_position_value = capital * config.max_position_size
        max_size = max_position_value / entry_price

        # Take the smaller of the two
        final_size = min(size_by_risk, max_size)
        position_value = final_size * entry_price

        logger.debug(
            f"{direction} position size: capital={capital:.0f}, "
            f"entry={entry_price:.0f}, atr={atr:.0f}, "
            f"risk_per_trade={config.risk_per_trade:.2%}, "
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
