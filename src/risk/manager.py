"""Risk management module."""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from loguru import logger


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


class RiskManager:
    """Risk management for trading operations."""

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        daily_loss_limit: float = 0.03,
        max_position_size: float = 0.10,
        max_daily_trades: int = 5,
        sl_atr_multiple: float = 2.0,
    ) -> None:
        """
        Initialize risk manager.

        Args:
            risk_per_trade: Max loss per trade as fraction of capital (2% = 0.02)
            daily_loss_limit: Max daily loss as fraction of capital (3% = 0.03)
            max_position_size: Max position size as fraction of capital (10% = 0.10)
            max_daily_trades: Maximum number of trades per day
            sl_atr_multiple: Stop loss in ATR multiples
        """
        self.risk_per_trade = risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.sl_atr_multiple = sl_atr_multiple

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset_date: date | None = None

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date = date.today()
        logger.info("Daily risk counters reset")

    def _check_and_reset_daily(self) -> None:
        """Check if we need to reset daily counters."""
        today = date.today()
        if self._last_reset_date != today:
            self.reset_daily()

    def add_trade_result(self, pnl: float) -> None:
        """
        Record a trade result.

        Args:
            pnl: Profit/loss from the trade (positive or negative)
        """
        self._check_and_reset_daily()
        self._daily_pnl += pnl
        self._daily_trades += 1
        logger.info(
            f"Trade recorded: PnL={pnl:.2f}, Daily PnL={self._daily_pnl:.2f}, "
            f"Trades today={self._daily_trades}"
        )

    def check_can_trade(self, capital: float) -> RiskCheck:
        """
        Check if trading is allowed based on risk limits.

        Args:
            capital: Current capital

        Returns:
            RiskCheck with allowed status and reason if not allowed
        """
        self._check_and_reset_daily()

        # Check daily loss limit
        daily_loss_ratio = abs(self._daily_pnl) / capital if capital > 0 else 0
        if self._daily_pnl < 0 and daily_loss_ratio >= self.daily_loss_limit:
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit reached: {daily_loss_ratio:.2%} >= {self.daily_loss_limit:.2%}",
            )

        # Check max daily trades
        if self._daily_trades >= self.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Max daily trades reached: {self._daily_trades} >= {self.max_daily_trades}",
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
        Calculate position size based on ATR-based stop loss.

        Args:
            capital: Available capital
            entry_price: Expected entry price
            atr: Current ATR value
            side: "BUY" or "SELL"

        Returns:
            PositionSize with size, stop loss, and risk details
        """
        # Calculate stop loss price
        stop_distance = atr * self.sl_atr_multiple

        if side == "BUY":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Calculate risk amount
        risk_amount = capital * self.risk_per_trade

        # Calculate position size based on risk
        # Risk = Position Size * Stop Distance
        # Position Size = Risk / Stop Distance
        size_by_risk = risk_amount / stop_distance

        # Check max position size limit
        max_position_value = capital * self.max_position_size
        max_size = max_position_value / entry_price

        # Take the smaller of the two
        final_size = min(size_by_risk, max_size)
        position_value = final_size * entry_price

        logger.debug(
            f"Position size calculation: capital={capital:.0f}, "
            f"entry={entry_price:.0f}, atr={atr:.0f}, "
            f"size_by_risk={size_by_risk:.6f}, max_size={max_size:.6f}, "
            f"final_size={final_size:.6f}"
        )

        return PositionSize(
            size=final_size,
            stop_loss=stop_loss,
            risk_amount=min(risk_amount, stop_distance * final_size),
            position_value=position_value,
        )

    def calculate_take_profit_levels(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
        tp_levels: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Calculate take profit price levels.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: "BUY" or "SELL"
            tp_levels: List of (R-multiple, ratio) tuples

        Returns:
            List of (price, ratio) tuples for take profit levels
        """
        # Calculate 1R (risk in price terms)
        one_r = abs(entry_price - stop_loss)

        result = []
        for r_multiple, ratio in tp_levels:
            if side == "BUY":
                tp_price = entry_price + (one_r * r_multiple)
            else:
                tp_price = entry_price - (one_r * r_multiple)

            result.append((tp_price, ratio))

        return result

    def get_daily_stats(self) -> dict[str, float | int]:
        """Get current daily statistics."""
        self._check_and_reset_daily()
        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "date": self._last_reset_date.isoformat() if self._last_reset_date else "",
        }

    def validate_order(
        self,
        capital: float,
        price: float,
        size: float,
    ) -> RiskCheck:
        """
        Validate an order against risk limits.

        Args:
            capital: Current capital
            price: Order price
            size: Order size

        Returns:
            RiskCheck with validation result
        """
        # Check position size limit
        position_value = price * size
        position_ratio = position_value / capital if capital > 0 else float("inf")

        if position_ratio > self.max_position_size:
            return RiskCheck(
                allowed=False,
                reason=f"Position size exceeds limit: {position_ratio:.2%} > {self.max_position_size:.2%}",
            )

        return RiskCheck(allowed=True)
