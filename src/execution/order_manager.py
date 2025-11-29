"""Order execution and position management module."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.api.gmo_client import GMOCoinClient, Order


class PositionSide(str, Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"


class PositionStatus(str, Enum):
    """Position status."""

    PENDING = "PENDING"  # Order placed, waiting for fill
    OPEN = "OPEN"  # Position is open
    PARTIAL_TP = "PARTIAL_TP"  # Some take profits executed
    CLOSED = "CLOSED"  # Position fully closed


@dataclass
class TakeProfitLevel:
    """Take profit level configuration."""

    price: float
    ratio: float  # Fraction of position to close
    order_id: str | None = None
    filled: bool = False


@dataclass
class Position:
    """Trading position."""

    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    stop_loss: float
    take_profit_levels: list[TakeProfitLevel] = field(default_factory=list)
    entry_order_id: str | None = None
    stop_order_id: str | None = None
    status: PositionStatus = PositionStatus.PENDING
    entry_time: datetime = field(default_factory=datetime.now)
    realized_pnl: float = 0.0
    remaining_size: float = 0.0

    def __post_init__(self) -> None:
        """Initialize remaining size."""
        if self.remaining_size == 0.0:
            self.remaining_size = self.size


class OrderManager:
    """Manages order execution and position tracking."""

    def __init__(
        self,
        client: GMOCoinClient,
        symbol: str = "BTC_JPY",
    ) -> None:
        """
        Initialize order manager.

        Args:
            client: GMO Coin API client
            symbol: Trading symbol
        """
        self.client = client
        self.symbol = symbol
        self.current_position: Position | None = None

    def open_position(
        self,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit_levels: list[tuple[float, float]],
    ) -> Position | None:
        """
        Open a new position with maker order.

        Args:
            side: "BUY" or "SELL"
            size: Position size
            entry_price: Limit price for entry
            stop_loss: Stop loss price
            take_profit_levels: List of (price, ratio) tuples

        Returns:
            Position object or None if failed
        """
        if self.current_position is not None:
            logger.warning("Position already exists, cannot open new position")
            return None

        try:
            # Place entry order (maker)
            order = self.client.place_order(
                symbol=self.symbol,
                side=side,
                size=size,
                order_type="LIMIT",
                price=entry_price,
                time_in_force="GTC",
            )

            logger.info(
                f"Entry order placed: {side} {size} @ {entry_price}, order_id={order.order_id}"
            )

            # Create position object
            position = Position(
                symbol=self.symbol,
                side=PositionSide.LONG if side == "BUY" else PositionSide.SHORT,
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profit_levels=[
                    TakeProfitLevel(price=p, ratio=r) for p, r in take_profit_levels
                ],
                entry_order_id=order.order_id,
                status=PositionStatus.PENDING,
            )

            self.current_position = position
            return position

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    def open_margin_position(
        self,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit_levels: list[tuple[float, float]],
    ) -> Position | None:
        """
        Open a new margin position (for leverage/short trading).

        Args:
            side: "BUY" for long, "SELL" for short
            size: Position size
            entry_price: Limit price for entry
            stop_loss: Stop loss price
            take_profit_levels: List of (price, ratio) tuples

        Returns:
            Position object or None if failed
        """
        if self.current_position is not None:
            logger.warning("Position already exists, cannot open new position")
            return None

        try:
            # Place margin entry order
            order = self.client.place_margin_order(
                symbol=self.symbol,
                side=side,
                size=size,
                order_type="LIMIT",
                price=entry_price,
                time_in_force="GTC",
            )

            direction = "LONG" if side == "BUY" else "SHORT"
            logger.info(
                f"Margin {direction} order placed: {side} {size} @ {entry_price}, "
                f"order_id={order.order_id}"
            )

            # Create position object
            position = Position(
                symbol=self.symbol,
                side=PositionSide.LONG if side == "BUY" else PositionSide.SHORT,
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profit_levels=[
                    TakeProfitLevel(price=p, ratio=r) for p, r in take_profit_levels
                ],
                entry_order_id=order.order_id,
                status=PositionStatus.PENDING,
            )

            self.current_position = position
            return position

        except Exception as e:
            logger.error(f"Failed to open margin position: {e}")
            return None

    def check_entry_filled(self) -> bool:
        """
        Check if entry order is filled.

        Returns:
            True if filled, False otherwise
        """
        if self.current_position is None:
            return False

        if self.current_position.status != PositionStatus.PENDING:
            return True

        try:
            executions = self.client.get_executions(
                order_id=self.current_position.entry_order_id,
                symbol=self.symbol,
            )

            if executions:
                # Calculate average fill price
                total_value = sum(
                    float(e["price"]) * float(e["size"]) for e in executions
                )
                total_size = sum(float(e["size"]) for e in executions)

                if total_size >= self.current_position.size * 0.99:  # Allow 1% slippage
                    self.current_position.entry_price = total_value / total_size
                    self.current_position.status = PositionStatus.OPEN
                    logger.info(
                        f"Entry order filled at avg price {self.current_position.entry_price}"
                    )
                    return True

        except Exception as e:
            logger.error(f"Error checking entry fill: {e}")

        return False

    def place_take_profit_orders(self) -> None:
        """Place take profit orders for the current position."""
        if self.current_position is None:
            return

        if self.current_position.status != PositionStatus.OPEN:
            return

        position = self.current_position
        exit_side = "SELL" if position.side == PositionSide.LONG else "BUY"

        for tp_level in position.take_profit_levels:
            if tp_level.order_id is not None:
                continue

            tp_size = position.size * tp_level.ratio

            try:
                order = self.client.place_order(
                    symbol=self.symbol,
                    side=exit_side,
                    size=tp_size,
                    order_type="LIMIT",
                    price=tp_level.price,
                    time_in_force="GTC",
                )
                tp_level.order_id = order.order_id
                logger.info(
                    f"TP order placed: {exit_side} {tp_size} @ {tp_level.price}"
                )

            except Exception as e:
                logger.error(f"Failed to place TP order: {e}")

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss is triggered.

        Args:
            current_price: Current market price

        Returns:
            True if stop loss triggered
        """
        if self.current_position is None:
            return False

        position = self.current_position

        if position.side == PositionSide.LONG:
            if current_price <= position.stop_loss:
                logger.warning(
                    f"Stop loss triggered: price {current_price} <= SL {position.stop_loss}"
                )
                return True
        else:
            if current_price >= position.stop_loss:
                logger.warning(
                    f"Stop loss triggered: price {current_price} >= SL {position.stop_loss}"
                )
                return True

        return False

    def execute_stop_loss(self) -> float:
        """
        Execute stop loss (market order).

        Returns:
            Realized PnL
        """
        if self.current_position is None:
            return 0.0

        position = self.current_position
        exit_side = "SELL" if position.side == PositionSide.LONG else "BUY"

        try:
            # Cancel all pending TP orders
            self._cancel_tp_orders()

            # Place market order to close
            order = self.client.place_order(
                symbol=self.symbol,
                side=exit_side,
                size=position.remaining_size,
                order_type="MARKET",
            )

            logger.info(f"Stop loss executed: {exit_side} {position.remaining_size}")

            # Calculate approximate PnL (actual will come from executions)
            if position.side == PositionSide.LONG:
                pnl = (position.stop_loss - position.entry_price) * position.remaining_size
            else:
                pnl = (position.entry_price - position.stop_loss) * position.remaining_size

            position.realized_pnl += pnl
            position.status = PositionStatus.CLOSED
            position.remaining_size = 0.0

            self.current_position = None
            return pnl

        except Exception as e:
            logger.error(f"Failed to execute stop loss: {e}")
            return 0.0

    def check_take_profit_fills(self) -> float:
        """
        Check if any take profit orders are filled.

        Returns:
            Total realized PnL from filled TP orders
        """
        if self.current_position is None:
            return 0.0

        position = self.current_position
        total_pnl = 0.0

        for tp_level in position.take_profit_levels:
            if tp_level.filled or tp_level.order_id is None:
                continue

            try:
                executions = self.client.get_executions(
                    order_id=tp_level.order_id,
                    symbol=self.symbol,
                )

                if executions:
                    total_size = sum(float(e["size"]) for e in executions)
                    total_value = sum(
                        float(e["price"]) * float(e["size"]) for e in executions
                    )
                    avg_price = total_value / total_size if total_size > 0 else 0

                    if total_size > 0:
                        tp_level.filled = True

                        if position.side == PositionSide.LONG:
                            pnl = (avg_price - position.entry_price) * total_size
                        else:
                            pnl = (position.entry_price - avg_price) * total_size

                        total_pnl += pnl
                        position.remaining_size -= total_size
                        position.realized_pnl += pnl

                        logger.info(
                            f"TP filled: {total_size} @ {avg_price}, PnL={pnl:.2f}"
                        )

            except Exception as e:
                logger.error(f"Error checking TP fill: {e}")

        # Check if position is fully closed
        if position.remaining_size <= 0:
            position.status = PositionStatus.CLOSED
            self.current_position = None
            logger.info(f"Position fully closed. Total PnL: {position.realized_pnl:.2f}")

        return total_pnl

    def _cancel_tp_orders(self) -> None:
        """Cancel all pending take profit orders."""
        if self.current_position is None:
            return

        for tp_level in self.current_position.take_profit_levels:
            if tp_level.order_id and not tp_level.filled:
                try:
                    self.client.cancel_order(tp_level.order_id)
                    logger.info(f"TP order cancelled: {tp_level.order_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel TP order: {e}")

    def close_position_market(self) -> float:
        """
        Close current position with market order.

        Returns:
            Realized PnL
        """
        if self.current_position is None:
            return 0.0

        position = self.current_position
        exit_side = "SELL" if position.side == PositionSide.LONG else "BUY"

        try:
            # Cancel all pending orders
            self._cancel_tp_orders()

            # Get current price
            ticker = self.client.get_ticker(self.symbol)
            exit_price = ticker.bid if exit_side == "SELL" else ticker.ask

            # Place market order
            order = self.client.place_order(
                symbol=self.symbol,
                side=exit_side,
                size=position.remaining_size,
                order_type="MARKET",
            )

            # Calculate PnL
            if position.side == PositionSide.LONG:
                pnl = (exit_price - position.entry_price) * position.remaining_size
            else:
                pnl = (position.entry_price - exit_price) * position.remaining_size

            position.realized_pnl += pnl
            position.status = PositionStatus.CLOSED
            position.remaining_size = 0.0

            logger.info(f"Position closed at market. PnL: {pnl:.2f}")

            self.current_position = None
            return pnl

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return 0.0

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.current_position is not None

    def get_position_info(self) -> dict[str, Any] | None:
        """Get current position information."""
        if self.current_position is None:
            return None

        pos = self.current_position
        return {
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "size": pos.size,
            "remaining_size": pos.remaining_size,
            "stop_loss": pos.stop_loss,
            "status": pos.status.value,
            "realized_pnl": pos.realized_pnl,
            "entry_time": pos.entry_time.isoformat(),
        }
