"""Paper trading executor for simulation mode."""

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from src.utils.timezone import now_jst


class PaperOrderStatus(str, Enum):
    """Paper order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PaperPositionSide(str, Enum):
    """Paper position side."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PaperOrder:
    """Represents a paper (simulated) order."""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    size: float
    price: float
    order_type: str  # LIMIT, MARKET
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    filled_size: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=now_jst)
    filled_at: datetime | None = None

    @property
    def is_filled(self) -> bool:
        return self.status == PaperOrderStatus.FILLED

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size


@dataclass
class PaperPosition:
    """Represents a paper (simulated) position."""
    position_id: str
    symbol: str
    side: PaperPositionSide
    entry_price: float
    size: float
    stop_loss: float
    take_profit_levels: list[tuple[float, float]] = field(default_factory=list)
    entry_time: datetime = field(default_factory=now_jst)
    exit_time: datetime | None = None
    exit_price: float | None = None
    realized_pnl: float = 0.0
    status: str = "OPEN"
    remaining_size: float = 0.0
    confidence: float = 0.0

    def __post_init__(self):
        self.remaining_size = self.size

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.side == PaperPositionSide.LONG:
            return (current_price - self.entry_price) * self.remaining_size
        else:
            return (self.entry_price - current_price) * self.remaining_size


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading simulation."""
    # Initial virtual capital (JPY)
    initial_capital: float = 1_000_000

    # Slippage simulation (basis points)
    slippage_bps: float = 5.0  # 0.05%

    # Fill probability for limit orders (0.0-1.0)
    limit_fill_probability: float = 0.95

    # Latency simulation (seconds)
    simulated_latency: float = 0.1

    # Commission rate (maker/taker)
    maker_fee: float = -0.0001  # GMO gives rebate for maker
    taker_fee: float = 0.0004   # 0.04% for taker

    # Spread simulation (basis points)
    spread_bps: float = 10.0  # 0.1%

    # Realistic rejection probability
    rejection_probability: float = 0.02  # 2% orders may be rejected


class PaperTradingExecutor:
    """Simulates order execution for paper trading mode."""

    def __init__(self, config: PaperTradingConfig | None = None) -> None:
        """
        Initialize paper trading executor.

        Args:
            config: Paper trading configuration
        """
        self.config = config or PaperTradingConfig()

        # Virtual capital tracking
        self.initial_capital = self.config.initial_capital
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        # Tracking
        self.positions: dict[str, PaperPosition] = {}
        self.orders: dict[str, PaperOrder] = {}
        self.closed_positions: list[PaperPosition] = []
        self.trade_history: list[dict] = []

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0

        # Session tracking
        self.session_start = now_jst()
        self.last_update = now_jst()

        # Order ID counter
        self._order_counter = 0

        logger.info(
            f"Paper trading executor initialized. "
            f"Capital: ¥{self.initial_capital:,.0f}"
        )

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PAPER-{self.session_start.strftime('%Y%m%d')}-{self._order_counter:06d}"

    def _simulate_slippage(self, price: float, side: str) -> float:
        """Simulate realistic slippage."""
        slippage_pct = self.config.slippage_bps / 10000

        # Slippage is always against the trader
        if side == "BUY":
            return price * (1 + slippage_pct * random.uniform(0, 1))
        else:
            return price * (1 - slippage_pct * random.uniform(0, 1))

    def _simulate_spread(self, mid_price: float, side: str) -> float:
        """Simulate bid/ask spread."""
        half_spread = (self.config.spread_bps / 10000) / 2

        if side == "BUY":
            return mid_price * (1 + half_spread)  # Ask
        else:
            return mid_price * (1 - half_spread)  # Bid

    def _calculate_commission(self, size: float, price: float, is_maker: bool) -> float:
        """Calculate commission for a trade."""
        notional = size * price
        rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return notional * rate

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit_levels: list[tuple[float, float]] | None = None,
        confidence: float = 0.0,
        use_limit_order: bool = True,
    ) -> PaperPosition | None:
        """
        Open a paper position.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            size: Position size
            entry_price: Target entry price
            stop_loss: Stop loss price
            take_profit_levels: List of (price, ratio) tuples
            confidence: Model confidence
            use_limit_order: Whether to use limit order (maker)

        Returns:
            PaperPosition if successful, None otherwise
        """
        # Check for rejection
        if random.random() < self.config.rejection_probability:
            logger.warning(f"Paper order rejected (simulated rejection)")
            return None

        # Simulate fill price
        if use_limit_order:
            # Limit order - check if it would fill
            if random.random() > self.config.limit_fill_probability:
                logger.info(f"Paper limit order not filled (simulated)")
                return None
            # Limit orders fill at specified price (maker)
            fill_price = entry_price
            is_maker = True
        else:
            # Market order - add slippage
            fill_price = self._simulate_slippage(entry_price, side)
            is_maker = False

        # Calculate commission
        commission = self._calculate_commission(size, fill_price, is_maker)
        self.total_commission += commission
        self.current_capital -= commission

        # Determine position side
        position_side = PaperPositionSide.LONG if side == "BUY" else PaperPositionSide.SHORT

        # Create position
        position_id = self._generate_order_id()
        position = PaperPosition(
            position_id=position_id,
            symbol=symbol,
            side=position_side,
            entry_price=fill_price,
            size=size,
            stop_loss=stop_loss,
            take_profit_levels=take_profit_levels or [],
            confidence=confidence,
        )

        self.positions[position_id] = position

        # Record trade entry
        self.trade_history.append({
            "type": "ENTRY",
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": fill_price,
            "stop_loss": stop_loss,
            "commission": commission,
            "timestamp": now_jst(),
            "confidence": confidence,
        })

        logger.info(
            f"Paper position opened: {position_side.value} {symbol} "
            f"size={size:.6f} @ ¥{fill_price:,.0f} "
            f"(commission: ¥{commission:,.0f})"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "MANUAL",
        partial_ratio: float = 1.0,
    ) -> float:
        """
        Close a paper position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            reason: Reason for closing (TP, SL, MANUAL, etc.)
            partial_ratio: Ratio of position to close (1.0 = full)

        Returns:
            Realized PnL
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return 0.0

        position = self.positions[position_id]

        # Calculate close size
        close_size = position.remaining_size * partial_ratio

        # Simulate slippage on exit
        actual_exit_price = self._simulate_slippage(
            exit_price,
            "SELL" if position.side == PaperPositionSide.LONG else "BUY"
        )

        # Calculate PnL
        if position.side == PaperPositionSide.LONG:
            pnl = (actual_exit_price - position.entry_price) * close_size
        else:
            pnl = (position.entry_price - actual_exit_price) * close_size

        # Commission on exit (usually taker for SL/TP)
        commission = self._calculate_commission(close_size, actual_exit_price, is_maker=False)
        self.total_commission += commission

        # Net PnL
        net_pnl = pnl - commission

        # Update position
        position.remaining_size -= close_size
        position.realized_pnl += net_pnl

        # Update capital
        self.current_capital += net_pnl
        self.total_pnl += net_pnl

        # Track peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Record trade exit
        self.trade_history.append({
            "type": "EXIT",
            "position_id": position_id,
            "symbol": position.symbol,
            "side": "SELL" if position.side == PaperPositionSide.LONG else "BUY",
            "size": close_size,
            "price": actual_exit_price,
            "pnl": net_pnl,
            "reason": reason,
            "commission": commission,
            "timestamp": now_jst(),
        })

        # Check if fully closed
        if position.remaining_size <= 0:
            position.status = "CLOSED"
            position.exit_time = now_jst()
            position.exit_price = actual_exit_price

            # Track statistics
            self.total_trades += 1
            if position.realized_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]

        logger.info(
            f"Paper position {'partially' if partial_ratio < 1 else 'fully'} closed: "
            f"{position.side.value} {position.symbol} "
            f"size={close_size:.6f} @ ¥{actual_exit_price:,.0f} "
            f"PnL: ¥{net_pnl:+,.0f} ({reason})"
        )

        return net_pnl

    def check_stop_losses(self, current_prices: dict[str, float]) -> list[dict]:
        """
        Check and execute stop losses.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            List of triggered stop losses
        """
        triggered = []

        for position_id, position in list(self.positions.items()):
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]

            # Check stop loss
            if position.side == PaperPositionSide.LONG:
                if current_price <= position.stop_loss:
                    pnl = self.close_position(position_id, position.stop_loss, "SL")
                    triggered.append({
                        "position_id": position_id,
                        "symbol": position.symbol,
                        "side": position.side.value,
                        "pnl": pnl,
                        "reason": "SL",
                    })
            else:  # SHORT
                if current_price >= position.stop_loss:
                    pnl = self.close_position(position_id, position.stop_loss, "SL")
                    triggered.append({
                        "position_id": position_id,
                        "symbol": position.symbol,
                        "side": position.side.value,
                        "pnl": pnl,
                        "reason": "SL",
                    })

        return triggered

    def check_take_profits(self, current_prices: dict[str, float]) -> list[dict]:
        """
        Check and execute take profits.

        Args:
            current_prices: Dict of symbol -> current price

        Returns:
            List of triggered take profits
        """
        triggered = []

        for position_id, position in list(self.positions.items()):
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]

            # Check each TP level
            for tp_price, tp_ratio in position.take_profit_levels[:]:
                hit = False

                if position.side == PaperPositionSide.LONG:
                    if current_price >= tp_price:
                        hit = True
                else:  # SHORT
                    if current_price <= tp_price:
                        hit = True

                if hit:
                    pnl = self.close_position(position_id, tp_price, "TP", partial_ratio=tp_ratio)
                    triggered.append({
                        "position_id": position_id,
                        "symbol": position.symbol,
                        "side": position.side.value,
                        "pnl": pnl,
                        "reason": "TP",
                        "tp_level": tp_price,
                    })
                    # Remove triggered TP level
                    position.take_profit_levels.remove((tp_price, tp_ratio))

        return triggered

    def get_open_positions(self) -> list[PaperPosition]:
        """Get list of open positions."""
        return list(self.positions.values())

    def get_position(self, position_id: str) -> PaperPosition | None:
        """Get a specific position."""
        return self.positions.get(position_id)

    def has_open_position(self, symbol: str | None = None) -> bool:
        """Check if there are open positions."""
        if symbol is None:
            return len(self.positions) > 0
        return any(p.symbol == symbol for p in self.positions.values())

    def get_capital(self) -> float:
        """Get current virtual capital."""
        return self.current_capital

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Calculate total unrealized PnL."""
        total = 0.0
        for position in self.positions.values():
            if position.symbol in current_prices:
                total += position.calculate_unrealized_pnl(current_prices[position.symbol])
        return total

    def get_statistics(self) -> dict:
        """Get paper trading statistics."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Calculate profit factor
        gross_profit = sum(
            p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0
        )
        gross_loss = abs(sum(
            p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0
        ))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0.0

        # Calculate return
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # Session duration
        duration = now_jst() - self.session_start

        return {
            "session_start": self.session_start.isoformat(),
            "duration_hours": duration.total_seconds() / 3600,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_pnl": self.total_pnl,
            "total_return_pct": total_return * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": drawdown * 100,
            "total_commission": self.total_commission,
            "open_positions": len(self.positions),
        }

    def get_trade_summary(self) -> dict:
        """Get trade summary for reporting."""
        stats = self.get_statistics()

        # Group by direction
        long_trades = [p for p in self.closed_positions if p.side == PaperPositionSide.LONG]
        short_trades = [p for p in self.closed_positions if p.side == PaperPositionSide.SHORT]

        long_pnl = sum(p.realized_pnl for p in long_trades)
        short_pnl = sum(p.realized_pnl for p in short_trades)

        long_wins = sum(1 for p in long_trades if p.realized_pnl > 0)
        short_wins = sum(1 for p in short_trades if p.realized_pnl > 0)

        return {
            **stats,
            "long_trades": len(long_trades),
            "long_pnl": long_pnl,
            "long_win_rate": long_wins / len(long_trades) if long_trades else 0.0,
            "short_trades": len(short_trades),
            "short_pnl": short_pnl,
            "short_win_rate": short_wins / len(short_trades) if short_trades else 0.0,
            "avg_trade_pnl": self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0,
        }

    def reset(self) -> None:
        """Reset paper trading state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.closed_positions.clear()
        self.trade_history.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.session_start = now_jst()
        self._order_counter = 0

        logger.info("Paper trading state reset")

    def export_history(self) -> list[dict]:
        """Export trade history for analysis."""
        return self.trade_history.copy()

    def save_state(self, filepath: str) -> None:
        """Save paper trading state to file."""
        import json

        state = {
            "session_start": self.session_start.isoformat(),
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_commission": self.total_commission,
            "trade_history": [
                {**t, "timestamp": t["timestamp"].isoformat()}
                for t in self.trade_history
            ],
            "closed_positions": [
                {
                    "position_id": p.position_id,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "size": p.size,
                    "realized_pnl": p.realized_pnl,
                    "entry_time": p.entry_time.isoformat(),
                    "exit_time": p.exit_time.isoformat() if p.exit_time else None,
                    "confidence": p.confidence,
                }
                for p in self.closed_positions
            ],
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Paper trading state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """Load paper trading state from file."""
        import json

        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            self.session_start = datetime.fromisoformat(state["session_start"])
            self.initial_capital = state["initial_capital"]
            self.current_capital = state["current_capital"]
            self.peak_capital = state["peak_capital"]
            self.total_trades = state["total_trades"]
            self.winning_trades = state["winning_trades"]
            self.losing_trades = state["losing_trades"]
            self.total_pnl = state["total_pnl"]
            self.total_commission = state["total_commission"]

            # Reconstruct trade history
            self.trade_history = [
                {**t, "timestamp": datetime.fromisoformat(t["timestamp"])}
                for t in state["trade_history"]
            ]

            # Reconstruct closed positions
            self.closed_positions = [
                PaperPosition(
                    position_id=p["position_id"],
                    symbol=p["symbol"],
                    side=PaperPositionSide(p["side"]),
                    entry_price=p["entry_price"],
                    size=p["size"],
                    stop_loss=0,  # Not needed for closed positions
                    entry_time=datetime.fromisoformat(p["entry_time"]),
                    exit_time=datetime.fromisoformat(p["exit_time"]) if p["exit_time"] else None,
                    exit_price=p["exit_price"],
                    realized_pnl=p["realized_pnl"],
                    status="CLOSED",
                    confidence=p.get("confidence", 0.0),
                )
                for p in state["closed_positions"]
            ]

            logger.info(f"Paper trading state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load paper trading state: {e}")
            return False
