"""Perception module for Meta AI Agent - gathers current state information."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from loguru import logger

from src.utils.timezone import now_jst, JST


@dataclass
class MarketState:
    """Current market state."""

    symbol: str
    current_price: float
    price_change_1h: float
    price_change_24h: float
    volatility: float
    atr: float
    trend: str  # "up", "down", "sideways"
    volume_ratio: float
    timestamp: datetime = field(default_factory=now_jst)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "volatility": self.volatility,
            "atr": self.atr,
            "trend": self.trend,
            "volume_ratio": self.volume_ratio,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SignalRecord:
    """A signal record with outcome if available."""

    id: int
    timestamp: datetime
    direction: str
    confidence: float
    price: float
    features: dict
    executed: bool
    outcome: str | None = None  # "correct", "incorrect", "pending"
    actual_move: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "confidence": self.confidence,
            "price": self.price,
            "features": self.features,
            "executed": self.executed,
            "outcome": self.outcome,
            "actual_move": self.actual_move,
        }


@dataclass
class TradeRecord:
    """A trade record with analysis."""

    id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None
    size: float
    pnl: float | None
    entry_time: datetime
    exit_time: datetime | None
    status: str
    stop_loss: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
        }


@dataclass
class SystemHealth:
    """System health status."""

    status: str  # "healthy", "degraded", "unhealthy"
    engine_running: bool
    database_ok: bool
    gmo_api_ok: bool
    memory_percent: float
    cpu_percent: float
    uptime_hours: float
    emergency_stop_active: bool
    long_stopped: bool
    short_stopped: bool
    last_check: datetime = field(default_factory=now_jst)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "engine_running": self.engine_running,
            "database_ok": self.database_ok,
            "gmo_api_ok": self.gmo_api_ok,
            "memory_percent": self.memory_percent,
            "cpu_percent": self.cpu_percent,
            "uptime_hours": self.uptime_hours,
            "emergency_stop_active": self.emergency_stop_active,
            "long_stopped": self.long_stopped,
            "short_stopped": self.short_stopped,
            "last_check": self.last_check.isoformat(),
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    trades_count: int
    drawdown: float
    consecutive_losses: int
    consecutive_wins: int
    long_win_rate: float
    short_win_rate: float
    capital: float

    def to_dict(self) -> dict:
        return {
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "monthly_pnl": self.monthly_pnl,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "trades_count": self.trades_count,
            "drawdown": self.drawdown,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "long_win_rate": self.long_win_rate,
            "short_win_rate": self.short_win_rate,
            "capital": self.capital,
        }


@dataclass
class AgentContext:
    """Full context for agent decision making."""

    market: MarketState | None
    recent_signals: list[SignalRecord]
    recent_trades: list[TradeRecord]
    open_positions: list[TradeRecord]
    system_health: SystemHealth | None
    performance: PerformanceMetrics | None
    current_time: datetime
    last_decision_time: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "market": self.market.to_dict() if self.market else None,
            "recent_signals": [s.to_dict() for s in self.recent_signals],
            "recent_trades": [t.to_dict() for t in self.recent_trades],
            "open_positions": [p.to_dict() for p in self.open_positions],
            "system_health": self.system_health.to_dict() if self.system_health else None,
            "performance": self.performance.to_dict() if self.performance else None,
            "current_time": self.current_time.isoformat(),
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
        }

    def to_prompt(self) -> str:
        """Convert context to a prompt string for Claude."""
        lines = []

        # Current time
        lines.append(f"### 現在時刻: {self.current_time.strftime('%Y-%m-%d %H:%M:%S JST')}")
        lines.append("")

        # Market state
        if self.market:
            lines.append("### 市場状態")
            lines.append(f"- 銘柄: {self.market.symbol}")
            lines.append(f"- 現在価格: ¥{self.market.current_price:,.0f}")
            lines.append(f"- 1時間変動: {self.market.price_change_1h:+.2%}")
            lines.append(f"- 24時間変動: {self.market.price_change_24h:+.2%}")
            lines.append(f"- ボラティリティ: {self.market.volatility:.4f}")
            lines.append(f"- ATR: ¥{self.market.atr:,.0f}")
            lines.append(f"- トレンド: {self.market.trend}")
            lines.append(f"- 出来高比率: {self.market.volume_ratio:.2f}")
            lines.append("")

        # System health
        if self.system_health:
            lines.append("### システム状態")
            lines.append(f"- 状態: {self.system_health.status}")
            lines.append(f"- エンジン稼働: {'はい' if self.system_health.engine_running else 'いいえ'}")
            lines.append(f"- 緊急停止: {'発動中' if self.system_health.emergency_stop_active else '正常'}")
            if self.system_health.long_stopped:
                lines.append("- LONG: 停止中")
            if self.system_health.short_stopped:
                lines.append("- SHORT: 停止中")
            lines.append(f"- メモリ使用率: {self.system_health.memory_percent:.1f}%")
            lines.append(f"- CPU使用率: {self.system_health.cpu_percent:.1f}%")
            lines.append("")

        # Performance
        if self.performance:
            lines.append("### パフォーマンス")
            lines.append(f"- 資本: ¥{self.performance.capital:,.0f}")
            lines.append(f"- 本日損益: ¥{self.performance.daily_pnl:+,.0f}")
            lines.append(f"- 週間損益: ¥{self.performance.weekly_pnl:+,.0f}")
            lines.append(f"- 月間損益: ¥{self.performance.monthly_pnl:+,.0f}")
            lines.append(f"- 勝率: {self.performance.win_rate:.1%}")
            lines.append(f"- プロフィットファクター: {self.performance.profit_factor:.2f}")
            lines.append(f"- ドローダウン: {self.performance.drawdown:.1%}")
            lines.append(f"- 連敗: {self.performance.consecutive_losses}回")
            lines.append(f"- 連勝: {self.performance.consecutive_wins}回")
            lines.append(f"- LONG勝率: {self.performance.long_win_rate:.1%}")
            lines.append(f"- SHORT勝率: {self.performance.short_win_rate:.1%}")
            lines.append("")

        # Open positions
        if self.open_positions:
            lines.append("### オープンポジション")
            for pos in self.open_positions:
                lines.append(f"- {pos.side} {pos.symbol}: ¥{pos.entry_price:,.0f} x {pos.size:.6f}")
            lines.append("")

        # Recent signals
        if self.recent_signals:
            lines.append("### 直近のシグナル（過去24時間）")
            for sig in self.recent_signals[-10:]:  # Last 10
                outcome_str = f"結果: {sig.outcome}" if sig.outcome else "結果: 未確定"
                lines.append(
                    f"- {sig.timestamp.strftime('%H:%M')} {sig.direction} "
                    f"(確信度: {sig.confidence:.1%}) {outcome_str}"
                )
            lines.append("")

        # Recent trades
        if self.recent_trades:
            lines.append("### 直近の取引（過去24時間）")
            for trade in self.recent_trades[-10:]:  # Last 10
                pnl_str = f"損益: ¥{trade.pnl:+,.0f}" if trade.pnl is not None else "未決済"
                lines.append(
                    f"- {trade.entry_time.strftime('%H:%M')} {trade.side} "
                    f"¥{trade.entry_price:,.0f} → {pnl_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def needs_attention(self) -> bool:
        """Check if current context requires agent attention."""
        # Emergency conditions
        if self.system_health:
            if self.system_health.status == "unhealthy":
                return True
            if self.system_health.emergency_stop_active:
                return True

        # Performance issues
        if self.performance:
            if self.performance.consecutive_losses >= 3:
                return True
            if self.performance.drawdown >= 0.10:  # 10% drawdown
                return True

        # Market volatility
        if self.market:
            if abs(self.market.price_change_1h) >= 0.03:  # 3% in 1 hour
                return True

        return False


class PerceptionModule:
    """Module for gathering current state information."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8088",
        gmo_client: Any = None,
        db_session: Any = None,
    ) -> None:
        """
        Initialize perception module.

        Args:
            api_base_url: Base URL for trading bot API
            gmo_client: GMO API client instance (optional)
            db_session: Database session (optional)
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.gmo_client = gmo_client
        self.db_session = db_session
        self._http_session: aiohttp.ClientSession | None = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def get_context(self) -> AgentContext:
        """
        Gather full context for agent decision making.

        Returns:
            AgentContext with all current state information
        """
        # Gather all data in parallel where possible
        market = await self.get_market_state()
        signals = await self.get_recent_signals()
        trades = await self.get_recent_trades()
        positions = await self.get_open_positions()
        health = await self.get_system_health()
        performance = await self.get_performance_metrics()

        return AgentContext(
            market=market,
            recent_signals=signals,
            recent_trades=trades,
            open_positions=positions,
            system_health=health,
            performance=performance,
            current_time=now_jst(),
        )

    async def get_market_state(self, symbol: str = "BTC_JPY") -> MarketState | None:
        """Get current market state."""
        try:
            session = await self._get_http_session()

            # Get ticker from GMO client if available
            if self.gmo_client:
                ticker = self.gmo_client.get_ticker(symbol)
                current_price = ticker.last
            else:
                # Try to get from API
                async with session.get(f"{self.api_base_url}/api/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        current_price = data.get("checks", {}).get("gmo_api", {}).get("btc_price", 0)
                    else:
                        return None

            # For now, return basic market state
            # TODO: Calculate actual values from OHLCV data
            return MarketState(
                symbol=symbol,
                current_price=current_price or 0,
                price_change_1h=0.0,
                price_change_24h=0.0,
                volatility=0.0,
                atr=0.0,
                trend="sideways",
                volume_ratio=1.0,
            )

        except Exception as e:
            logger.error(f"Failed to get market state: {e}")
            return None

    async def get_recent_signals(self, hours: int = 24) -> list[SignalRecord]:
        """Get recent signals from database."""
        signals = []

        try:
            # Try to get from database directly
            if self.db_session:
                from src.database.models import Signal
                from sqlalchemy import desc

                cutoff = now_jst() - timedelta(hours=hours)
                db_signals = (
                    self.db_session.query(Signal)
                    .filter(Signal.timestamp >= cutoff)
                    .order_by(desc(Signal.timestamp))
                    .limit(100)
                    .all()
                )

                for s in db_signals:
                    signals.append(SignalRecord(
                        id=s.id,
                        timestamp=s.timestamp,
                        direction=s.direction,
                        confidence=s.confidence,
                        price=s.price,
                        features=json.loads(s.features) if s.features else {},
                        executed=s.executed,
                        outcome=s.outcome if hasattr(s, 'outcome') else None,
                        actual_move=s.actual_move if hasattr(s, 'actual_move') else None,
                    ))

        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")

        return signals

    async def get_recent_trades(self, hours: int = 24) -> list[TradeRecord]:
        """Get recent trades."""
        trades = []

        try:
            session = await self._get_http_session()
            async with session.get(f"{self.api_base_url}/api/trades?limit=100") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    cutoff = now_jst() - timedelta(hours=hours)

                    for t in data.get("trades", []):
                        entry_time = datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else now_jst()
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=JST)

                        if entry_time >= cutoff:
                            trades.append(TradeRecord(
                                id=t["id"],
                                symbol=t["symbol"],
                                side=t["side"],
                                entry_price=t["entry_price"],
                                exit_price=t.get("exit_price"),
                                size=t["size"],
                                pnl=t.get("pnl"),
                                entry_time=entry_time,
                                exit_time=datetime.fromisoformat(t["exit_time"]) if t.get("exit_time") else None,
                                status=t["status"],
                            ))

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")

        return trades

    async def get_open_positions(self) -> list[TradeRecord]:
        """Get current open positions."""
        positions = []

        try:
            session = await self._get_http_session()
            async with session.get(f"{self.api_base_url}/api/positions") as resp:
                if resp.status == 200:
                    data = await resp.json()

                    for p in data.get("positions", []):
                        positions.append(TradeRecord(
                            id=p["id"],
                            symbol=p["symbol"],
                            side=p["side"],
                            entry_price=p["entry_price"],
                            exit_price=None,
                            size=p["size"],
                            pnl=None,
                            entry_time=datetime.fromisoformat(p["entry_time"]) if p.get("entry_time") else now_jst(),
                            exit_time=None,
                            status="open",
                            stop_loss=p.get("stop_loss"),
                        ))

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")

        return positions

    async def get_system_health(self) -> SystemHealth | None:
        """Get system health status."""
        try:
            session = await self._get_http_session()
            async with session.get(f"{self.api_base_url}/api/health") as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Get emergency stop status
                    async with session.get(f"{self.api_base_url}/api/emergency") as em_resp:
                        em_data = await em_resp.json() if em_resp.status == 200 else {}

                    return SystemHealth(
                        status=data.get("status", "unknown"),
                        engine_running=data.get("checks", {}).get("engine", {}).get("status") == "ok",
                        database_ok=data.get("checks", {}).get("database", {}).get("status") == "ok",
                        gmo_api_ok=data.get("checks", {}).get("gmo_api", {}).get("status") == "ok",
                        memory_percent=data.get("system", {}).get("memory_percent", 0),
                        cpu_percent=data.get("system", {}).get("cpu_percent", 0),
                        uptime_hours=data.get("uptime_seconds", 0) / 3600,
                        emergency_stop_active=em_data.get("is_active", False),
                        long_stopped=em_data.get("long_stopped", False),
                        short_stopped=em_data.get("short_stopped", False),
                    )

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return None

    async def get_performance_metrics(self) -> PerformanceMetrics | None:
        """Get performance metrics."""
        try:
            session = await self._get_http_session()

            # Get PnL summary
            async with session.get(f"{self.api_base_url}/api/pnl") as resp:
                if resp.status != 200:
                    return None
                pnl_data = await resp.json()

            # Get status for additional metrics
            async with session.get(f"{self.api_base_url}/api/status") as resp:
                status_data = await resp.json() if resp.status == 200 else {}

            return PerformanceMetrics(
                daily_pnl=pnl_data.get("daily", 0),
                weekly_pnl=pnl_data.get("weekly", 0),
                monthly_pnl=pnl_data.get("monthly", 0),
                total_pnl=pnl_data.get("total", 0),
                win_rate=pnl_data.get("win_rate", 0),
                profit_factor=pnl_data.get("profit_factor", 0),
                trades_count=pnl_data.get("trades_count", 0),
                drawdown=0.0,  # TODO: Get from status
                consecutive_losses=0,  # TODO: Get from status
                consecutive_wins=0,  # TODO: Get from status
                long_win_rate=0.0,  # TODO: Get from detailed stats
                short_win_rate=0.0,  # TODO: Get from detailed stats
                capital=status_data.get("capital", 0),
            )

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return None

    async def get_price_at_time(
        self,
        symbol: str,
        target_time: datetime,
    ) -> float | None:
        """
        Get price at a specific time (for signal verification).

        Args:
            symbol: Trading symbol
            target_time: Target timestamp

        Returns:
            Price at that time or None if not available
        """
        # TODO: Implement historical price lookup
        # This would query OHLCV data from database or GMO API
        return None

    async def calculate_price_move(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float | None:
        """
        Calculate price movement between two times.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Percentage price change or None if not available
        """
        start_price = await self.get_price_at_time(symbol, start_time)
        end_price = await self.get_price_at_time(symbol, end_time)

        if start_price and end_price and start_price > 0:
            return (end_price - start_price) / start_price

        return None
