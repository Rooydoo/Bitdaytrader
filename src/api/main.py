"""FastAPI backend for web dashboard."""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config.settings import get_settings
from src.settings.runtime import get_runtime_settings


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._broadcast_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_status_update(self):
        """Broadcast current status to all clients."""
        if not self.active_connections:
            return

        try:
            status = await _get_full_status()
            await self.broadcast({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            logger.error(f"Error broadcasting status: {e}")

    def start_periodic_broadcast(self, interval: float = 5.0):
        """Start periodic status broadcasts."""
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(
                self._periodic_broadcast_loop(interval)
            )
            logger.info(f"Started periodic broadcast (interval: {interval}s)")

    async def _periodic_broadcast_loop(self, interval: float):
        """Internal loop for periodic broadcasts."""
        while True:
            try:
                await asyncio.sleep(interval)
                if self.active_connections:
                    await self.broadcast_status_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")


# Global WebSocket manager
ws_manager = ConnectionManager()

# Default API port
API_PORT = 8088

app = FastAPI(
    title="GMO Trading Bot API",
    description="API for monitoring and configuring the trading bot",
    version="1.0.0",
)


class EmergencyStopMode(str, Enum):
    """Emergency stop modes."""
    FULL_STOP = "full_stop"  # Stop everything, close all positions
    NO_NEW_POSITIONS = "no_new_positions"  # Keep existing, don't open new
    NONE = "none"  # Normal operation


class EmergencyStopReason(str, Enum):
    """Reasons for emergency stop."""
    MANUAL = "manual"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MARGIN_ALERT = "margin_alert"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"


# Emergency stop state
class EmergencyStopState:
    """Manages emergency stop state."""

    def __init__(self):
        self.mode: EmergencyStopMode = EmergencyStopMode.NONE
        self.reason: EmergencyStopReason | None = None
        self.triggered_at: datetime | None = None
        self.message: str = ""
        self.auto_resume_at: datetime | None = None
        # Direction-specific stops
        self.long_stopped: bool = False
        self.short_stopped: bool = False
        self.long_stop_reason: str = ""
        self.short_stop_reason: str = ""

    def activate(
        self,
        mode: EmergencyStopMode,
        reason: EmergencyStopReason,
        message: str = "",
        auto_resume_hours: int | None = None,
    ) -> None:
        """Activate emergency stop."""
        self.mode = mode
        self.reason = reason
        self.triggered_at = datetime.now()
        self.message = message

        if auto_resume_hours:
            self.auto_resume_at = datetime.now() + timedelta(hours=auto_resume_hours)

        # Full stop affects both directions
        if mode in [EmergencyStopMode.FULL_STOP, EmergencyStopMode.NO_NEW_POSITIONS]:
            self.long_stopped = True
            self.short_stopped = True

        logger.warning(
            f"EMERGENCY STOP ACTIVATED: mode={mode.value}, reason={reason.value}, "
            f"message={message}"
        )

    def deactivate(self) -> None:
        """Deactivate emergency stop."""
        logger.info(f"Emergency stop deactivated (was: {self.mode.value})")
        self.mode = EmergencyStopMode.NONE
        self.reason = None
        self.triggered_at = None
        self.message = ""
        self.auto_resume_at = None
        self.long_stopped = False
        self.short_stopped = False
        self.long_stop_reason = ""
        self.short_stop_reason = ""

    def stop_direction(self, direction: str, reason: str = "手動停止") -> None:
        """Stop a specific direction (LONG or SHORT)."""
        if direction.upper() == "LONG":
            self.long_stopped = True
            self.long_stop_reason = reason
            logger.warning(f"LONG trading stopped: {reason}")
        elif direction.upper() == "SHORT":
            self.short_stopped = True
            self.short_stop_reason = reason
            logger.warning(f"SHORT trading stopped: {reason}")

    def resume_direction(self, direction: str) -> None:
        """Resume a specific direction (LONG or SHORT)."""
        if direction.upper() == "LONG":
            self.long_stopped = False
            self.long_stop_reason = ""
            logger.info("LONG trading resumed")
        elif direction.upper() == "SHORT":
            self.short_stopped = False
            self.short_stop_reason = ""
            logger.info("SHORT trading resumed")

        # If both directions resumed and no global stop, clear mode
        if not self.long_stopped and not self.short_stopped and self.mode == EmergencyStopMode.NONE:
            pass  # Already clear

    def check_auto_resume(self) -> bool:
        """Check if auto-resume time has passed."""
        if self.auto_resume_at and datetime.now() >= self.auto_resume_at:
            self.deactivate()
            return True
        return False

    def is_active(self) -> bool:
        """Check if any emergency stop is active."""
        self.check_auto_resume()
        return self.mode != EmergencyStopMode.NONE or self.long_stopped or self.short_stopped

    def can_open_positions(self) -> bool:
        """Check if new positions can be opened (any direction)."""
        self.check_auto_resume()
        return self.mode == EmergencyStopMode.NONE and not self.long_stopped and not self.short_stopped

    def can_open_long(self) -> bool:
        """Check if LONG positions can be opened."""
        self.check_auto_resume()
        return self.mode == EmergencyStopMode.NONE and not self.long_stopped

    def can_open_short(self) -> bool:
        """Check if SHORT positions can be opened."""
        self.check_auto_resume()
        return self.mode == EmergencyStopMode.NONE and not self.short_stopped

    def should_close_all(self) -> bool:
        """Check if all positions should be closed."""
        return self.mode == EmergencyStopMode.FULL_STOP

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "reason": self.reason.value if self.reason else None,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "message": self.message,
            "auto_resume_at": self.auto_resume_at.isoformat() if self.auto_resume_at else None,
            "is_active": self.is_active(),
            "can_open_positions": self.can_open_positions(),
            "long_stopped": self.long_stopped,
            "short_stopped": self.short_stopped,
            "long_stop_reason": self.long_stop_reason,
            "short_stop_reason": self.short_stop_reason,
            "can_open_long": self.can_open_long(),
            "can_open_short": self.can_open_short(),
        }


# Global emergency stop state
_emergency_stop = EmergencyStopState()


def get_emergency_stop() -> EmergencyStopState:
    """Get global emergency stop state."""
    return _emergency_stop

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SettingUpdate(BaseModel):
    """Request model for updating a setting."""
    key: str
    value: str | float | int


class SettingsResponse(BaseModel):
    """Response model for settings."""
    settings: dict[str, Any]
    overrides: dict[str, Any]


class StatusResponse(BaseModel):
    """Response model for status."""
    mode: str
    is_running: bool
    capital: float
    daily_pnl: float
    daily_trades: int
    conservative_mode: bool
    emergency_stop: dict
    last_update: str


class EmergencyStopRequest(BaseModel):
    """Request model for emergency stop."""
    mode: str  # "full_stop" or "no_new_positions"
    reason: str = "manual"
    message: str = ""
    auto_resume_hours: int | None = None


class AllocationResponse(BaseModel):
    """Response model for allocation."""
    total_capital: float
    usable_capital: float
    utilization_rate: float
    long_ratio: float
    short_ratio: float
    symbols: dict[str, dict[str, float]]


class TradeRecord(BaseModel):
    """Model for a trade record."""
    id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None
    size: float
    pnl: float | None
    entry_time: str
    exit_time: str | None
    status: str


class PnLSummary(BaseModel):
    """Model for PnL summary."""
    daily: float
    weekly: float
    monthly: float
    total: float
    win_rate: float
    profit_factor: float
    trades_count: int


# Global reference to engine (set by main.py when starting)
_engine: Any = None


def set_engine(engine: Any) -> None:
    """Set the trading engine reference."""
    global _engine
    _engine = engine


def get_engine() -> Any:
    """Get the trading engine reference."""
    return _engine


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    uptime_seconds: float
    checks: dict[str, dict[str, Any]]
    system: dict[str, Any]


# Track API start time for uptime calculation
_api_start_time = time.time()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GMO Trading Bot API", "status": "running"}


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Checks:
    - API responsiveness
    - Database connectivity
    - GMO API connectivity
    - Engine status
    - System resources (memory, CPU, disk)
    """
    checks = {}
    overall_status = "healthy"

    # Check 1: Database connectivity
    try:
        from src.database.session import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        checks["database"] = {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}
        overall_status = "degraded"

    # Check 2: GMO API connectivity
    try:
        settings = get_settings()
        if settings.gmo_api_key and settings.gmo_api_secret:
            from src.api.gmo_client import GMOCoinClient
            client = GMOCoinClient(
                api_key=settings.gmo_api_key,
                api_secret=settings.gmo_api_secret,
            )
            # Just try to get ticker (public API, no auth needed)
            ticker = client.get_ticker("BTC_JPY")
            client.close()
            checks["gmo_api"] = {
                "status": "ok",
                "message": "GMO API reachable",
                "btc_price": ticker.last,
            }
        else:
            checks["gmo_api"] = {"status": "warning", "message": "API keys not configured"}
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        checks["gmo_api"] = {"status": "error", "message": str(e)}
        overall_status = "degraded"

    # Check 3: Trading engine status
    if _engine:
        try:
            capital = _engine._get_capital()
            risk_stats = _engine.risk_manager.get_daily_stats()
            checks["engine"] = {
                "status": "ok",
                "message": "Engine running",
                "capital": capital,
                "daily_trades": risk_stats["total"]["trades"],
                "daily_pnl": risk_stats["total"]["pnl"],
            }
        except Exception as e:
            checks["engine"] = {"status": "error", "message": str(e)}
            overall_status = "degraded"
    else:
        checks["engine"] = {"status": "warning", "message": "Engine not attached"}
        if overall_status == "healthy":
            overall_status = "degraded"

    # Check 4: Emergency stop status
    if _emergency_stop.is_active():
        checks["emergency_stop"] = {
            "status": "warning",
            "message": f"Emergency stop active: {_emergency_stop.mode.value}",
            "reason": _emergency_stop.reason.value if _emergency_stop.reason else None,
        }
        if overall_status == "healthy":
            overall_status = "degraded"
    else:
        direction_stops = []
        if _emergency_stop.long_stopped:
            direction_stops.append("LONG")
        if _emergency_stop.short_stopped:
            direction_stops.append("SHORT")
        if direction_stops:
            checks["emergency_stop"] = {
                "status": "warning",
                "message": f"Direction stops active: {', '.join(direction_stops)}",
            }
        else:
            checks["emergency_stop"] = {"status": "ok", "message": "Normal operation"}

    # System resources
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_percent = psutil.cpu_percent(interval=0.1)

    system_info = {
        "memory_percent": memory.percent,
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "cpu_percent": cpu_percent,
        "process_memory_mb": round(psutil.Process().memory_info().rss / (1024**2), 2),
    }

    # Check if system resources are critical
    if memory.percent > 90 or disk.percent > 95:
        overall_status = "unhealthy"
        checks["system_resources"] = {
            "status": "error",
            "message": f"Critical: Memory {memory.percent}%, Disk {disk.percent}%"
        }
    elif memory.percent > 80 or disk.percent > 85:
        if overall_status == "healthy":
            overall_status = "degraded"
        checks["system_resources"] = {
            "status": "warning",
            "message": f"Warning: Memory {memory.percent}%, Disk {disk.percent}%"
        }
    else:
        checks["system_resources"] = {
            "status": "ok",
            "message": f"Memory {memory.percent}%, Disk {disk.percent}%"
        }

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(time.time() - _api_start_time, 2),
        checks=checks,
        system=system_info,
    )


@app.get("/api/health/simple")
async def health_check_simple():
    """Simple health check for load balancers and monitoring tools."""
    try:
        # Quick check - just verify API is responsive
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Helper function for WebSocket broadcasts
async def _get_full_status() -> dict:
    """Get full status for WebSocket broadcast."""
    settings = get_settings()
    rs = get_runtime_settings()

    mode = rs.get("mode", settings.mode)

    # Default values
    status = {
        "mode": mode,
        "is_running": False,
        "capital": 0.0,
        "daily_pnl": 0.0,
        "daily_trades": 0,
        "conservative_mode": False,
        "emergency_stop": _emergency_stop.to_dict(),
        "positions": [],
        "drawdown": None,
    }

    if _engine:
        try:
            status["is_running"] = True
            status["capital"] = _engine._get_capital()
            stats = _engine.risk_manager.get_daily_stats()
            status["daily_pnl"] = stats["total"]["pnl"]
            status["daily_trades"] = stats["total"]["trades"]
            status["conservative_mode"] = _engine.risk_manager.is_conservative_mode
            status["drawdown"] = stats.get("drawdown")

            # Get long/short stats
            status["long_stats"] = stats.get("long", {})
            status["short_stats"] = stats.get("short", {})
        except Exception as e:
            logger.error(f"Error getting status: {e}")

    return status


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)

    # Start periodic broadcasts if not already running
    ws_manager.start_periodic_broadcast(interval=5.0)

    # Send initial status
    try:
        status = await _get_full_status()
        await ws_manager.send_personal_message({
            "type": "initial_status",
            "data": status,
            "timestamp": datetime.now().isoformat(),
        }, websocket)
    except Exception as e:
        logger.error(f"Error sending initial status: {e}")

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await ws_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    }, websocket)

                elif message.get("type") == "subscribe":
                    # Client can subscribe to specific events
                    await ws_manager.send_personal_message({
                        "type": "subscribed",
                        "channel": message.get("channel", "all"),
                        "timestamp": datetime.now().isoformat(),
                    }, websocket)

                elif message.get("type") == "request_status":
                    # Client requests immediate status update
                    status = await _get_full_status()
                    await ws_manager.send_personal_message({
                        "type": "status_update",
                        "data": status,
                        "timestamp": datetime.now().isoformat(),
                    }, websocket)

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Function to broadcast trade events
async def broadcast_trade_event(event_type: str, trade_data: dict):
    """Broadcast a trade event to all connected clients."""
    await ws_manager.broadcast({
        "type": event_type,
        "data": trade_data,
        "timestamp": datetime.now().isoformat(),
    })


# Function to broadcast alerts
async def broadcast_alert(alert_type: str, message: str, severity: str = "info"):
    """Broadcast an alert to all connected clients."""
    await ws_manager.broadcast({
        "type": "alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current bot status."""
    settings = get_settings()
    rs = get_runtime_settings()

    mode = rs.get("mode", settings.mode)

    # Default values if engine not connected
    capital = 0.0
    daily_pnl = 0.0
    daily_trades = 0
    conservative_mode = False
    is_running = False

    if _engine:
        try:
            is_running = True
            capital = _engine._get_capital()
            stats = _engine.risk_manager.get_daily_stats()
            daily_pnl = stats["total"]["pnl"]
            daily_trades = stats["total"]["trades"]
            conservative_mode = _engine.risk_manager.is_conservative_mode
        except Exception:
            pass

    return StatusResponse(
        mode=mode,
        is_running=is_running,
        capital=capital,
        daily_pnl=daily_pnl,
        daily_trades=daily_trades,
        conservative_mode=conservative_mode,
        emergency_stop=_emergency_stop.to_dict(),
        last_update=datetime.now().isoformat(),
    )


@app.get("/api/settings", response_model=SettingsResponse)
async def get_all_settings():
    """Get all current settings."""
    settings = get_settings()
    rs = get_runtime_settings()

    # Build settings dict
    current_settings = {
        "symbols_config": rs.get("symbols_config", settings.symbols_config),
        "total_capital_utilization": rs.get("total_capital_utilization", settings.total_capital_utilization),
        "long_allocation_ratio": rs.get("long_allocation_ratio", settings.long_allocation_ratio),
        "short_allocation_ratio": rs.get("short_allocation_ratio", settings.short_allocation_ratio),
        "long_risk_per_trade": rs.get("long_risk_per_trade", settings.long_risk_per_trade),
        "short_risk_per_trade": rs.get("short_risk_per_trade", settings.short_risk_per_trade),
        "long_confidence_threshold": rs.get("long_confidence_threshold", settings.long_confidence_threshold),
        "short_confidence_threshold": rs.get("short_confidence_threshold", settings.short_confidence_threshold),
        "long_max_daily_trades": rs.get("long_max_daily_trades", settings.long_max_daily_trades),
        "short_max_daily_trades": rs.get("short_max_daily_trades", settings.short_max_daily_trades),
        "daily_loss_limit": rs.get("daily_loss_limit", settings.daily_loss_limit),
        "max_daily_trades": rs.get("max_daily_trades", settings.max_daily_trades),
        "mode": rs.get("mode", settings.mode),
        "leverage": settings.leverage,
        "use_leverage": settings.use_leverage,
    }

    return SettingsResponse(
        settings=current_settings,
        overrides=rs.get_all_overrides(),
    )


@app.post("/api/settings")
async def update_setting(update: SettingUpdate):
    """Update a setting."""
    rs = get_runtime_settings()

    success, message = rs.set(update.key, update.value)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Apply to engine if running
    if _engine:
        try:
            _apply_setting_to_engine(update.key)
        except Exception as e:
            pass  # Log but don't fail

    return {"success": True, "message": message}


@app.delete("/api/settings/{key}")
async def reset_setting(key: str):
    """Reset a setting to default."""
    rs = get_runtime_settings()

    success, message = rs.delete(key)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"success": True, "message": message}


@app.get("/api/allocation", response_model=AllocationResponse)
async def get_allocation():
    """Get current capital allocation."""
    settings = get_settings()
    rs = get_runtime_settings()

    symbols_config = rs.get("symbols_config", settings.symbols_config)
    utilization = rs.get("total_capital_utilization", settings.total_capital_utilization)
    long_ratio = rs.get("long_allocation_ratio", settings.long_allocation_ratio)
    short_ratio = rs.get("short_allocation_ratio", settings.short_allocation_ratio)

    # Parse symbols
    allocations = {}
    for item in symbols_config.split(","):
        item = item.strip()
        if ":" in item:
            symbol, alloc = item.split(":")
            allocations[symbol.strip()] = float(alloc.strip())

    # Get capital
    capital = 0.0
    if _engine:
        try:
            capital = _engine._get_capital()
        except Exception:
            capital = 1000000.0  # Default for display
    else:
        capital = 1000000.0

    usable = capital * utilization

    # Build symbol allocation details
    symbols_detail = {}
    for symbol, pct in allocations.items():
        symbol_capital = usable * pct
        symbols_detail[symbol] = {
            "allocation_pct": pct,
            "total_allocated": symbol_capital,
            "long_allocated": symbol_capital * long_ratio,
            "short_allocated": symbol_capital * short_ratio,
        }

    return AllocationResponse(
        total_capital=capital,
        usable_capital=usable,
        utilization_rate=utilization,
        long_ratio=long_ratio,
        short_ratio=short_ratio,
        symbols=symbols_detail,
    )


@app.get("/api/trades")
async def get_trades(
    limit: int = 50,
    status: str | None = None,
    symbol: str | None = None,
):
    """Get trade history."""
    if not _engine:
        return {"trades": [], "total": 0}

    try:
        if status == "open":
            trades = _engine.trade_repo.get_open_trades()
        else:
            trades = _engine.trade_repo.get_recent_trades(limit=limit)

        # Filter by symbol if specified
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        trade_list = []
        for t in trades:
            trade_list.append({
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "status": "open" if t.exit_time is None else "closed",
            })

        return {"trades": trade_list, "total": len(trade_list)}

    except Exception as e:
        return {"trades": [], "total": 0, "error": str(e)}


@app.get("/api/pnl", response_model=PnLSummary)
async def get_pnl():
    """Get PnL summary."""
    if not _engine:
        return PnLSummary(
            daily=0.0,
            weekly=0.0,
            monthly=0.0,
            total=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            trades_count=0,
        )

    try:
        stats = _engine.risk_manager.get_daily_stats()

        # Get historical trades for weekly/monthly
        trades = _engine.trade_repo.get_recent_trades(limit=1000)

        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        weekly_pnl = 0.0
        monthly_pnl = 0.0
        total_pnl = 0.0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for t in trades:
            if t.pnl is None:
                continue

            total_pnl += t.pnl

            if t.pnl > 0:
                wins += 1
                gross_profit += t.pnl
            else:
                losses += 1
                gross_loss += abs(t.pnl)

            if t.exit_time:
                if t.exit_time >= week_ago:
                    weekly_pnl += t.pnl
                if t.exit_time >= month_ago:
                    monthly_pnl += t.pnl

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return PnLSummary(
            daily=stats["total"]["pnl"],
            weekly=weekly_pnl,
            monthly=monthly_pnl,
            total=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=total_trades,
        )

    except Exception as e:
        return PnLSummary(
            daily=0.0,
            weekly=0.0,
            monthly=0.0,
            total=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            trades_count=0,
        )


@app.get("/api/pnl/history")
async def get_pnl_history(days: int = 30):
    """Get daily PnL history for charts."""
    if not _engine:
        return {"history": []}

    try:
        trades = _engine.trade_repo.get_recent_trades(limit=1000)

        # Group by date
        daily_pnl: dict[str, float] = {}
        now = datetime.now()

        for i in range(days):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            daily_pnl[date] = 0.0

        for t in trades:
            if t.exit_time and t.pnl:
                date = t.exit_time.strftime("%Y-%m-%d")
                if date in daily_pnl:
                    daily_pnl[date] += t.pnl

        # Convert to list sorted by date
        history = [
            {"date": date, "pnl": pnl}
            for date, pnl in sorted(daily_pnl.items())
        ]

        return {"history": history}

    except Exception as e:
        return {"history": [], "error": str(e)}


@app.get("/api/positions")
async def get_positions():
    """Get current open positions."""
    if not _engine:
        return {"positions": []}

    try:
        trades = _engine.trade_repo.get_open_trades()

        positions = []
        for t in trades:
            positions.append({
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "size": t.size,
                "stop_loss": t.stop_loss,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            })

        return {"positions": positions}

    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.get("/api/model/status")
async def get_model_status():
    """Get model training status."""
    settings = get_settings()

    from pathlib import Path
    model_path = Path(settings.model_path)

    model_exists = model_path.exists()
    last_trained = None

    if model_exists:
        last_trained = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()

    # Check for walk-forward results
    wf_results_path = Path("data/walkforward_results.json")
    wf_status = None

    if wf_results_path.exists():
        import json
        try:
            with open(wf_results_path) as f:
                wf_data = json.load(f)
                wf_status = {
                    "last_run": wf_data.get("timestamp"),
                    "sharpe_ratio": wf_data.get("summary", {}).get("avg_sharpe_ratio"),
                    "win_rate": wf_data.get("summary", {}).get("avg_win_rate"),
                    "overfitting_detected": wf_data.get("summary", {}).get("train_test_gap", 0) > 0.10,
                }
        except Exception:
            pass

    return {
        "model_exists": model_exists,
        "model_path": str(model_path),
        "last_trained": last_trained,
        "walkforward_status": wf_status,
    }


# ============================================
# Emergency Stop Endpoints
# ============================================

@app.get("/api/emergency")
async def get_emergency_status():
    """Get current emergency stop status."""
    return _emergency_stop.to_dict()


@app.post("/api/emergency/stop")
async def activate_emergency_stop(request: EmergencyStopRequest):
    """
    Activate emergency stop.

    Modes:
    - full_stop: Stop all trading, close all positions immediately
    - no_new_positions: Keep existing positions, don't open new ones
    """
    try:
        mode = EmergencyStopMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Use 'full_stop' or 'no_new_positions'"
        )

    try:
        reason = EmergencyStopReason(request.reason)
    except ValueError:
        reason = EmergencyStopReason.MANUAL

    _emergency_stop.activate(
        mode=mode,
        reason=reason,
        message=request.message,
        auto_resume_hours=request.auto_resume_hours,
    )

    # If full stop and engine is running, close all positions
    if mode == EmergencyStopMode.FULL_STOP and _engine:
        try:
            await _close_all_positions()
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")

    # Send Telegram alert
    await _send_emergency_alert(
        f"🚨 緊急停止発動\n"
        f"モード: {mode.value}\n"
        f"理由: {reason.value}\n"
        f"メッセージ: {request.message or 'なし'}"
    )

    return {
        "success": True,
        "message": f"Emergency stop activated: {mode.value}",
        "status": _emergency_stop.to_dict(),
    }


@app.post("/api/emergency/resume")
async def deactivate_emergency_stop():
    """Deactivate emergency stop and resume normal operation."""
    if not _emergency_stop.is_active():
        return {
            "success": False,
            "message": "Emergency stop is not active",
        }

    _emergency_stop.deactivate()

    # Send Telegram alert
    await _send_emergency_alert("✅ 緊急停止解除 - 取引を再開します")

    return {
        "success": True,
        "message": "Emergency stop deactivated, trading resumed",
        "status": _emergency_stop.to_dict(),
    }


# Direction-specific stop/resume endpoints
@app.post("/api/emergency/stop/{direction}")
async def stop_direction(direction: str, reason: str = "手動停止"):
    """
    Stop a specific direction (LONG or SHORT).

    Args:
        direction: "long" or "short"
        reason: Reason for stopping
    """
    direction = direction.upper()
    if direction not in ["LONG", "SHORT"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid direction. Use 'long' or 'short'"
        )

    _emergency_stop.stop_direction(direction, reason)

    # Send Telegram alert
    await _send_emergency_alert(
        f"🛑 {direction}取引を停止\n理由: {reason}"
    )

    return {
        "success": True,
        "message": f"{direction} trading stopped",
        "status": _emergency_stop.to_dict(),
    }


@app.post("/api/emergency/resume/{direction}")
async def resume_direction(direction: str):
    """
    Resume a specific direction (LONG or SHORT).

    Args:
        direction: "long" or "short"
    """
    direction = direction.upper()
    if direction not in ["LONG", "SHORT"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid direction. Use 'long' or 'short'"
        )

    # Check if this direction is actually stopped
    if direction == "LONG" and not _emergency_stop.long_stopped:
        return {
            "success": False,
            "message": "LONG trading is not stopped",
        }
    if direction == "SHORT" and not _emergency_stop.short_stopped:
        return {
            "success": False,
            "message": "SHORT trading is not stopped",
        }

    _emergency_stop.resume_direction(direction)

    # Send Telegram alert
    await _send_emergency_alert(f"✅ {direction}取引を再開")

    return {
        "success": True,
        "message": f"{direction} trading resumed",
        "status": _emergency_stop.to_dict(),
    }


async def _close_all_positions():
    """Close all open positions (called during full stop)."""
    if not _engine:
        return

    try:
        open_trades = _engine.trade_repo.get_open_trades()
        for trade in open_trades:
            logger.info(f"Emergency closing position: {trade.symbol} {trade.side}")
            # This would need to be implemented in the engine
            # For now, we just log the intent
    except Exception as e:
        logger.error(f"Error closing positions: {e}")


async def _send_emergency_alert(message: str):
    """Send emergency alert via Telegram."""
    try:
        settings = get_settings()
        if settings.telegram_bot_token and settings.telegram_chat_id:
            import aiohttp
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    "chat_id": settings.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                })
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def check_auto_emergency_stop(daily_pnl: float, capital: float, margin_ratio: float | None = None) -> bool:
    """
    Check if auto emergency stop should be triggered.

    Called by the trading engine periodically.

    Args:
        daily_pnl: Today's PnL
        capital: Current capital
        margin_ratio: Current margin ratio (if using leverage)

    Returns:
        True if emergency stop was triggered
    """
    settings = get_settings()

    # Check daily loss limit
    daily_loss_pct = abs(daily_pnl) / capital if capital > 0 and daily_pnl < 0 else 0
    if daily_loss_pct >= settings.daily_loss_limit:
        _emergency_stop.activate(
            mode=EmergencyStopMode.NO_NEW_POSITIONS,
            reason=EmergencyStopReason.DAILY_LOSS_LIMIT,
            message=f"日次損失上限到達: {daily_loss_pct:.1%} >= {settings.daily_loss_limit:.1%}",
            auto_resume_hours=24,  # Auto resume next day
        )
        return True

    # Check margin ratio (if using leverage)
    if margin_ratio is not None and margin_ratio <= settings.margin_call_threshold:
        _emergency_stop.activate(
            mode=EmergencyStopMode.NO_NEW_POSITIONS,
            reason=EmergencyStopReason.MARGIN_ALERT,
            message=f"証拠金維持率低下: {margin_ratio:.0%} <= {settings.margin_call_threshold:.0%}",
        )
        return True

    return False


def _apply_setting_to_engine(key: str) -> None:
    """Apply a setting change to the running engine."""
    if not _engine:
        return

    settings = get_settings()
    rs = get_runtime_settings()

    if key in ["symbols_config", "total_capital_utilization", "long_allocation_ratio", "short_allocation_ratio"]:
        # Get effective values
        symbols_str = rs.get("symbols_config", settings.symbols_config)
        allocations = {}
        for item in symbols_str.split(","):
            item = item.strip()
            if ":" in item:
                sym, alloc = item.split(":")
                allocations[sym.strip()] = float(alloc.strip())

        _engine.risk_manager.configure_allocation(
            symbol_allocations=allocations,
            total_capital_utilization=rs.get("total_capital_utilization", settings.total_capital_utilization),
            long_allocation_ratio=rs.get("long_allocation_ratio", settings.long_allocation_ratio),
            short_allocation_ratio=rs.get("short_allocation_ratio", settings.short_allocation_ratio),
        )


# ============================================
# Backup Endpoints
# ============================================

# Global backup manager (lazy initialized)
_backup_manager: Any = None


def get_backup_manager():
    """Get or create backup manager instance."""
    global _backup_manager
    if _backup_manager is None:
        from src.backup.manager import BackupManager, BackupConfig
        _backup_manager = BackupManager(BackupConfig())
    return _backup_manager


@app.get("/api/backup/list")
async def list_backups():
    """List all available backups."""
    try:
        manager = get_backup_manager()
        backups = manager.list_backups()
        stats = manager.get_backup_stats()
        return {
            "backups": backups,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backup/create")
async def create_backup(name_suffix: str = ""):
    """Create a new backup."""
    try:
        manager = get_backup_manager()
        # Run backup in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, manager.create_backup, name_suffix or "manual"
        )

        if result.success:
            # Send notification
            await _send_emergency_alert(
                f"Backup completed: {result.backup_path.name if result.backup_path else 'unknown'}\n"
                f"Files: {len(result.files_backed_up)}, "
                f"Size: {result.size_bytes / 1024 / 1024:.2f} MB"
            )
            return {
                "success": True,
                "backup_name": result.backup_path.name if result.backup_path else None,
                "files_backed_up": len(result.files_backed_up),
                "size_bytes": result.size_bytes,
                "duration_seconds": result.duration_seconds,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Backup failed: {result.error_message}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backup/restore/{backup_name}")
async def restore_backup(backup_name: str, confirm: bool = False):
    """
    Restore a backup.

    WARNING: This will overwrite existing data!
    Set confirm=true to actually perform the restore.
    """
    if not confirm:
        return {
            "success": False,
            "message": "Please set confirm=true to perform restore. WARNING: This will overwrite existing data!",
        }

    try:
        manager = get_backup_manager()
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, manager.restore_backup, backup_name
        )

        if success:
            await _send_emergency_alert(f"Backup restored: {backup_name}")
            return {
                "success": True,
                "message": f"Backup {backup_name} restored successfully",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restore backup: {backup_name}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/backup/{backup_name}")
async def delete_backup(backup_name: str):
    """Delete a specific backup."""
    try:
        manager = get_backup_manager()
        backup_path = manager.backup_dir / backup_name

        if not backup_path.exists():
            # Try with .tar.gz
            backup_path = manager.backup_dir / f"{backup_name}.tar.gz"
            if not backup_path.exists():
                raise HTTPException(status_code=404, detail="Backup not found")

        import shutil
        if backup_path.is_file():
            backup_path.unlink()
        else:
            shutil.rmtree(backup_path)

        logger.info(f"Deleted backup: {backup_name}")
        return {"success": True, "message": f"Backup {backup_name} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup/stats")
async def get_backup_stats():
    """Get backup statistics."""
    try:
        manager = get_backup_manager()
        return manager.get_backup_stats()
    except Exception as e:
        logger.error(f"Failed to get backup stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8088
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
