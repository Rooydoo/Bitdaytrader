"""Prometheus metrics collector for the trading system."""

import time
from functools import wraps
from typing import Any, Callable

import psutil
from loguru import logger

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        multiprocess,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")


# Default buckets for latency histograms (in seconds)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
PNL_BUCKETS = (-100000, -50000, -10000, -5000, -1000, 0, 1000, 5000, 10000, 50000, 100000)


class MetricsCollector:
    """Collects and exposes Prometheus metrics for the trading system."""

    def __init__(self, registry: Any = None) -> None:
        """
        Initialize the metrics collector.

        Args:
            registry: Optional custom Prometheus registry
        """
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            return

        self._enabled = True
        self._registry = registry

        # ========================================
        # Trading Metrics
        # ========================================

        # Trade counts
        self.trades_total = Counter(
            "trading_trades_total",
            "Total number of trades executed",
            ["symbol", "side", "status"],
            registry=registry,
        )

        # Current positions
        self.open_positions = Gauge(
            "trading_open_positions",
            "Number of currently open positions",
            ["symbol", "side"],
            registry=registry,
        )

        # PnL metrics
        self.daily_pnl = Gauge(
            "trading_daily_pnl_jpy",
            "Daily profit/loss in JPY",
            registry=registry,
        )

        self.weekly_pnl = Gauge(
            "trading_weekly_pnl_jpy",
            "Weekly profit/loss in JPY",
            registry=registry,
        )

        self.monthly_pnl = Gauge(
            "trading_monthly_pnl_jpy",
            "Monthly profit/loss in JPY",
            registry=registry,
        )

        self.total_pnl = Gauge(
            "trading_total_pnl_jpy",
            "Total profit/loss in JPY",
            registry=registry,
        )

        self.trade_pnl = Histogram(
            "trading_trade_pnl_jpy",
            "Distribution of individual trade PnL",
            ["symbol", "side"],
            buckets=PNL_BUCKETS,
            registry=registry,
        )

        # Win rate
        self.win_rate = Gauge(
            "trading_win_rate",
            "Current win rate (0.0-1.0)",
            registry=registry,
        )

        self.profit_factor = Gauge(
            "trading_profit_factor",
            "Profit factor (gross profit / gross loss)",
            registry=registry,
        )

        # Capital and drawdown
        self.capital = Gauge(
            "trading_capital_jpy",
            "Current trading capital in JPY",
            registry=registry,
        )

        self.drawdown = Gauge(
            "trading_drawdown_percent",
            "Current drawdown from peak (%)",
            registry=registry,
        )

        self.consecutive_losses = Gauge(
            "trading_consecutive_losses",
            "Current consecutive losing trades",
            registry=registry,
        )

        # ========================================
        # Model Metrics
        # ========================================

        self.prediction_confidence = Histogram(
            "model_prediction_confidence",
            "Distribution of model prediction confidence",
            ["symbol", "direction"],
            buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=registry,
        )

        self.predictions_total = Counter(
            "model_predictions_total",
            "Total number of predictions made",
            ["symbol", "direction", "outcome"],
            registry=registry,
        )

        # ========================================
        # API Metrics
        # ========================================

        self.api_requests_total = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
            registry=registry,
        )

        self.api_request_latency = Histogram(
            "api_request_latency_seconds",
            "API request latency in seconds",
            ["method", "endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )

        self.gmo_api_requests_total = Counter(
            "gmo_api_requests_total",
            "Total requests to GMO Coin API",
            ["endpoint", "status"],
            registry=registry,
        )

        self.gmo_api_latency = Histogram(
            "gmo_api_latency_seconds",
            "GMO Coin API request latency",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )

        self.gmo_api_retries = Counter(
            "gmo_api_retries_total",
            "Total GMO API retries",
            ["endpoint", "reason"],
            registry=registry,
        )

        # ========================================
        # WebSocket Metrics
        # ========================================

        self.websocket_connections = Gauge(
            "websocket_connections_active",
            "Number of active WebSocket connections",
            registry=registry,
        )

        self.websocket_messages_sent = Counter(
            "websocket_messages_sent_total",
            "Total WebSocket messages sent",
            ["message_type"],
            registry=registry,
        )

        # ========================================
        # System Metrics
        # ========================================

        self.system_memory_percent = Gauge(
            "system_memory_percent",
            "System memory usage percentage",
            registry=registry,
        )

        self.system_cpu_percent = Gauge(
            "system_cpu_percent",
            "System CPU usage percentage",
            registry=registry,
        )

        self.process_memory_bytes = Gauge(
            "process_memory_bytes",
            "Process memory usage in bytes",
            registry=registry,
        )

        self.uptime_seconds = Gauge(
            "uptime_seconds",
            "Application uptime in seconds",
            registry=registry,
        )

        # ========================================
        # Backup Metrics
        # ========================================

        self.backup_last_success_timestamp = Gauge(
            "backup_last_success_timestamp",
            "Timestamp of last successful backup",
            registry=registry,
        )

        self.backup_size_bytes = Gauge(
            "backup_size_bytes",
            "Size of last backup in bytes",
            registry=registry,
        )

        self.backups_total = Counter(
            "backups_total",
            "Total number of backups created",
            ["status"],
            registry=registry,
        )

        # ========================================
        # Emergency Stop Metrics
        # ========================================

        self.emergency_stop_active = Gauge(
            "emergency_stop_active",
            "Whether emergency stop is active (1=active, 0=inactive)",
            registry=registry,
        )

        self.direction_stopped = Gauge(
            "direction_stopped",
            "Whether a specific direction is stopped",
            ["direction"],
            registry=registry,
        )

        # ========================================
        # Info Metric
        # ========================================

        self.trading_info = Info(
            "trading_bot",
            "Trading bot information",
            registry=registry,
        )

        # Start time for uptime calculation
        self._start_time = time.time()

        logger.info("Prometheus metrics collector initialized")

    @property
    def enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self._enabled

    def record_trade(
        self,
        symbol: str,
        side: str,
        status: str,
        pnl: float | None = None,
    ) -> None:
        """
        Record a trade execution.

        Args:
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            status: Trade status (opened/closed/cancelled)
            pnl: Profit/loss if closed
        """
        if not self._enabled:
            return

        self.trades_total.labels(symbol=symbol, side=side, status=status).inc()

        if status == "closed" and pnl is not None:
            direction = "long" if side == "BUY" else "short"
            self.trade_pnl.labels(symbol=symbol, side=direction).observe(pnl)

    def record_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        outcome: str = "pending",
    ) -> None:
        """
        Record a model prediction.

        Args:
            symbol: Trading symbol
            direction: Predicted direction (long/short)
            confidence: Prediction confidence (0.0-1.0)
            outcome: Prediction outcome (pending/correct/incorrect)
        """
        if not self._enabled:
            return

        self.prediction_confidence.labels(symbol=symbol, direction=direction).observe(confidence)
        self.predictions_total.labels(symbol=symbol, direction=direction, outcome=outcome).inc()

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        latency: float,
    ) -> None:
        """
        Record an API request.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            status_code: Response status code
            latency: Request latency in seconds
        """
        if not self._enabled:
            return

        self.api_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
        self.api_request_latency.labels(method=method, endpoint=endpoint).observe(latency)

    def record_gmo_request(
        self,
        endpoint: str,
        status: str,
        latency: float,
        retries: int = 0,
        retry_reason: str | None = None,
    ) -> None:
        """
        Record a GMO Coin API request.

        Args:
            endpoint: GMO API endpoint
            status: Request status (success/error)
            latency: Request latency in seconds
            retries: Number of retries
            retry_reason: Reason for retries
        """
        if not self._enabled:
            return

        self.gmo_api_requests_total.labels(endpoint=endpoint, status=status).inc()
        self.gmo_api_latency.labels(endpoint=endpoint).observe(latency)

        if retries > 0 and retry_reason:
            self.gmo_api_retries.labels(endpoint=endpoint, reason=retry_reason).inc(retries)

    def record_backup(self, success: bool, size_bytes: int = 0) -> None:
        """
        Record a backup operation.

        Args:
            success: Whether backup was successful
            size_bytes: Backup size in bytes
        """
        if not self._enabled:
            return

        status = "success" if success else "failure"
        self.backups_total.labels(status=status).inc()

        if success:
            self.backup_last_success_timestamp.set(time.time())
            self.backup_size_bytes.set(size_bytes)

    def update_trading_stats(
        self,
        capital: float,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        total_pnl: float,
        win_rate: float,
        profit_factor: float,
        drawdown: float = 0.0,
        consecutive_losses: int = 0,
    ) -> None:
        """
        Update trading statistics gauges.

        Args:
            capital: Current capital
            daily_pnl: Daily PnL
            weekly_pnl: Weekly PnL
            monthly_pnl: Monthly PnL
            total_pnl: Total PnL
            win_rate: Win rate (0.0-1.0)
            profit_factor: Profit factor
            drawdown: Current drawdown percentage
            consecutive_losses: Current consecutive losses
        """
        if not self._enabled:
            return

        self.capital.set(capital)
        self.daily_pnl.set(daily_pnl)
        self.weekly_pnl.set(weekly_pnl)
        self.monthly_pnl.set(monthly_pnl)
        self.total_pnl.set(total_pnl)
        self.win_rate.set(win_rate)
        self.profit_factor.set(profit_factor)
        self.drawdown.set(drawdown)
        self.consecutive_losses.set(consecutive_losses)

    def update_positions(self, positions: list[dict]) -> None:
        """
        Update open positions gauges.

        Args:
            positions: List of position dicts with symbol and side
        """
        if not self._enabled:
            return

        # Reset all position gauges to 0
        # This is a simplified approach - in production you'd want to track more granularly

        # Count positions by symbol and side
        counts: dict[tuple[str, str], int] = {}
        for pos in positions:
            key = (pos["symbol"], pos["side"])
            counts[key] = counts.get(key, 0) + 1

        for (symbol, side), count in counts.items():
            direction = "long" if side == "BUY" else "short"
            self.open_positions.labels(symbol=symbol, side=direction).set(count)

    def update_emergency_stop(
        self,
        active: bool,
        long_stopped: bool = False,
        short_stopped: bool = False,
    ) -> None:
        """
        Update emergency stop metrics.

        Args:
            active: Whether global emergency stop is active
            long_stopped: Whether LONG is stopped
            short_stopped: Whether SHORT is stopped
        """
        if not self._enabled:
            return

        self.emergency_stop_active.set(1 if active else 0)
        self.direction_stopped.labels(direction="long").set(1 if long_stopped else 0)
        self.direction_stopped.labels(direction="short").set(1 if short_stopped else 0)

    def update_websocket_connections(self, count: int) -> None:
        """Update WebSocket connection count."""
        if not self._enabled:
            return
        self.websocket_connections.set(count)

    def record_websocket_message(self, message_type: str) -> None:
        """Record a WebSocket message sent."""
        if not self._enabled:
            return
        self.websocket_messages_sent.labels(message_type=message_type).inc()

    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        if not self._enabled:
            return

        try:
            memory = psutil.virtual_memory()
            self.system_memory_percent.set(memory.percent)

            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_cpu_percent.set(cpu_percent)

            process = psutil.Process()
            self.process_memory_bytes.set(process.memory_info().rss)

            uptime = time.time() - self._start_time
            self.uptime_seconds.set(uptime)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def set_info(self, mode: str, version: str = "1.0.0") -> None:
        """
        Set bot info metric.

        Args:
            mode: Trading mode (live/paper/backtest)
            version: Bot version
        """
        if not self._enabled:
            return

        self.trading_info.info({
            "mode": mode,
            "version": version,
        })

    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        if not self._enabled:
            return b"# Prometheus metrics disabled\n"

        self.update_system_metrics()
        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def setup_metrics(registry: Any = None) -> MetricsCollector:
    """
    Setup metrics collector with optional custom registry.

    Args:
        registry: Optional Prometheus registry

    Returns:
        Configured MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(registry=registry)
    return _metrics_collector


def track_latency(endpoint: str) -> Callable:
    """
    Decorator to track API endpoint latency.

    Args:
        endpoint: Endpoint name for labeling

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start
                metrics = get_metrics_collector()
                # Infer method from function name
                method = "GET" if func.__name__.startswith("get") else "POST"
                metrics.record_api_request(method, endpoint, 200, latency)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start
                metrics = get_metrics_collector()
                method = "GET" if func.__name__.startswith("get") else "POST"
                metrics.record_api_request(method, endpoint, 200, latency)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
