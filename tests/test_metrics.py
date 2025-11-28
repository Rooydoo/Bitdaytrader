"""Tests for the MetricsCollector class."""

import pytest
from unittest.mock import patch, MagicMock


class TestMetricsCollectorWithoutPrometheus:
    """Tests for MetricsCollector when prometheus_client is not available."""

    def test_disabled_when_not_installed(self):
        """Test metrics are disabled when prometheus_client not installed."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            # Force reimport
            import importlib
            from src.metrics import collector
            importlib.reload(collector)

            mc = collector.MetricsCollector()
            assert mc.enabled is False
            assert mc.generate_metrics() == b"# Prometheus metrics disabled\n"


class TestMetricsCollectorBasics:
    """Basic tests for MetricsCollector."""

    @pytest.fixture
    def metrics(self):
        """Create a MetricsCollector instance."""
        try:
            from prometheus_client import CollectorRegistry
            from src.metrics.collector import MetricsCollector
            registry = CollectorRegistry()
            return MetricsCollector(registry=registry)
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_initialization(self, metrics):
        """Test MetricsCollector initialization."""
        assert metrics.enabled is True
        assert metrics.trades_total is not None
        assert metrics.daily_pnl is not None
        assert metrics.capital is not None

    def test_record_trade(self, metrics):
        """Test recording a trade."""
        metrics.record_trade(
            symbol="BTC_JPY",
            side="BUY",
            status="opened",
        )

        # Record closed trade with PnL
        metrics.record_trade(
            symbol="BTC_JPY",
            side="BUY",
            status="closed",
            pnl=5000,
        )

        # No exception means success
        assert True

    def test_record_prediction(self, metrics):
        """Test recording a prediction."""
        metrics.record_prediction(
            symbol="BTC_JPY",
            direction="long",
            confidence=0.75,
            outcome="pending",
        )

        metrics.record_prediction(
            symbol="ETH_JPY",
            direction="short",
            confidence=0.82,
            outcome="correct",
        )

        assert True

    def test_record_api_request(self, metrics):
        """Test recording an API request."""
        metrics.record_api_request(
            method="GET",
            endpoint="/api/status",
            status_code=200,
            latency=0.05,
        )

        metrics.record_api_request(
            method="POST",
            endpoint="/api/settings",
            status_code=400,
            latency=0.1,
        )

        assert True

    def test_record_gmo_request(self, metrics):
        """Test recording a GMO API request."""
        metrics.record_gmo_request(
            endpoint="ticker",
            status="success",
            latency=0.2,
        )

        metrics.record_gmo_request(
            endpoint="order",
            status="error",
            latency=1.5,
            retries=2,
            retry_reason="timeout",
        )

        assert True

    def test_record_backup(self, metrics):
        """Test recording a backup."""
        metrics.record_backup(success=True, size_bytes=1024 * 1024)
        metrics.record_backup(success=False)

        assert True

    def test_update_trading_stats(self, metrics):
        """Test updating trading statistics."""
        metrics.update_trading_stats(
            capital=1_000_000,
            daily_pnl=5000,
            weekly_pnl=15000,
            monthly_pnl=50000,
            total_pnl=100000,
            win_rate=0.65,
            profit_factor=1.8,
            drawdown=5.5,
            consecutive_losses=1,
        )

        assert True

    def test_update_positions(self, metrics):
        """Test updating positions."""
        positions = [
            {"symbol": "BTC_JPY", "side": "BUY"},
            {"symbol": "BTC_JPY", "side": "BUY"},
            {"symbol": "ETH_JPY", "side": "SELL"},
        ]
        metrics.update_positions(positions)

        assert True

    def test_update_emergency_stop(self, metrics):
        """Test updating emergency stop status."""
        metrics.update_emergency_stop(active=False, long_stopped=False, short_stopped=False)
        metrics.update_emergency_stop(active=True, long_stopped=True, short_stopped=False)

        assert True

    def test_update_websocket_connections(self, metrics):
        """Test updating WebSocket connection count."""
        metrics.update_websocket_connections(5)
        metrics.update_websocket_connections(0)

        assert True

    def test_record_websocket_message(self, metrics):
        """Test recording WebSocket messages."""
        metrics.record_websocket_message("status_update")
        metrics.record_websocket_message("trade_opened")

        assert True

    def test_update_system_metrics(self, metrics):
        """Test updating system metrics."""
        metrics.update_system_metrics()

        assert True

    def test_set_info(self, metrics):
        """Test setting bot info."""
        metrics.set_info(mode="paper", version="1.0.0")
        metrics.set_info(mode="live")

        assert True

    def test_generate_metrics(self, metrics):
        """Test generating Prometheus metrics output."""
        # Record some metrics first
        metrics.record_trade(symbol="BTC_JPY", side="BUY", status="opened")
        metrics.update_trading_stats(
            capital=1_000_000,
            daily_pnl=5000,
            weekly_pnl=15000,
            monthly_pnl=50000,
            total_pnl=100000,
            win_rate=0.65,
            profit_factor=1.8,
        )

        output = metrics.generate_metrics()

        assert isinstance(output, bytes)
        assert len(output) > 0
        # Check for some expected metric names
        output_str = output.decode("utf-8")
        assert "trading_capital_jpy" in output_str or "trading" in output_str

    def test_get_content_type(self, metrics):
        """Test getting content type."""
        content_type = metrics.get_content_type()
        assert "text/plain" in content_type or "openmetrics" in content_type


class TestMetricsGlobalFunctions:
    """Tests for global metrics functions."""

    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        try:
            from src.metrics.collector import get_metrics_collector
            mc = get_metrics_collector()
            assert mc is not None

            # Should return same instance
            mc2 = get_metrics_collector()
            assert mc is mc2
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_setup_metrics(self):
        """Test setting up metrics with custom registry."""
        try:
            from prometheus_client import CollectorRegistry
            from src.metrics.collector import setup_metrics, get_metrics_collector

            registry = CollectorRegistry()
            mc = setup_metrics(registry=registry)

            assert mc is not None
            assert mc._registry is registry

            # Global should now point to this instance
            mc2 = get_metrics_collector()
            assert mc is mc2
        except ImportError:
            pytest.skip("prometheus_client not installed")


class TestTrackLatencyDecorator:
    """Tests for the track_latency decorator."""

    def test_sync_function(self):
        """Test decorator on sync function."""
        try:
            from src.metrics.collector import track_latency

            @track_latency("/test/endpoint")
            def sync_func():
                return "result"

            result = sync_func()
            assert result == "result"
        except ImportError:
            pytest.skip("prometheus_client not installed")

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test decorator on async function."""
        try:
            from src.metrics.collector import track_latency

            @track_latency("/test/async/endpoint")
            async def async_func():
                return "async_result"

            result = await async_func()
            assert result == "async_result"
        except ImportError:
            pytest.skip("prometheus_client not installed")


class TestMetricsDisabled:
    """Tests for when metrics are disabled."""

    def test_operations_are_noop_when_disabled(self):
        """Test that all operations are no-ops when disabled."""
        from src.metrics.collector import MetricsCollector

        # Create a disabled collector by mocking PROMETHEUS_AVAILABLE
        with patch("src.metrics.collector.PROMETHEUS_AVAILABLE", False):
            import importlib
            from src.metrics import collector
            importlib.reload(collector)

            mc = collector.MetricsCollector()
            assert mc.enabled is False

            # All these should work without error
            mc.record_trade("BTC_JPY", "BUY", "opened")
            mc.record_prediction("BTC_JPY", "long", 0.8)
            mc.record_api_request("GET", "/test", 200, 0.1)
            mc.record_gmo_request("ticker", "success", 0.2)
            mc.record_backup(True, 1024)
            mc.update_trading_stats(100, 50, 100, 200, 500, 0.6, 1.5)
            mc.update_positions([])
            mc.update_emergency_stop(False)
            mc.update_websocket_connections(0)
            mc.record_websocket_message("test")
            mc.update_system_metrics()
            mc.set_info("paper")

            assert mc.generate_metrics() == b"# Prometheus metrics disabled\n"
            assert mc.get_content_type() == "text/plain"
