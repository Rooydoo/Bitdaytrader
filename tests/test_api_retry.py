"""Tests for the API retry mechanism."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from src.api.gmo_client import (
    GMOCoinClient,
    RetryConfig,
    APIError,
    is_retryable_error,
    RETRYABLE_STATUS_CODES,
    RETRYABLE_GMO_ERROR_CODES,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 60.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_delay_calculation_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=30.0,
            jitter=False,
        )

        # Attempt 0: 1.0 * 2^0 = 1.0
        assert config.get_delay(0) == 1.0

        # Attempt 1: 1.0 * 2^1 = 2.0
        assert config.get_delay(1) == 2.0

        # Attempt 2: 1.0 * 2^2 = 4.0
        assert config.get_delay(2) == 4.0

        # Attempt 3: 1.0 * 2^3 = 8.0
        assert config.get_delay(3) == 8.0

    def test_delay_respects_max_delay(self):
        """Test delay respects maximum."""
        config = RetryConfig(
            base_delay=10.0,
            exponential_base=2.0,
            max_delay=20.0,
            jitter=False,
        )

        # Attempt 5 would be 10 * 2^5 = 320, but capped at 20
        assert config.get_delay(5) == 20.0

    def test_delay_with_jitter(self):
        """Test delay with jitter stays within bounds."""
        config = RetryConfig(
            base_delay=10.0,
            jitter=True,
        )

        for attempt in range(5):
            delay = config.get_delay(attempt)
            base = min(10.0 * (2.0 ** attempt), 30.0)
            jitter_range = base * 0.25

            # Should be within ±25% of base
            assert delay >= base - jitter_range - 0.01
            assert delay <= base + jitter_range + 0.01


class TestAPIError:
    """Tests for APIError exception."""

    def test_basic_error(self):
        """Test basic APIError creation."""
        error = APIError("Test error")

        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.gmo_error_code is None
        assert error.is_retryable is False

    def test_error_with_status_code(self):
        """Test APIError with HTTP status code."""
        error = APIError(
            "HTTP Error",
            status_code=500,
            is_retryable=True,
        )

        assert error.status_code == 500
        assert error.is_retryable is True

    def test_error_with_gmo_code(self):
        """Test APIError with GMO error code."""
        error = APIError(
            "GMO Error",
            gmo_error_code=1,
            is_retryable=True,
        )

        assert error.gmo_error_code == 1
        assert error.is_retryable is True


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_retryable_api_error(self):
        """Test retryable APIError is identified."""
        error = APIError("Test", is_retryable=True)
        assert is_retryable_error(error) is True

    def test_non_retryable_api_error(self):
        """Test non-retryable APIError is identified."""
        error = APIError("Test", is_retryable=False)
        assert is_retryable_error(error) is False

    def test_timeout_is_retryable(self):
        """Test timeout errors are retryable."""
        error = httpx.TimeoutException("Timeout")
        assert is_retryable_error(error) is True

    def test_connect_error_is_retryable(self):
        """Test connection errors are retryable."""
        error = httpx.ConnectError("Connection failed")
        assert is_retryable_error(error) is True

    def test_http_500_is_retryable(self):
        """Test HTTP 500 is retryable."""
        response = Mock()
        response.status_code = 500
        error = httpx.HTTPStatusError("", request=Mock(), response=response)
        assert is_retryable_error(error) is True

    def test_http_400_is_not_retryable(self):
        """Test HTTP 400 is not retryable."""
        response = Mock()
        response.status_code = 400
        error = httpx.HTTPStatusError("", request=Mock(), response=response)
        assert is_retryable_error(error) is False

    def test_unknown_error_is_not_retryable(self):
        """Test unknown errors are not retryable."""
        error = ValueError("Unknown error")
        assert is_retryable_error(error) is False

    @pytest.mark.parametrize("status_code", RETRYABLE_STATUS_CODES)
    def test_all_retryable_status_codes(self, status_code):
        """Test all retryable status codes."""
        response = Mock()
        response.status_code = status_code
        error = httpx.HTTPStatusError("", request=Mock(), response=response)
        assert is_retryable_error(error) is True


class TestGMOCoinClientRetry:
    """Tests for GMOCoinClient retry behavior."""

    def test_client_initialization_with_retry_config(self):
        """Test client accepts retry configuration."""
        config = RetryConfig(max_retries=5)
        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=config,
        )

        assert client.retry_config.max_retries == 5
        client.close()

    def test_client_uses_default_retry_config(self):
        """Test client uses default retry config when not specified."""
        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
        )

        assert client.retry_config.max_retries == 3
        client.close()

    @patch("httpx.Client.get")
    def test_retry_on_timeout(self, mock_get):
        """Test retry on timeout error."""
        # First call times out, second succeeds
        mock_get.side_effect = [
            httpx.TimeoutException("Timeout"),
            Mock(
                json=lambda: {"status": 0, "data": [{"symbol": "BTC_JPY", "ask": "10000000", "bid": "9999000", "high": "10100000", "low": "9900000", "last": "10000000", "volume": "100", "timestamp": "2024-01-01T00:00:00Z"}]},
                raise_for_status=lambda: None,
            ),
        ]

        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        # Should succeed on retry
        ticker = client.get_ticker("BTC_JPY")
        assert ticker.symbol == "BTC_JPY"

        # Should have been called twice
        assert mock_get.call_count == 2

        client.close()

    @patch("httpx.Client.get")
    def test_max_retries_exceeded(self, mock_get):
        """Test exception raised when max retries exceeded."""
        mock_get.side_effect = httpx.TimeoutException("Timeout")

        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=RetryConfig(max_retries=2, base_delay=0.01),
        )

        with pytest.raises(httpx.TimeoutException):
            client.get_ticker("BTC_JPY")

        # 1 initial + 2 retries = 3 total calls
        assert mock_get.call_count == 3

        client.close()

    @patch("httpx.Client.get")
    def test_no_retry_on_non_retryable_error(self, mock_get):
        """Test no retry on non-retryable errors."""
        mock_get.side_effect = ValueError("Invalid")

        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        with pytest.raises(ValueError):
            client.get_ticker("BTC_JPY")

        # Should only be called once (no retries)
        assert mock_get.call_count == 1

        client.close()


class TestGMOCoinClientErrorHandling:
    """Tests for GMOCoinClient error handling."""

    @patch("httpx.Client.get")
    def test_gmo_api_error_raised(self, mock_get):
        """Test GMO API error is properly raised."""
        mock_get.return_value = Mock(
            json=lambda: {
                "status": 5,  # Error status
                "messages": ["System unavailable"],
            },
            raise_for_status=lambda: None,
        )

        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=RetryConfig(max_retries=0),  # No retries
        )

        with pytest.raises(APIError) as exc_info:
            client.get_ticker("BTC_JPY")

        assert exc_info.value.gmo_error_code == 5
        client.close()

    @patch("httpx.Client.get")
    def test_retryable_gmo_error_is_retried(self, mock_get):
        """Test retryable GMO errors are retried."""
        # First call returns error, second succeeds
        mock_get.side_effect = [
            Mock(
                json=lambda: {"status": 1, "messages": ["System error"]},  # Retryable
                raise_for_status=lambda: None,
            ),
            Mock(
                json=lambda: {"status": 0, "data": [{"symbol": "BTC_JPY", "ask": "10000000", "bid": "9999000", "high": "10100000", "low": "9900000", "last": "10000000", "volume": "100", "timestamp": "2024-01-01T00:00:00Z"}]},
                raise_for_status=lambda: None,
            ),
        ]

        client = GMOCoinClient(
            api_key="test_key",
            api_secret="test_secret",
            retry_config=RetryConfig(max_retries=3, base_delay=0.01),
        )

        # Should succeed on retry
        ticker = client.get_ticker("BTC_JPY")
        assert ticker.symbol == "BTC_JPY"

        # Should have been called twice
        assert mock_get.call_count == 2

        client.close()


class TestRetryableConstants:
    """Test retryable error code constants."""

    def test_retryable_status_codes(self):
        """Verify all expected retryable HTTP status codes."""
        expected = {408, 429, 500, 502, 503, 504}
        assert RETRYABLE_STATUS_CODES == expected

    def test_retryable_gmo_error_codes(self):
        """Verify all expected retryable GMO error codes."""
        expected = {1, 4, 5}  # System error, maintenance, temporary unavailable
        assert RETRYABLE_GMO_ERROR_CODES == expected
