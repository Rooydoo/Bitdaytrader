"""GMO Coin API Client."""

import hashlib
import hmac
import random
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx
import pandas as pd
from loguru import logger
from pydantic import BaseModel

# Type variable for generic return type
T = TypeVar("T")

# Retryable error codes (temporary errors)
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RETRYABLE_GMO_ERROR_CODES = {
    1,  # System error
    4,  # Under maintenance
    5,  # Temporary unavailable
}


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delay
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)


class APIError(Exception):
    """Custom exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        gmo_error_code: int | None = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.gmo_error_code = gmo_error_code
        self.is_retryable = is_retryable


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(error, APIError):
        return error.is_retryable

    if isinstance(error, httpx.TimeoutException):
        return True

    if isinstance(error, httpx.ConnectError):
        return True

    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES

    return False


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add retry logic to API methods."""

    @wraps(func)
    def wrapper(self: "GMOCoinClient", *args: Any, **kwargs: Any) -> T:
        last_error: Exception | None = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func(self, *args, **kwargs)

            except Exception as e:
                last_error = e

                if not is_retryable_error(e):
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"Retryable error in {func.__name__} (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Max retries exceeded for {func.__name__}: {e}"
                    )

        if last_error:
            raise last_error
        raise RuntimeError(f"Unexpected error in {func.__name__}")

    return wrapper


class Ticker(BaseModel):
    """Ticker data model."""

    symbol: str
    ask: float
    bid: float
    high: float
    low: float
    last: float
    volume: float
    timestamp: datetime


class Balance(BaseModel):
    """Balance data model."""

    currency: str
    amount: float
    available: float


class Order(BaseModel):
    """Order data model."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    price: float | None
    size: float
    executed_size: float
    status: str
    timestamp: datetime


class GMOCoinClient:
    """GMO Coin API Client for trading operations."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.coin.z.com/public",
        private_url: str = "https://api.coin.z.com/private",
        retry_config: RetryConfig | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize GMO Coin client.

        Args:
            api_key: GMO Coin API key
            api_secret: GMO Coin API secret
            base_url: Public API base URL
            private_url: Private API base URL
            retry_config: Retry configuration (default: 3 retries with exponential backoff)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.private_url = private_url
        self.retry_config = retry_config or RetryConfig()
        self._client = httpx.Client(timeout=timeout)

    def _create_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create HMAC-SHA256 signature for authentication."""
        text = timestamp + method + path + body
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sign

    def _private_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated request to private API."""
        timestamp = str(int(time.time() * 1000))
        body_str = "" if body is None else str(body).replace("'", '"').replace(" ", "")

        signature = self._create_signature(timestamp, method, path, body_str)

        headers = {
            "API-KEY": self.api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": signature,
        }

        url = f"{self.private_url}{path}"

        try:
            if method == "GET":
                response = self._client.get(url, headers=headers, params=params)
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                response = self._client.post(url, headers=headers, json=body)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            result = response.json()

            # Check GMO API response status
            if result.get("status") != 0:
                gmo_code = result.get("status")
                messages = result.get("messages", [])
                error_msg = messages[0] if messages else "Unknown error"
                is_retryable = gmo_code in RETRYABLE_GMO_ERROR_CODES

                raise APIError(
                    message=f"GMO API Error: {error_msg}",
                    gmo_error_code=gmo_code,
                    is_retryable=is_retryable,
                )

            return result

        except httpx.HTTPStatusError as e:
            is_retryable = e.response.status_code in RETRYABLE_STATUS_CODES
            raise APIError(
                message=f"HTTP Error: {e.response.status_code}",
                status_code=e.response.status_code,
                is_retryable=is_retryable,
            ) from e

    def _public_request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make request to public API."""
        url = f"{self.base_url}{path}"

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            result = response.json()

            # Check GMO API response status
            if result.get("status") != 0:
                gmo_code = result.get("status")
                messages = result.get("messages", [])
                error_msg = messages[0] if messages else "Unknown error"
                is_retryable = gmo_code in RETRYABLE_GMO_ERROR_CODES

                raise APIError(
                    message=f"GMO API Error: {error_msg}",
                    gmo_error_code=gmo_code,
                    is_retryable=is_retryable,
                )

            return result

        except httpx.HTTPStatusError as e:
            is_retryable = e.response.status_code in RETRYABLE_STATUS_CODES
            raise APIError(
                message=f"HTTP Error: {e.response.status_code}",
                status_code=e.response.status_code,
                is_retryable=is_retryable,
            ) from e

    # Public API Methods

    @with_retry
    def get_ticker(self, symbol: str = "BTC_JPY") -> Ticker:
        """Get current ticker information."""
        result = self._public_request("/v1/ticker", {"symbol": symbol})
        data = result["data"][0]
        return Ticker(
            symbol=data["symbol"],
            ask=float(data["ask"]),
            bid=float(data["bid"]),
            high=float(data["high"]),
            low=float(data["low"]),
            last=float(data["last"]),
            volume=float(data["volume"]),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )

    @with_retry
    def get_klines(
        self,
        symbol: str = "BTC_JPY",
        interval: str = "15min",
        date: str | None = None,
    ) -> pd.DataFrame:
        """
        Get candlestick (kline) data.

        Args:
            symbol: Trading symbol (e.g., BTC_JPY)
            interval: Candle interval (1min, 5min, 10min, 15min, 30min, 1hour, 4hour, 8hour, 12hour, 1day, 1week, 1month)
            date: Date in YYYYMMDD format (default: today)

        Returns:
            DataFrame with OHLCV data
        """
        params = {"symbol": symbol, "interval": interval}
        if date:
            params["date"] = date

        result = self._public_request("/v1/klines", params)
        data = result["data"]
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @with_retry
    def get_orderbooks(self, symbol: str = "BTC_JPY") -> dict[str, Any]:
        """Get order book data."""
        result = self._public_request("/v1/orderbooks", {"symbol": symbol})
        return result["data"]

    # Private API Methods

    @with_retry
    def get_balance(self) -> list[Balance]:
        """Get account balance."""
        result = self._private_request("GET", "/v1/account/assets")
        balances = []
        for item in result["data"]:
            balances.append(
                Balance(
                    currency=item["symbol"],
                    amount=float(item["amount"]),
                    available=float(item["available"]),
                )
            )
        return balances

    def get_jpy_balance(self) -> float:
        """Get JPY balance."""
        balances = self.get_balance()
        for b in balances:
            if b.currency == "JPY":
                return b.available
        return 0.0

    @with_retry
    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "LIMIT",
        price: float | None = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """
        Place an order.

        Args:
            symbol: Trading symbol (e.g., BTC_JPY)
            side: BUY or SELL
            size: Order size
            order_type: MARKET or LIMIT
            price: Limit price (required for LIMIT orders)
            time_in_force: GTC (Good Till Cancel) or IOC (Immediate Or Cancel)

        Returns:
            Order object
        """
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "executionType": order_type,
            "size": str(size),
        }

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            body["price"] = str(int(price))
            body["timeInForce"] = time_in_force

        result = self._private_request("POST", "/v1/order", body=body)
        data = result["data"]
        return Order(
            order_id=data,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            executed_size=0.0,
            status="ORDERED",
            timestamp=datetime.now(),
        )

    @with_retry
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        body = {"orderId": order_id}
        try:
            self._private_request("POST", "/v1/cancelOrder", body=body)
            return True
        except APIError as e:
            logger.warning(f"Cancel order failed: {e}")
            return False

    @with_retry
    def get_active_orders(self, symbol: str = "BTC_JPY") -> list[Order]:
        """Get active orders."""
        result = self._private_request("GET", "/v1/activeOrders", params={"symbol": symbol})
        orders = []
        for item in result["data"]["list"]:
            orders.append(
                Order(
                    order_id=str(item["orderId"]),
                    symbol=item["symbol"],
                    side=item["side"],
                    order_type=item["executionType"],
                    price=float(item["price"]) if item.get("price") else None,
                    size=float(item["size"]),
                    executed_size=float(item["executedSize"]),
                    status=item["status"],
                    timestamp=datetime.fromisoformat(
                        item["timestamp"].replace("Z", "+00:00")
                    ),
                )
            )
        return orders

    @with_retry
    def get_executions(
        self,
        order_id: str | None = None,
        symbol: str = "BTC_JPY",
    ) -> list[dict[str, Any]]:
        """Get execution history."""
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id

        result = self._private_request("GET", "/v1/executions", params=params)
        return result["data"]["list"]

    @with_retry
    def get_positions(self, symbol: str = "BTC_JPY") -> list[dict[str, Any]]:
        """Get current positions (for margin trading)."""
        result = self._private_request("GET", "/v1/openPositions", params={"symbol": symbol})
        return result["data"]["list"]

    # Margin Trading Methods

    @with_retry
    def get_margin_status(self) -> dict[str, Any]:
        """Get margin account status."""
        result = self._private_request("GET", "/v1/account/margin")
        return result["data"]

    @with_retry
    def place_margin_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "LIMIT",
        price: float | None = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """
        Place a margin order (for leverage trading).

        Args:
            symbol: Trading symbol (e.g., BTC_JPY)
            side: BUY (long) or SELL (short)
            size: Order size
            order_type: MARKET or LIMIT
            price: Limit price (required for LIMIT orders)
            time_in_force: GTC or IOC

        Returns:
            Order object
        """
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "executionType": order_type,
            "size": str(size),
        }

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            body["price"] = str(int(price))
            body["timeInForce"] = time_in_force

        result = self._private_request("POST", "/v1/order", body=body)
        data = result["data"]
        return Order(
            order_id=data,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            executed_size=0.0,
            status="ORDERED",
            timestamp=datetime.now(),
        )

    @with_retry
    def close_margin_position(
        self,
        symbol: str,
        side: str,
        size: float,
        position_id: str | None = None,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> Order:
        """
        Close a margin position.

        Args:
            symbol: Trading symbol
            side: BUY (to close short) or SELL (to close long)
            size: Size to close
            position_id: Specific position ID to close (optional)
            order_type: MARKET or LIMIT
            price: Limit price (for LIMIT orders)

        Returns:
            Order object
        """
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "executionType": order_type,
            "size": str(size),
            "settlePosition": [{"size": str(size)}],
        }

        if position_id:
            body["settlePosition"] = [{"positionId": position_id, "size": str(size)}]

        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            body["price"] = str(int(price))

        result = self._private_request("POST", "/v1/closeOrder", body=body)
        data = result["data"]
        return Order(
            order_id=data,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            executed_size=0.0,
            status="ORDERED",
            timestamp=datetime.now(),
        )

    def close_all_positions(self, symbol: str = "BTC_JPY") -> list[Order]:
        """
        Close all open positions for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of close orders
        """
        positions = self.get_positions(symbol)
        orders = []

        for pos in positions:
            # Determine close side (opposite of position side)
            close_side = "SELL" if pos["side"] == "BUY" else "BUY"
            size = float(pos["size"]) - float(pos.get("executedSize", 0))

            if size > 0:
                order = self.close_margin_position(
                    symbol=symbol,
                    side=close_side,
                    size=size,
                    position_id=str(pos["positionId"]),
                    order_type="MARKET",
                )
                orders.append(order)
                logger.info(f"Closed position {pos['positionId']}: {close_side} {size}")

        return orders

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "GMOCoinClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
