"""GMO Coin API Client."""

import hashlib
import hmac
import time
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from pydantic import BaseModel


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
    ) -> None:
        """Initialize GMO Coin client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.private_url = private_url
        self._client = httpx.Client(timeout=30.0)

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

        if method == "GET":
            response = self._client.get(url, headers=headers, params=params)
        elif method == "POST":
            headers["Content-Type"] = "application/json"
            response = self._client.post(url, headers=headers, json=body)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    def _public_request(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make request to public API."""
        url = f"{self.base_url}{path}"
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    # Public API Methods

    def get_ticker(self, symbol: str = "BTC_JPY") -> Ticker:
        """Get current ticker information."""
        result = self._public_request("/v1/ticker", {"symbol": symbol})

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

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

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

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

    def get_orderbooks(self, symbol: str = "BTC_JPY") -> dict[str, Any]:
        """Get order book data."""
        result = self._public_request("/v1/orderbooks", {"symbol": symbol})

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

        return result["data"]

    # Private API Methods

    def get_balance(self) -> list[Balance]:
        """Get account balance."""
        result = self._private_request("GET", "/v1/account/assets")

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

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

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

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

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        body = {"orderId": order_id}
        result = self._private_request("POST", "/v1/cancelOrder", body=body)

        if result["status"] != 0:
            logger.warning(f"Cancel order failed: {result.get('messages', 'Unknown error')}")
            return False
        return True

    def get_active_orders(self, symbol: str = "BTC_JPY") -> list[Order]:
        """Get active orders."""
        result = self._private_request("GET", "/v1/activeOrders", params={"symbol": symbol})

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

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

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

        return result["data"]["list"]

    def get_positions(self, symbol: str = "BTC_JPY") -> list[dict[str, Any]]:
        """Get current positions (for margin trading)."""
        result = self._private_request("GET", "/v1/openPositions", params={"symbol": symbol})

        if result["status"] != 0:
            raise RuntimeError(f"API Error: {result.get('messages', 'Unknown error')}")

        return result["data"]["list"]

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "GMOCoinClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
