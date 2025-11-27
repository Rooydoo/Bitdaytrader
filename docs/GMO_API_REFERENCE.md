# GMOコイン API リファレンス

## 1. 概要

### 1.1 エンドポイント

| 種別 | URL |
|------|-----|
| Public REST API | `https://api.coin.z.com/public/v1` |
| Private REST API | `https://api.coin.z.com/private/v1` |
| Public WebSocket | `wss://api.coin.z.com/ws/public/v1` |
| Private WebSocket | `wss://api.coin.z.com/ws/private/v1` |

### 1.2 認証

Private APIでは以下のHTTPヘッダーが必要：

```python
headers = {
    "API-KEY": "your_api_key",
    "API-TIMESTAMP": str(timestamp_ms),
    "API-SIGN": signature
}
```

**署名生成方法:**
```python
import hmac
import hashlib
import time

def create_signature(secret_key: str, timestamp: int, method: str, path: str, body: str = "") -> str:
    """
    GMOコインAPI用の署名を生成

    Args:
        secret_key: APIシークレットキー
        timestamp: Unixタイムスタンプ（ミリ秒）
        method: HTTPメソッド（GET/POST）
        path: APIパス（例: /v1/order）
        body: リクエストボディ（POSTの場合）
    """
    text = str(timestamp) + method + path + body
    signature = hmac.new(
        secret_key.encode('utf-8'),
        text.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature
```

### 1.3 レート制限

| Tier | 条件 | 制限 |
|------|------|------|
| Tier 1 | 取引高 < 10億円 | 20リクエスト/秒（GET/POSTそれぞれ）|
| Tier 2 | 取引高 ≧ 10億円 | 30リクエスト/秒（GET/POSTそれぞれ）|

WebSocket: 同一IPから1秒に1リクエスト

---

## 2. Public API

### 2.1 取引所ステータス

```
GET /v1/status
```

**レスポンス:**
```json
{
  "status": 0,
  "data": {
    "status": "OPEN"  // MAINTENANCE, PREOPEN, OPEN
  }
}
```

### 2.2 最新レート取得

```
GET /v1/ticker?symbol=BTC
```

**パラメータ:**
- `symbol` (optional): 通貨ペア（省略時は全銘柄）

**レスポンス:**
```json
{
  "status": 0,
  "data": [
    {
      "symbol": "BTC",
      "ask": "5000000",
      "bid": "4999000",
      "high": "5100000",
      "low": "4900000",
      "last": "5000000",
      "timestamp": "2024-01-01T00:00:00.000Z",
      "volume": "100.5"
    }
  ]
}
```

### 2.3 板情報取得

```
GET /v1/orderbooks?symbol=BTC
```

**レスポンス:**
```json
{
  "status": 0,
  "data": {
    "asks": [
      {"price": "5001000", "size": "0.5"},
      {"price": "5002000", "size": "1.0"}
    ],
    "bids": [
      {"price": "4999000", "size": "0.8"},
      {"price": "4998000", "size": "1.2"}
    ],
    "symbol": "BTC",
    "timestamp": "2024-01-01T00:00:00.000Z"
  }
}
```

### 2.4 取引履歴

```
GET /v1/trades?symbol=BTC&page=1&count=100
```

### 2.5 KLine（ローソク足）

```
GET /v1/klines?symbol=BTC&interval=1min&date=20240101
```

**interval値:**
- `1min`, `5min`, `10min`, `15min`, `30min`
- `1hour`, `4hour`, `8hour`, `12hour`
- `1day`, `1week`, `1month`

**レスポンス:**
```json
{
  "status": 0,
  "data": [
    {
      "openTime": "2024-01-01T00:00:00.000Z",
      "open": "5000000",
      "high": "5010000",
      "low": "4990000",
      "close": "5005000",
      "volume": "10.5"
    }
  ]
}
```

### 2.6 取引ルール

```
GET /v1/symbols
```

**レスポンス:**
```json
{
  "status": 0,
  "data": [
    {
      "symbol": "BTC",
      "minOrderSize": "0.0001",
      "maxOrderSize": "5",
      "sizeStep": "0.0001",
      "tickSize": "1",
      "takerFee": "0.0005",
      "makerFee": "0.0003"
    }
  ]
}
```

---

## 3. Private API

### 3.1 余力情報取得

```
GET /v1/account/margin
```

**レスポンス:**
```json
{
  "status": 0,
  "data": {
    "actualProfitLoss": "1000000",
    "availableAmount": "900000",
    "margin": "100000",
    "marginRatio": "900"
  }
}
```

### 3.2 資産残高取得

```
GET /v1/account/assets
```

**レスポンス:**
```json
{
  "status": 0,
  "data": [
    {
      "symbol": "JPY",
      "amount": "1000000",
      "available": "900000"
    },
    {
      "symbol": "BTC",
      "amount": "0.5",
      "available": "0.3"
    }
  ]
}
```

### 3.3 注文

```
POST /v1/order
```

**リクエストボディ:**
```json
{
  "symbol": "BTC",
  "side": "BUY",          // BUY, SELL
  "executionType": "LIMIT", // MARKET, LIMIT, STOP
  "price": "5000000",      // LIMIT時必須
  "size": "0.01"
}
```

**レスポンス:**
```json
{
  "status": 0,
  "data": "12345678"  // 注文ID
}
```

### 3.4 注文変更

```
POST /v1/changeOrder
```

**リクエストボディ:**
```json
{
  "orderId": "12345678",
  "price": "5100000"
}
```

### 3.5 注文キャンセル

```
POST /v1/cancelOrder
```

**リクエストボディ:**
```json
{
  "orderId": "12345678"
}
```

### 3.6 注文一覧取得

```
GET /v1/activeOrders?symbol=BTC&page=1&count=100
```

### 3.7 約定一覧取得

```
GET /v1/executions?orderId=12345678
```

### 3.8 建玉一覧取得

```
GET /v1/openPositions?symbol=BTC&page=1&count=100
```

### 3.9 決済注文

```
POST /v1/closeOrder
```

**リクエストボディ:**
```json
{
  "symbol": "BTC",
  "side": "SELL",
  "executionType": "MARKET",
  "size": "0.01",
  "positionId": "123456"
}
```

### 3.10 一括決済

```
POST /v1/closeBulkOrder
```

---

## 4. WebSocket API

### 4.1 Public WebSocket

**接続:**
```
wss://api.coin.z.com/ws/public/v1
```

**購読メッセージ:**
```json
{
  "command": "subscribe",
  "channel": "ticker",
  "symbol": "BTC"
}
```

**チャンネル種別:**
- `ticker` - 最新レート
- `orderbooks` - 板情報
- `trades` - 取引履歴

**受信データ例（ticker）:**
```json
{
  "channel": "ticker",
  "symbol": "BTC",
  "ask": "5000000",
  "bid": "4999000",
  "high": "5100000",
  "low": "4900000",
  "last": "5000000",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "volume": "100.5"
}
```

### 4.2 Private WebSocket

**1. アクセストークン取得（REST API）:**
```
POST /v1/ws-auth
```

**2. WebSocket接続:**
```
wss://api.coin.z.com/ws/private/v1/{token}
```

**購読メッセージ:**
```json
{
  "command": "subscribe",
  "channel": "executionEvents"
}
```

**チャンネル種別:**
- `executionEvents` - 約定通知
- `orderEvents` - 注文イベント
- `positionEvents` - ポジションイベント
- `positionSummaryEvents` - ポジションサマリー

---

## 5. 取り扱い通貨

### 5.1 現物取引

| 通貨 | シンボル | 最小注文数量 |
|------|---------|-------------|
| ビットコイン | BTC | 0.0001 |
| イーサリアム | ETH | 0.001 |
| ライトコイン | LTC | 0.01 |
| リップル | XRP | 1 |
| ビットコインキャッシュ | BCH | 0.001 |
| ネム | XEM | 1 |
| ステラルーメン | XLM | 1 |
| ベーシックアテンショントークン | BAT | 1 |
| オーエムジー | OMG | 0.1 |
| テゾス | XTZ | 0.1 |
| クアンタム | QTUM | 0.1 |
| エンジンコイン | ENJ | 1 |
| ポルカドット | DOT | 0.1 |
| コスモス | ATOM | 0.1 |
| メイカー | MKR | 0.001 |
| ダイ | DAI | 1 |
| チェーンリンク | LINK | 0.1 |
| モナコイン | MONA | 0.1 |
| カルダノ | ADA | 1 |
| ソラナ | SOL | 0.01 |
| ドージコイン | DOGE | 1 |
| アスター | ASTR | 1 |
| ファイルコイン | FIL | 0.01 |
| サンドボックス | SAND | 1 |

### 5.2 レバレッジ取引

| 通貨 | シンボル | レバレッジ |
|------|---------|-----------|
| ビットコイン | BTC_JPY | 2倍 |
| イーサリアム | ETH_JPY | 2倍 |
| その他 | *_JPY | 2倍 |

---

## 6. エラーコード

| コード | 説明 |
|--------|------|
| 0 | 成功 |
| 1 | エラー |
| 2 | メンテナンス中 |
| 3 | レート制限超過 |
| 4 | 認証エラー |
| 5 | パラメータエラー |
| 6 | 残高不足 |
| 7 | 注文不可 |

---

## 7. 実装上の注意

### 7.1 タイムスタンプ
- ミリ秒単位のUnixタイムスタンプを使用
- サーバー時刻との差が大きいとエラー

### 7.2 小数点精度
- 価格・数量は文字列で送受信
- Decimal型での処理を推奨

### 7.3 WebSocket再接続
- 定期的にpingを送信
- 切断時は自動再接続ロジックを実装

### 7.4 注文サイズ
- 各通貨の最小注文数量・刻み幅を確認
- `/v1/symbols`で最新情報を取得
