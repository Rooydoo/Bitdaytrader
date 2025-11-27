# 技術スタック詳細

## 1. 言語・ランタイム

| 項目 | 選択 | バージョン | 理由 |
|------|------|-----------|------|
| 言語 | Python | 3.11+ | 型ヒント強化、パフォーマンス向上 |
| パッケージ管理 | Poetry or pip | - | 依存関係管理 |

---

## 2. 主要ライブラリ

### 2.1 API通信

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `httpx` | HTTP クライアント | async対応、モダン |
| `websockets` | WebSocket クライアント | 非同期WebSocket |
| `aiohttp` | 代替HTTPクライアント | 必要に応じて |

### 2.2 データ処理・分析

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `pandas` | データ操作 | 時系列データ処理 |
| `numpy` | 数値計算 | 高速配列演算 |
| `scipy` | 統計分析 | コインテグレーション検定等 |
| `statsmodels` | 統計モデル | 回帰分析、時系列分析 |
| `ta-lib` | テクニカル指標 | 高速指標計算（要別途インストール）|
| `pandas-ta` | テクニカル指標 | ta-lib代替（pure Python）|

### 2.3 データベース

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `sqlalchemy` | ORM | データベース抽象化 |
| `asyncpg` | PostgreSQL async | 非同期PostgreSQL |
| `alembic` | マイグレーション | スキーマ管理 |
| `redis` | Redis クライアント | キャッシュ・リアルタイムデータ |
| `aioredis` | Redis async | 非同期Redis |

### 2.4 バックテスト

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `vectorbt` | 高速バックテスト | ベクトル化演算 |
| `backtrader` | 代替バックテスト | イベント駆動型 |

### 2.5 ダッシュボード

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `streamlit` | ダッシュボードUI | 簡単構築、リアルタイム更新 |
| `plotly` | チャート | インタラクティブチャート |
| `altair` | 代替チャート | 宣言的可視化 |

### 2.6 スケジューリング

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `apscheduler` | タスクスケジューラ | 定期実行 |
| `celery` | 分散タスク | 重い処理の非同期実行（オプション）|

### 2.7 通知

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `python-telegram-bot` | Telegram通知 | アラート送信 |
| `discord.py` | Discord通知 | オプション |

### 2.8 開発・テスト

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `pytest` | テストフレームワーク | 単体・統合テスト |
| `pytest-asyncio` | 非同期テスト | async関数テスト |
| `pytest-cov` | カバレッジ | テストカバレッジ計測 |
| `mypy` | 型チェック | 静的型検査 |
| `ruff` | リンター | 高速リンター |
| `black` | フォーマッター | コード整形 |
| `pre-commit` | Git hooks | コミット前チェック |

### 2.9 設定・ログ

| ライブラリ | 用途 | 備考 |
|-----------|------|------|
| `pydantic` | 設定バリデーション | 型安全な設定 |
| `pydantic-settings` | 環境変数管理 | .env読み込み |
| `structlog` | 構造化ログ | JSON形式ログ |
| `loguru` | 代替ロギング | シンプルで高機能 |

---

## 3. インフラストラクチャ

### 3.1 データベース

```yaml
# docker-compose.yml (抜粋)
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: bitdaytrader
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
```

### 3.2 PostgreSQL スキーマ設計

```sql
-- 価格データ（時系列）
CREATE TABLE ohlcv (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    UNIQUE(symbol, timestamp)
);
CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp DESC);

-- 注文履歴
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- BUY, SELL
    order_type VARCHAR(20) NOT NULL,  -- MARKET, LIMIT
    price DECIMAL(20, 8),
    size DECIMAL(20, 8) NOT NULL,
    filled_size DECIMAL(20, 8) DEFAULT 0,
    status VARCHAR(20) NOT NULL,  -- PENDING, FILLED, CANCELLED
    strategy VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ポジション
CREATE TABLE positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_size DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8),
    strategy VARCHAR(50),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    UNIQUE(symbol, side, strategy, opened_at)
);

-- 取引履歴
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    executed_at TIMESTAMPTZ NOT NULL
);

-- 日次パフォーマンス
CREATE TABLE daily_performance (
    id BIGSERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    starting_balance DECIMAL(20, 8),
    ending_balance DECIMAL(20, 8),
    pnl DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    trade_count INTEGER,
    win_count INTEGER,
    max_drawdown DECIMAL(10, 4)
);

-- バックテスト結果
CREATE TABLE backtest_results (
    id BIGSERIAL PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    params JSONB,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    trade_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.3 Redis データ構造

```python
# キー設計
REDIS_KEYS = {
    # リアルタイム価格
    "price:{symbol}": "最新価格（Hash: bid, ask, last, timestamp）",

    # 板情報
    "orderbook:{symbol}": "板情報（Sorted Set）",

    # ポジションキャッシュ
    "positions": "現在のポジション（Hash）",

    # システム状態
    "system:status": "システム稼働状態",
    "system:last_heartbeat": "最終ハートビート",

    # 一時データ
    "temp:signals": "未処理シグナル（List）",
}
```

---

## 4. requirements.txt

```txt
# Core
python-dotenv>=1.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# API Client
httpx>=0.26.0
websockets>=12.0
aiohttp>=3.9.0

# Data Processing
pandas>=2.1.0
numpy>=1.26.0
scipy>=1.11.0
statsmodels>=0.14.0
pandas-ta>=0.3.14b

# Database
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
redis>=5.0.0

# Backtest
vectorbt>=0.26.0

# Dashboard
streamlit>=1.29.0
plotly>=5.18.0

# Scheduling
apscheduler>=3.10.0

# Notifications
python-telegram-bot>=20.7

# Logging
structlog>=23.2.0
loguru>=0.7.0

# Async
asyncio>=3.4.3
```

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
mypy>=1.7.0
ruff>=0.1.0
black>=23.12.0
pre-commit>=3.6.0
ipython>=8.18.0
jupyter>=1.0.0
```

---

## 5. 非同期処理設計

```python
# 基本的な非同期パターン
import asyncio
from typing import AsyncGenerator

class TradingBot:
    async def run(self):
        """メインループ"""
        tasks = [
            asyncio.create_task(self._price_stream()),
            asyncio.create_task(self._signal_processor()),
            asyncio.create_task(self._risk_monitor()),
            asyncio.create_task(self._heartbeat()),
        ]
        await asyncio.gather(*tasks)

    async def _price_stream(self) -> AsyncGenerator:
        """WebSocketから価格データを受信"""
        async for price in self.ws_client.subscribe():
            await self.on_price_update(price)

    async def _signal_processor(self):
        """シグナルを処理して注文を発行"""
        while True:
            signal = await self.signal_queue.get()
            if await self.risk_manager.check(signal):
                await self.execution_engine.execute(signal)

    async def _risk_monitor(self):
        """定期的なリスクチェック"""
        while True:
            await self.risk_manager.monitor()
            await asyncio.sleep(1)

    async def _heartbeat(self):
        """システムヘルスチェック"""
        while True:
            await self.health_check()
            await asyncio.sleep(5)
```

---

## 6. 選択理由まとめ

| 項目 | 選択 | 理由 |
|------|------|------|
| HTTP Client | httpx | モダン、async対応、タイムアウト管理が優秀 |
| ORM | SQLAlchemy 2.0 | async対応、型ヒント強化 |
| バックテスト | vectorbt | 高速（NumPy/Numba）、デイトレ向き |
| ダッシュボード | Streamlit | 開発速度、リアルタイム更新が簡単 |
| ログ | structlog | 構造化ログ、デバッグしやすい |
| 設定管理 | pydantic-settings | 型安全、バリデーション自動 |
