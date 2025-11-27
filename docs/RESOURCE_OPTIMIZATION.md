# 計算資源最適化設計

## 1. リソース制約

### 1.1 VPSスペック

| リソース | 総量 | 他プロセス | 本システム割当 |
|----------|------|-----------|--------------|
| RAM | 8 GB | ~4 GB | **3-4 GB** |
| CPU | 2 コア | ~1 コア | **1 コア** |
| ストレージ | - | - | 10 GB 推奨 |

### 1.2 並行プロセス

```
VPS (8GB RAM, 2 Core)
├── 他のMLモデル群 (~4GB, 1 Core)
│   ├── Model A
│   ├── Model B
│   └── ...
│
└── Bitdaytrader (~3GB, 1 Core)
    ├── Trading Bot
    ├── Prediction Models
    └── Dashboard (optional)
```

---

## 2. メモリ管理

### 2.1 コンポーネント別メモリ割当

| コンポーネント | 最大メモリ | 備考 |
|---------------|-----------|------|
| Trading Bot (常駐) | 500 MB | WebSocket + ロジック |
| Prediction Models | 500 MB | LightGBM + Ridge + GARCH |
| Data Cache | 300 MB | Redis代替のインメモリキャッシュ |
| Historical Data | 500 MB | ローリングウィンドウ |
| Backtest (バッチ) | 1.5 GB | 実行時のみ |
| Dashboard (オプション) | 500 MB | 必要時のみ起動 |

### 2.2 メモリ最適化テクニック

```python
# 1. データ型最適化
import numpy as np
import pandas as pd

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """メモリ使用量を50-70%削減"""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')

    return df


# 2. ローリングウィンドウでデータ保持
class RollingDataStore:
    """メモリ効率の良いデータ保持"""

    def __init__(self, max_rows: int = 10000):
        self.max_rows = max_rows
        self.data = pd.DataFrame()

    def append(self, new_data: pd.DataFrame):
        self.data = pd.concat([self.data, new_data]).tail(self.max_rows)

    def get(self, n: int = None) -> pd.DataFrame:
        if n is None:
            return self.data
        return self.data.tail(n)


# 3. 遅延読み込み
class LazyModelLoader:
    """必要になるまでモデルを読み込まない"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        import joblib
        return joblib.load(self.model_path)

    def unload(self):
        """メモリ解放"""
        self._model = None
        import gc
        gc.collect()
```

### 2.3 ガベージコレクション

```python
import gc

class MemoryManager:
    """メモリ管理ユーティリティ"""

    def __init__(self, threshold_mb: int = 2500):
        self.threshold = threshold_mb * 1024 * 1024

    def check_and_cleanup(self):
        """メモリ使用量をチェックして必要なら解放"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        if memory_info.rss > self.threshold:
            self._cleanup()

    def _cleanup(self):
        gc.collect()

        # キャッシュクリア
        if hasattr(self, 'cache'):
            self.cache.clear()

    def get_memory_usage_mb(self) -> float:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
```

---

## 3. CPU管理

### 3.1 プロセス優先度

```python
import os

def set_low_priority():
    """CPUプライオリティを下げる"""
    try:
        os.nice(10)  # 優先度を下げる（-20 ~ 19、高いほど低優先）
    except PermissionError:
        pass

def set_cpu_affinity(cpu_id: int = 1):
    """特定のCPUコアに固定"""
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([cpu_id])
    except Exception:
        pass
```

### 3.2 並列処理の制限

```python
# 環境変数でスレッド数を制限
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# LightGBMのスレッド制限
lgbm_params = {
    "n_jobs": 1,
    "num_threads": 1,
}

# NumPyのスレッド制限
import numpy as np
# np.set_num_threads(1)  # NumPy 2.0+
```

### 3.3 非同期処理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncExecutor:
    """CPU集約タスクを別スレッドで実行"""

    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_cpu_bound(self, func, *args):
        """CPU集約タスクを非同期実行"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

# 使用例
executor = AsyncExecutor(max_workers=1)

async def predict_async(features):
    return await executor.run_cpu_bound(model.predict, features)
```

---

## 4. データベース最適化

### 4.1 SQLite設定（開発・軽量運用）

```python
# SQLite最適化設定
SQLITE_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;      -- 64MB キャッシュ
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;    -- 256MB メモリマップ
"""

def create_optimized_connection(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(SQLITE_PRAGMAS)
    return conn
```

### 4.2 インメモリキャッシュ（Redis代替）

```python
from collections import OrderedDict
from threading import Lock
import time

class LRUCache:
    """Redis代替のインメモリLRUキャッシュ"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = Lock()

    def get(self, key: str):
        with self.lock:
            if key not in self.cache:
                return None

            # TTLチェック
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None

            # LRU更新
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key: str, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                    del self.timestamps[oldest]

            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

# グローバルキャッシュインスタンス
price_cache = LRUCache(max_size=500, ttl_seconds=60)
feature_cache = LRUCache(max_size=100, ttl_seconds=300)
```

---

## 5. バックテスト最適化

### 5.1 メモリ効率の良いバックテスト

```python
class LightweightBacktester:
    """メモリ効率重視のバックテスター"""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def run(self, data_path: str, strategy) -> dict:
        """
        チャンク単位でデータを処理
        """
        results = []
        position = None

        # データをチャンクで読み込み
        for chunk in pd.read_csv(data_path, chunksize=self.chunk_size):
            chunk = optimize_dtypes(chunk)

            for _, row in chunk.iterrows():
                signal = strategy.on_bar(row)

                if signal and not position:
                    position = self._open_position(row, signal)
                elif position:
                    position = self._update_position(row, position)
                    if position.get('closed'):
                        results.append(position)
                        position = None

            # チャンク処理後にGC
            del chunk
            gc.collect()

        return self._aggregate_results(results)
```

### 5.2 バックテストスケジューリング

```python
# 負荷分散のためのスケジュール
BACKTEST_SCHEDULE = {
    # 深夜帯に重い処理を実行
    "full_backtest": {
        "time": "03:00",           # 午前3時
        "max_duration_minutes": 60,
        "memory_limit_mb": 1500,
    },

    # ウォークフォワードは週末
    "walk_forward": {
        "day": "sunday",
        "time": "02:00",
        "max_duration_minutes": 120,
        "memory_limit_mb": 1500,
    },
}
```

---

## 6. ダッシュボード最適化

### 6.1 オンデマンド起動

```python
# ダッシュボードは常時起動しない
# 必要な時だけ起動するスクリプト

# scripts/start_dashboard.sh
#!/bin/bash
# 既存プロセスチェック
if pgrep -f "streamlit run dashboard" > /dev/null; then
    echo "Dashboard already running"
    exit 0
fi

# メモリ確認
FREE_MEM=$(free -m | awk '/^Mem:/{print $7}')
if [ $FREE_MEM -lt 600 ]; then
    echo "Not enough memory (${FREE_MEM}MB free, need 600MB)"
    exit 1
fi

# 起動
streamlit run dashboard/app.py --server.port 8501 &
echo "Dashboard started on port 8501"
```

### 6.2 軽量ダッシュボード

```python
# Streamlitの軽量設定
# .streamlit/config.toml

[server]
maxUploadSize = 1
maxMessageSize = 10

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false

[global]
disableWatchdogWarning = true
```

---

## 7. 監視とアラート

### 7.1 リソース監視

```python
import psutil
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResourceMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_percent: float

class ResourceMonitor:
    """リソース使用量を監視"""

    def __init__(self):
        self.history = []
        self.alerts = []

    def collect(self) -> ResourceMetrics:
        process = psutil.Process()

        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            memory_percent=process.memory_percent(),
            disk_percent=psutil.disk_usage('/').percent,
        )

        self.history.append(metrics)
        self._check_alerts(metrics)

        # 履歴は直近1000件のみ保持
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        return metrics

    def _check_alerts(self, metrics: ResourceMetrics):
        if metrics.memory_mb > 3000:
            self.alerts.append(f"High memory: {metrics.memory_mb:.0f}MB")

        if metrics.cpu_percent > 80:
            self.alerts.append(f"High CPU: {metrics.cpu_percent:.0f}%")
```

### 7.2 自動スケールダウン

```python
class AdaptiveController:
    """リソース状況に応じて動作を調整"""

    def __init__(self):
        self.monitor = ResourceMonitor()
        self.mode = "normal"  # normal, low_resource, critical

    def update(self):
        metrics = self.monitor.collect()

        if metrics.memory_mb > 3500 or metrics.cpu_percent > 90:
            self.mode = "critical"
            self._enter_critical_mode()
        elif metrics.memory_mb > 2500 or metrics.cpu_percent > 70:
            self.mode = "low_resource"
            self._enter_low_resource_mode()
        else:
            self.mode = "normal"

    def _enter_critical_mode(self):
        """緊急モード: 最小限の機能のみ"""
        # 予測モデルを一時停止
        # バックテストをキャンセル
        # キャッシュをクリア
        gc.collect()

    def _enter_low_resource_mode(self):
        """低リソースモード: 処理頻度を下げる"""
        # 予測頻度を下げる
        # データ取得間隔を延ばす
        pass
```

---

## 8. 設定ファイル

### 8.1 リソース制限設定

```python
# config/resource_limits.py

RESOURCE_LIMITS = {
    # メモリ制限
    "max_memory_mb": 3000,
    "warning_memory_mb": 2500,
    "cache_memory_mb": 300,

    # CPU制限
    "max_cpu_percent": 50,
    "n_threads": 1,
    "cpu_affinity": [1],          # 2番目のコアを使用

    # データ制限
    "max_historical_rows": 10000,
    "max_cache_items": 500,

    # バッチ処理制限
    "backtest_chunk_size": 1000,
    "backtest_max_memory_mb": 1500,
}
```

### 8.2 .env設定

```bash
# .env

# Resource Limits
MAX_MEMORY_MB=3000
N_THREADS=1
CPU_AFFINITY=1

# Feature Flags
ENABLE_DASHBOARD=false          # 必要時のみtrue
ENABLE_BACKTEST=true
ENABLE_PREDICTIONS=true

# Optimization
USE_SQLITE=true                 # PostgreSQL代わりにSQLite
USE_MEMORY_CACHE=true           # Redis代わりにインメモリ
```

---

## 9. 推奨構成

### 9.1 最小構成（常時稼働）

```
Trading Bot:     ~500 MB
Prediction:      ~500 MB
Data Cache:      ~300 MB
SQLite:          ~100 MB
─────────────────────────
合計:            ~1.4 GB
```

### 9.2 フル構成（バックテスト時）

```
Trading Bot:     ~500 MB
Backtest:        ~1.5 GB
SQLite:          ~100 MB
─────────────────────────
合計:            ~2.1 GB
```

→ 3-4GB割当で余裕を持って運用可能
