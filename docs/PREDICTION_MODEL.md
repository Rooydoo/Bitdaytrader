# 予測モデル設計

## 1. 設計方針

### 1.1 リソース制約

| 項目 | 制約 |
|------|------|
| RAM | 8GB（共有、他プロセスあり）|
| CPU | 2コア（共有）|
| このシステム割当 | RAM 2-3GB, CPU 50% |

### 1.2 要件

- **軽量**: メモリ使用量 < 500MB
- **高速推論**: < 100ms / 予測
- **定期学習**: 日次でモデル更新
- **解釈可能性**: 取引判断の説明が可能

---

## 2. モデル構成

### 2.1 アンサンブル構成

```
┌─────────────────────────────────────────────────┐
│              Ensemble Predictor                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ LightGBM  │  │   Ridge   │  │  GARCH    │   │
│  │ (方向予測) │  │ (リターン) │  │(ボラ予測) │   │
│  │           │  │           │  │           │   │
│  │ Weight:   │  │ Weight:   │  │ Weight:   │   │
│  │   0.5     │  │   0.3     │  │   0.2     │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │          │
│        └──────────────┼──────────────┘          │
│                       ▼                         │
│              ┌───────────────┐                  │
│              │  Final Signal │                  │
│              │  (加重平均)    │                  │
│              └───────────────┘                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 2.2 各モデルの役割

| モデル | 目的 | 出力 | メモリ |
|--------|------|------|--------|
| LightGBM | 価格方向予測 | 確率 (0-1) | ~100MB |
| Ridge | リターン予測 | 数値 | ~10MB |
| GARCH | ボラティリティ予測 | 数値 | ~20MB |

---

## 3. LightGBM 方向予測モデル

### 3.1 目的

次のN期間で価格が上昇するか下降するかを予測。

### 3.2 特徴量

```python
FEATURES = {
    # 価格系（20特徴）
    "price": [
        "return_1h", "return_4h", "return_24h",      # リターン
        "log_return_1h", "log_return_4h",            # 対数リターン
        "high_low_ratio", "close_open_ratio",        # 比率
        "price_vs_ema9", "price_vs_ema21",           # EMA乖離
        "price_vs_ema55",
    ],

    # テクニカル指標（15特徴）
    "technical": [
        "rsi_14", "rsi_7",                           # RSI
        "macd", "macd_signal", "macd_hist",          # MACD
        "bb_upper_dist", "bb_lower_dist",            # ボリンジャー
        "atr_14", "atr_ratio",                       # ATR
        "adx_14",                                    # ADX
        "cci_20",                                    # CCI
        "williams_r",                                # Williams %R
        "stoch_k", "stoch_d",                        # ストキャスティクス
    ],

    # 出来高系（5特徴）
    "volume": [
        "volume_ratio_20",                           # 出来高比率
        "volume_trend",                              # 出来高トレンド
        "obv_slope",                                 # OBV傾き
        "vwap_dist",                                 # VWAP乖離
        "volume_volatility",                         # 出来高ボラ
    ],

    # 時間系（5特徴）
    "time": [
        "hour_sin", "hour_cos",                      # 時間（周期）
        "day_of_week",                               # 曜日
        "is_weekend",                                # 週末フラグ
        "is_asian_session",                          # アジア時間
    ],

    # 市場構造（5特徴）
    "market": [
        "spread_pct",                                # スプレッド
        "orderbook_imbalance",                       # 板の偏り
        "funding_rate",                              # 資金調達率
        "btc_correlation",                           # BTC相関
        "market_regime",                             # 市場レジーム
    ],
}

# 合計: 約50特徴量
```

### 3.3 ラベル

```python
def create_label(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """
    予測ホライズン後のリターンに基づくラベル

    Args:
        horizon: 予測期間（時間足の本数）

    Returns:
        0: 下落（< -0.3%）
        1: 横ばい（-0.3% ~ 0.3%）
        2: 上昇（> 0.3%）
    """
    future_return = df['close'].pct_change(horizon).shift(-horizon)

    labels = pd.cut(
        future_return,
        bins=[-np.inf, -0.003, 0.003, np.inf],
        labels=[0, 1, 2]
    )
    return labels
```

### 3.4 モデルパラメータ

```python
LIGHTGBM_PARAMS = {
    # 軽量設定
    "n_estimators": 100,           # ツリー数（少なめ）
    "max_depth": 6,                # 深さ制限
    "num_leaves": 31,              # 葉の数
    "min_child_samples": 50,       # 最小サンプル数

    # 学習設定
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,

    # 正則化
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,

    # その他
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbose": -1,
    "n_jobs": 1,                   # CPU使用制限
}
```

### 3.5 メモリ最適化

```python
# データ型最適化
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """メモリ使用量を削減"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    return df

# 増分学習（全データをメモリに載せない）
def train_incremental(model, data_generator, epochs: int = 1):
    """
    データをチャンクで読み込みながら学習
    """
    for epoch in range(epochs):
        for chunk in data_generator:
            X, y = prepare_features(chunk)
            model.fit(X, y, init_model=model if epoch > 0 else None)
```

---

## 4. Ridge リターン予測モデル

### 4.1 目的

次のN期間のリターン（%）を数値予測。

### 4.2 特徴量

LightGBMと同じ特徴量を使用（線形モデルなので標準化が必要）。

### 4.3 モデル設定

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RIDGE_PARAMS = {
    "alpha": 1.0,                  # 正則化強度
    "fit_intercept": True,
    "solver": "auto",
}

class RidgePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = Ridge(**RIDGE_PARAMS)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
```

---

## 5. GARCH ボラティリティ予測

### 5.1 目的

将来のボラティリティを予測してリスク管理に活用。

### 5.2 モデル設定

```python
from arch import arch_model

GARCH_PARAMS = {
    "p": 1,                        # GARCH次数
    "q": 1,                        # ARCH次数
    "vol": "Garch",
    "dist": "normal",
}

class GarchPredictor:
    def __init__(self):
        self.model = None

    def fit(self, returns: pd.Series):
        """
        リターン系列からGARCHモデルを学習
        """
        # 直近500本のデータのみ使用（メモリ節約）
        recent_returns = returns.tail(500) * 100  # パーセント変換

        self.model = arch_model(
            recent_returns,
            **GARCH_PARAMS
        )
        self.result = self.model.fit(disp='off')

    def predict(self, horizon: int = 4) -> float:
        """
        将来のボラティリティを予測
        """
        forecast = self.result.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.iloc[-1].mean()) / 100
```

---

## 6. アンサンブル統合

### 6.1 最終シグナル生成

```python
class EnsemblePredictor:
    def __init__(self):
        self.lgbm = LightGBMPredictor()
        self.ridge = RidgePredictor()
        self.garch = GarchPredictor()

        # 重み
        self.weights = {
            "lgbm": 0.5,
            "ridge": 0.3,
            "garch": 0.2,
        }

    def predict(self, features: pd.DataFrame, returns: pd.Series) -> dict:
        """
        統合予測を実行

        Returns:
            {
                "direction": "long" | "short" | "neutral",
                "confidence": 0.0 - 1.0,
                "expected_return": float,
                "expected_volatility": float,
                "risk_adjusted_signal": float,
            }
        """
        # 各モデルの予測
        lgbm_probs = self.lgbm.predict_proba(features)  # [下落, 横ばい, 上昇]
        ridge_return = self.ridge.predict(features)[0]
        garch_vol = self.garch.predict()

        # 方向判定
        direction_score = (
            lgbm_probs[2] - lgbm_probs[0]  # 上昇確率 - 下落確率
        ) * self.weights["lgbm"] + (
            np.sign(ridge_return) * min(abs(ridge_return), 1.0)
        ) * self.weights["ridge"]

        # 信頼度
        confidence = abs(direction_score)

        # 方向決定
        if direction_score > 0.2:
            direction = "long"
        elif direction_score < -0.2:
            direction = "short"
        else:
            direction = "neutral"

        # リスク調整済みシグナル
        risk_adjusted = direction_score / (garch_vol + 0.01)

        return {
            "direction": direction,
            "confidence": confidence,
            "expected_return": ridge_return,
            "expected_volatility": garch_vol,
            "risk_adjusted_signal": risk_adjusted,
            "lgbm_probs": lgbm_probs.tolist(),
        }
```

### 6.2 シグナルと戦略の統合

```python
def combine_with_technical(
    model_signal: dict,
    technical_signal: dict
) -> dict:
    """
    予測モデルとテクニカル分析を統合

    - モデルとテクニカルが一致: 高信頼度でエントリー
    - モデルのみ: 低信頼度でエントリー or 見送り
    - テクニカルのみ: 通常エントリー
    - 矛盾: 見送り
    """
    model_dir = model_signal["direction"]
    tech_dir = technical_signal["direction"]
    model_conf = model_signal["confidence"]

    if model_dir == tech_dir and model_dir != "neutral":
        # 一致: 信頼度ブースト
        return {
            "direction": model_dir,
            "confidence": min(model_conf * 1.5, 1.0),
            "source": "both",
            "action": "strong_entry",
        }

    elif model_dir != "neutral" and tech_dir == "neutral":
        # モデルのみ: 弱いシグナル
        if model_conf > 0.6:
            return {
                "direction": model_dir,
                "confidence": model_conf * 0.7,
                "source": "model",
                "action": "weak_entry",
            }
        return {"action": "skip"}

    elif model_dir == "neutral" and tech_dir != "neutral":
        # テクニカルのみ: 通常エントリー
        return {
            "direction": tech_dir,
            "confidence": technical_signal.get("confidence", 0.5),
            "source": "technical",
            "action": "normal_entry",
        }

    elif model_dir != tech_dir and model_dir != "neutral" and tech_dir != "neutral":
        # 矛盾: 見送り
        return {"action": "skip", "reason": "conflict"}

    return {"action": "skip"}
```

---

## 7. 学習スケジュール

### 7.1 定期学習

```python
TRAINING_SCHEDULE = {
    # LightGBM: 日次更新
    "lgbm": {
        "frequency": "daily",
        "time": "00:00 UTC",        # 市場が比較的静かな時間
        "data_window": 90,          # 過去90日
        "retrain_threshold": 0.02,  # 精度低下2%で再学習
    },

    # Ridge: 日次更新
    "ridge": {
        "frequency": "daily",
        "time": "00:00 UTC",
        "data_window": 30,          # 過去30日
    },

    # GARCH: 4時間ごと更新
    "garch": {
        "frequency": "4hours",
        "data_window": 500,         # 過去500本
    },
}
```

### 7.2 ウォークフォワード検証

```python
def walk_forward_validation(
    data: pd.DataFrame,
    train_window: int = 60,      # 60日
    test_window: int = 7,        # 7日
    step: int = 7                # 7日ずつスライド
) -> list:
    """
    ウォークフォワード検証

    1. 訓練期間でモデル学習
    2. テスト期間で評価
    3. ウィンドウをスライド
    4. 繰り返し
    """
    results = []

    for start in range(0, len(data) - train_window - test_window, step):
        train_end = start + train_window
        test_end = train_end + test_window

        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        # 学習
        model = train_model(train_data)

        # 評価
        metrics = evaluate_model(model, test_data)
        results.append(metrics)

    return results
```

---

## 8. メモリ使用量見積もり

| コンポーネント | 推定メモリ |
|---------------|-----------|
| LightGBM モデル | ~100 MB |
| Ridge モデル | ~10 MB |
| GARCH モデル | ~20 MB |
| 特徴量キャッシュ | ~100 MB |
| データバッファ | ~200 MB |
| **合計** | **~430 MB** |

→ 目標の500MB以内に収まる設計

---

## 9. 推論パイプライン

```python
class InferencePipeline:
    """
    軽量な推論パイプライン
    """

    def __init__(self):
        self.models = load_models()
        self.feature_cache = FeatureCache(max_size=1000)

    async def predict(self, symbol: str) -> dict:
        """
        リアルタイム予測（目標: <100ms）
        """
        start = time.time()

        # 1. 特徴量取得（キャッシュ活用）
        features = await self.get_features(symbol)

        # 2. 予測実行
        signal = self.models.predict(features)

        # 3. 後処理
        result = self.post_process(signal)

        elapsed = (time.time() - start) * 1000
        result["inference_time_ms"] = elapsed

        return result

    async def get_features(self, symbol: str) -> pd.DataFrame:
        """
        特徴量を取得（キャッシュ優先）
        """
        cached = self.feature_cache.get(symbol)
        if cached is not None:
            return cached

        # キャッシュミス: 計算
        features = await compute_features(symbol)
        self.feature_cache.set(symbol, features)

        return features
```

---

## 10. モニタリング

### 10.1 モデル性能監視

```python
MODEL_METRICS = {
    # 精度系
    "accuracy": "方向予測の正解率",
    "precision": "シグナルの適合率",
    "recall": "シグナルの再現率",

    # 収益系
    "signal_pnl": "シグナル従った場合のPnL",
    "sharpe_ratio": "シャープレシオ",

    # 運用系
    "inference_time": "推論時間",
    "memory_usage": "メモリ使用量",
}
```

### 10.2 アラート条件

```python
ALERT_CONDITIONS = {
    # 精度低下
    "accuracy_drop": {
        "threshold": 0.05,          # 5%低下
        "window": "7d",
        "action": "notify + retrain",
    },

    # 推論遅延
    "slow_inference": {
        "threshold_ms": 200,
        "action": "notify",
    },

    # メモリ逼迫
    "high_memory": {
        "threshold_mb": 600,
        "action": "notify + gc",
    },
}
```
