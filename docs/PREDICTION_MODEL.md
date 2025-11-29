# LightGBM 予測モデル設計

## 1. 設計方針

### 1.1 モデル選択

**LightGBM (Light Gradient Boosting Machine)**

| 項目 | 内容 |
|------|------|
| 選定理由 | メモリ効率が高い（XGBoostの約1/3）、推論速度が速い |
| 目的 | 1時間後の価格上昇/下落を2クラス分類 |
| 特徴量数 | 12個 |
| モデルサイズ | ~5MB |
| 推論時間 | <10ms |

### 1.2 リソース制約

| 項目 | 制約 |
|------|------|
| VPS RAM割当 | 4GB |
| VPS CPU割当 | 1コア |
| モデルメモリ | ~50MB |
| 学習場所 | **ローカルPC**（VPSでは推論のみ）|

---

## 2. ハイパーパラメータ

```python
LIGHTGBM_PARAMS = {
    # 基本設定
    'objective': 'binary',           # 上昇/下落の2クラス分類
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',

    # ツリー構造
    'num_leaves': 31,                # 過学習防止のため控えめ
    'max_depth': 6,
    'min_child_samples': 20,

    # 学習設定
    'learning_rate': 0.05,
    'n_estimators': 100,             # 軽量化
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    # 正則化
    'reg_alpha': 0.1,                # L1正則化
    'reg_lambda': 0.1,               # L2正則化

    # その他
    'n_jobs': 1,                     # シングルコア制約
    'random_state': 42,
    'verbose': -1,
}
```

---

## 3. 特徴量設計

### 3.1 特徴量一覧（12個）

| # | カテゴリ | 特徴量 | 計算方法 | 正規化 |
|---|---------|--------|----------|--------|
| 1 | リターン | return_1 | (close - close[1]) / close[1] | なし |
| 2 | リターン | return_5 | (close - close[5]) / close[5] | なし |
| 3 | リターン | return_15 | (close - close[15]) / close[15] | なし |
| 4 | ボラティリティ | volatility_20 | 20期間リターンの標準偏差 | なし |
| 5 | ボラティリティ | atr_14 | ATR(14) / close | 価格で割る |
| 6 | モメンタム | rsi_14 | RSI(14) | /100 |
| 7 | モメンタム | macd_diff | MACD - Signal | /close |
| 8 | トレンド | ema_ratio | EMA9 / EMA21 | なし |
| 9 | トレンド | bb_position | (close - BB_lower) / BB_width | 0-1 |
| 10 | 出来高 | volume_ratio | volume / SMA(volume, 20) | なし |
| 11 | 時間 | hour | UTC時間 | /23 |
| 12 | 時間 | day_of_week | 曜日 | /6 |

### 3.2 特徴量計算コード

```python
import pandas as pd
import pandas_ta as ta
from typing import List

FEATURE_NAMES: List[str] = [
    'return_1', 'return_5', 'return_15',
    'volatility_20', 'atr_14',
    'rsi_14', 'macd_diff',
    'ema_ratio', 'bb_position',
    'volume_ratio',
    'hour', 'day_of_week'
]

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    12個の特徴量を計算

    Args:
        df: OHLCV データフレーム
            - index: DatetimeIndex
            - columns: open, high, low, close, volume

    Returns:
        特徴量データフレーム (12列)
    """
    features = pd.DataFrame(index=df.index)

    # 1-3. リターン系
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_15'] = df['close'].pct_change(15)

    # 4-5. ボラティリティ
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()
    features['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']

    # 6-7. モメンタム
    features['rsi_14'] = ta.rsi(df['close'], length=14) / 100
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    features['macd_diff'] = macd['MACDh_12_26_9'] / df['close']

    # 8-9. トレンド
    ema9 = ta.ema(df['close'], length=9)
    ema21 = ta.ema(df['close'], length=21)
    features['ema_ratio'] = ema9 / ema21

    bb = ta.bbands(df['close'], length=20, std=2)
    bb_width = bb['BBU_20_2.0'] - bb['BBL_20_2.0']
    features['bb_position'] = (df['close'] - bb['BBL_20_2.0']) / bb_width

    # 10. 出来高
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # 11-12. 時間
    features['hour'] = df.index.hour / 23
    features['day_of_week'] = df.index.dayofweek / 6

    return features[FEATURE_NAMES]
```

---

## 4. ラベル設計

### 4.1 ラベル定義

```python
def create_label(
    df: pd.DataFrame,
    horizon: int = 4,        # 15分足×4 = 1時間
    threshold: float = 0.003  # 0.3%
) -> pd.Series:
    """
    予測ターゲット: 次の1時間後の価格変化

    Returns:
        1: 1時間後に+0.3%以上上昇
        0: それ以外（横ばい or 下落）
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    label = (future_return > threshold).astype(int)
    return label
```

### 4.2 ラベルバランス

想定されるラベル分布（BTC/JPYの場合）:
- **1 (上昇)**: 約35-40%
- **0 (その他)**: 約60-65%

→ 不均衡データのため、`class_weight='balanced'` または SMOTE を検討

---

## 5. 学習パイプライン

### 5.1 概要

```
[ローカルPC] ※VPSでは実行しない

1. データ収集
   └── 過去6ヶ月の15分足データ取得

2. 前処理
   ├── 欠損値処理（forward fill）
   ├── 外れ値クリッピング（3σ）
   └── 特徴量生成

3. 分割（時系列順）
   ├── 訓練: 70%
   ├── 検証: 15%
   └── テスト: 15%

4. 学習
   ├── Walk-Forward Validation（3ヶ月窓）
   ├── Optuna によるハイパラ探索（50試行）
   └── Early Stopping（patience=10）

5. 評価
   ├── Accuracy
   ├── Precision / Recall (クラス1)
   ├── ROC-AUC
   └── バックテスト収益

6. モデル保存
   └── model.pkl (~5MB)

7. VPSにアップロード
```

### 5.2 Walk-Forward Validation

```python
def walk_forward_validation(
    df: pd.DataFrame,
    train_months: int = 3,
    test_months: int = 1
) -> list:
    """
    ウォークフォワード検証

    1. 訓練期間(3ヶ月)でモデル学習
    2. テスト期間(1ヶ月)で評価
    3. ウィンドウをスライドして繰り返し
    """
    results = []
    dates = df.index.to_series()

    # 月単位でウィンドウをスライド
    start_date = dates.min()
    end_date = dates.max()

    current_date = start_date + pd.DateOffset(months=train_months)

    while current_date + pd.DateOffset(months=test_months) <= end_date:
        # 訓練データ
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date
        train_data = df[train_start:train_end]

        # テストデータ
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=test_months)
        test_data = df[test_start:test_end]

        # 学習と評価
        model = train_model(train_data)
        metrics = evaluate_model(model, test_data)

        results.append({
            'train_period': f"{train_start} - {train_end}",
            'test_period': f"{test_start} - {test_end}",
            'metrics': metrics
        })

        # ウィンドウをスライド
        current_date += pd.DateOffset(months=1)

    return results
```

### 5.3 学習コード

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def train_model(df: pd.DataFrame) -> lgb.Booster:
    """
    LightGBMモデルの学習
    """
    # 特徴量とラベル
    X = calculate_features(df)
    y = create_label(df)

    # 欠損値削除
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    # 分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 時系列なのでシャッフルしない
    )

    # データセット作成
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 学習
    model = lgb.train(
        LIGHTGBM_PARAMS,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )

    return model

def save_model(model: lgb.Booster, path: str = 'model.pkl'):
    """モデル保存"""
    joblib.dump(model, path)

def load_model(path: str = 'model.pkl') -> lgb.Booster:
    """モデル読み込み"""
    return joblib.load(path)
```

---

## 6. 推論パイプライン

### 6.1 VPSでの推論

```python
import lightgbm as lgb
import joblib
import pandas as pd
from typing import Tuple

class Predictor:
    """VPS上での推論クラス"""

    def __init__(self, model_path: str = 'model.pkl'):
        self.model = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> Tuple[float, str, float]:
        """
        予測を実行

        Args:
            df: 直近のOHLCVデータ（最低20本以上）

        Returns:
            probability: 上昇確率 (0-1)
            direction: 'long' or 'short' or 'neutral'
            confidence: 確信度 (0-1)
        """
        # 特徴量計算
        features = calculate_features(df)
        latest_features = features.iloc[-1:].values

        # 予測
        probability = self.model.predict(latest_features)[0]

        # 方向と確信度
        confidence = abs(probability - 0.5) * 2

        if probability > 0.5:
            direction = 'long'
        elif probability < 0.5:
            direction = 'short'
        else:
            direction = 'neutral'

        return probability, direction, confidence
```

### 6.2 推論フロー

```
[15分毎cron]
      │
      ▼
┌─────────────┐
│ データ取得  │ ← GMO Public API
│ (直近100本) │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ 特徴量計算  │ ← 12特徴量
│ (最新1本)   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ モデル推論  │ ← LightGBM (~5ms)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ 確信度計算  │ ← |prob - 0.5| × 2
└─────┬───────┘
      │
      ▼
┌─────────────┐     [LONG: 確信度 < 75%]
│ シグナル    │     [SHORT: 確信度 < 80%]
│ フィルター  │ ───────────────────→ ログのみ
└─────┬───────┘
      │ [LONG: 確信度 ≥ 75%]
      │ [SHORT: 確信度 ≥ 80%]
      ▼
┌─────────────┐
│ Telegram   │
│ 通知       │
└─────────────┘
```

---

## 7. 評価指標

### 7.1 モデル評価指標

| 指標 | 目標値 | 説明 |
|------|--------|------|
| Accuracy | > 55% | 全体の正解率 |
| Precision (クラス1) | > 55% | 上昇予測の的中率 |
| Recall (クラス1) | > 50% | 実際の上昇をどれだけ捉えたか |
| ROC-AUC | > 0.55 | 識別能力 |
| Profit Factor | > 1.2 | バックテスト収益性 |

### 7.2 モニタリング指標

```python
class ModelMonitor:
    """モデルパフォーマンス監視"""

    def __init__(self):
        self.predictions = []
        self.actuals = []

    def add_prediction(self, predicted: int, actual: int):
        self.predictions.append(predicted)
        self.actuals.append(actual)

        # 直近100件で評価
        if len(self.predictions) > 100:
            self.predictions.pop(0)
            self.actuals.pop(0)

    def get_rolling_auc(self) -> float:
        """直近100件のAUC"""
        from sklearn.metrics import roc_auc_score
        if len(self.predictions) < 20:
            return 0.5
        return roc_auc_score(self.actuals, self.predictions)

    def should_retrain(self) -> bool:
        """再学習が必要か判定"""
        return self.get_rolling_auc() < 0.52
```

---

## 8. 再学習スケジュール

### 8.1 定期再学習

| 項目 | 設定 |
|------|------|
| 頻度 | 2週間ごと |
| 実行場所 | ローカルPC |
| データ期間 | 過去6ヶ月 |

### 8.2 トリガー条件

| トリガー | 条件 | アクション |
|----------|------|-----------|
| 定期スケジュール | 2週間経過 | 再学習実行 |
| 精度低下 | 直近100予測のAUC < 0.52 | アラート + 再学習推奨 |
| 市場変動 | ボラティリティ急上昇 | 手動で検討 |

### 8.3 再学習フロー

```
[トリガー検知]
      │
      ▼
┌─────────────┐
│ 最新データ  │ ← 過去6ヶ月分
│ 取得       │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ モデル学習  │ ← ローカルPCで実行
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ 評価       │ ← 精度チェック
└─────┬───────┘
      │
      ▼
┌─────────────┐     [精度不足]
│ デプロイ   │ ───────────────→ 手動レビュー
│ 判定       │
└─────┬───────┘
      │ [OK]
      ▼
┌─────────────┐
│ VPSに      │ ← SCP/SFTP
│ アップロード│
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ モデル切替  │ ← ホットスワップ
└─────────────┘
```

---

## 9. 特徴量重要度

バックテスト結果に基づく想定重要度:

| 順位 | 特徴量 | 重要度 | 説明 |
|------|--------|--------|------|
| 1 | rsi_14 | 高 | 過熱感の捕捉に有効 |
| 2 | macd_diff | 高 | モメンタムの変化を捉える |
| 3 | return_1 | 中 | 直近の勢い |
| 4 | ema_ratio | 中 | トレンド方向 |
| 5 | volume_ratio | 中 | 出来高の異常検知 |
| 6 | volatility_20 | 中 | リスク調整 |
| 7 | bb_position | 中 | 価格位置 |
| 8 | atr_14 | 低 | ボラティリティ |
| 9 | return_5 | 低 | 中期リターン |
| 10 | return_15 | 低 | 長期リターン |
| 11 | hour | 低 | 時間帯効果 |
| 12 | day_of_week | 低 | 曜日効果 |

---

## 10. リスクと制限

### 10.1 モデルの限界

| リスク | 説明 | 対策 |
|--------|------|------|
| 過学習 | 訓練データに適合しすぎ | 正則化、交差検証 |
| コンセプトドリフト | 市場構造の変化 | 定期再学習 |
| ブラックスワン | 極端なイベント | 損切りルール |
| データ品質 | 欠損・異常値 | 前処理パイプライン |

### 10.2 運用上の注意

1. **VPSでは推論のみ** - 学習はローカルPCで実行
2. **モデル更新は慎重に** - 新モデルは検証後にデプロイ
3. **確信度フィルター必須** - 低確信度では取引しない
4. **人間承認を維持** - モデルを盲信しない
