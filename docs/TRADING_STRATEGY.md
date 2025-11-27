# 売買戦略詳細設計

## 1. 戦略概要

### 1.1 アプローチ

**LightGBMによる方向予測 + 人間承認**

機械学習モデルで価格の上昇/下落を予測し、高確信度のシグナルのみをTelegramで通知。
人間が承認した場合のみ注文を執行する。

### 1.2 ペアトレード不採用の理由

| 理由 | 説明 |
|------|------|
| 保有期間の不一致 | 暗号通貨ペアトレードの平均保有期間は90日程度、デイトレ時間軸と合わない |
| 機会の希少性 | 統計的に有意な乖離発生頻度が低い |
| 執行遅延リスク | 1日の遅延でリターンの大部分が消失 |
| 収益性の低下 | 近年、ペアトレード戦略の収益性は全般的に低下 |

### 1.3 代替: BTC/ETH比率を特徴量として活用

```python
def calculate_btc_eth_features(btc_price: pd.Series, eth_price: pd.Series) -> dict:
    """BTC/ETH関連特徴量（ペアトレードの代わり）"""
    ratio = btc_price / eth_price
    ratio_ma = ratio.rolling(20).mean()
    ratio_zscore = (ratio - ratio_ma) / ratio.rolling(20).std()

    return {
        'btc_eth_ratio': ratio.iloc[-1],
        'btc_eth_ratio_zscore': ratio_zscore.iloc[-1],
        'btc_eth_ratio_above_ma': int(ratio.iloc[-1] > ratio_ma.iloc[-1]),
    }
```

---

## 2. 予測ターゲット

### 2.1 ラベル定義

```python
def create_label(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.003):
    """
    予測ターゲット: 次の4本（1時間）後の価格変化

    Args:
        horizon: 予測期間（15分足の本数）= 4本 = 1時間
        threshold: 上昇判定閾値 = 0.3%

    Returns:
        1: 1時間後に+0.3%以上上昇
        0: それ以外
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    label = (future_return > threshold).astype(int)
    return label
```

### 2.2 ラベル選定理由

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| 予測期間 | 1時間 | デイトレードに適切な時間軸 |
| 閾値 | 0.3% | 取引コスト（~0.1%）を上回る最小利益 |
| 2クラス分類 | 上昇/それ以外 | シンプルで解釈しやすい |

---

## 3. 特徴量設計

### 3.1 特徴量一覧（12個）

| カテゴリ | 特徴量 | 計算方法 |
|---------|--------|----------|
| **リターン系** | return_1 | (close - close[1]) / close[1] |
| | return_5 | (close - close[5]) / close[5] |
| | return_15 | (close - close[15]) / close[15] |
| **ボラティリティ** | volatility_20 | 20期間リターンの標準偏差 |
| | atr_14 | Average True Range (14期間) |
| **モメンタム** | rsi_14 | RSI (14期間) |
| | macd_diff | MACD - Signal Line |
| **トレンド** | ema_ratio | EMA9 / EMA21 |
| | bb_position | (close - BB_lower) / (BB_upper - BB_lower) |
| **出来高** | volume_ratio | volume / SMA(volume, 20) |
| **時間** | hour | 0-23 (UTC) |
| | day_of_week | 0-6 (月-日) |

### 3.2 特徴量計算コード

```python
import pandas as pd
import pandas_ta as ta

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    12個の特徴量を計算

    Args:
        df: OHLCV データフレーム (columns: open, high, low, close, volume)

    Returns:
        特徴量を追加したデータフレーム
    """
    features = pd.DataFrame(index=df.index)

    # リターン系
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_15'] = df['close'].pct_change(15)

    # ボラティリティ
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()
    features['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']

    # モメンタム
    features['rsi_14'] = ta.rsi(df['close'], length=14) / 100  # 0-1に正規化

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    features['macd_diff'] = macd['MACDh_12_26_9'] / df['close']  # 価格で正規化

    # トレンド
    ema9 = ta.ema(df['close'], length=9)
    ema21 = ta.ema(df['close'], length=21)
    features['ema_ratio'] = ema9 / ema21

    bb = ta.bbands(df['close'], length=20, std=2)
    features['bb_position'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])

    # 出来高
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # 時間
    features['hour'] = df.index.hour / 23  # 0-1に正規化
    features['day_of_week'] = df.index.dayofweek / 6  # 0-1に正規化

    return features
```

---

## 4. シグナル生成

### 4.1 確信度計算

```python
def calculate_confidence(probability: float) -> float:
    """
    予測確率から確信度を計算

    確信度 = |probability - 0.5| * 2
    - probability = 0.5 → confidence = 0 (最低)
    - probability = 1.0 → confidence = 1 (最高)
    - probability = 0.0 → confidence = 1 (最高、ショート方向)
    """
    return abs(probability - 0.5) * 2

def generate_signal(probability: float, features: dict) -> dict:
    """
    シグナル生成

    Returns:
        {
            'direction': 'long' or 'short' or 'neutral',
            'probability': float,
            'confidence': float,
            'features': dict,
            'timestamp': datetime,
        }
    """
    confidence = calculate_confidence(probability)

    if probability > 0.5:
        direction = 'long'
    elif probability < 0.5:
        direction = 'short'
    else:
        direction = 'neutral'

    return {
        'direction': direction,
        'probability': probability,
        'confidence': confidence,
        'features': features,
        'timestamp': datetime.utcnow(),
    }
```

### 4.2 シグナルフィルター

```python
class TradeFilter:
    """取引フィルター"""

    def __init__(self):
        self.daily_trade_count = 0
        self.daily_loss = 0.0

    def should_trade(self, signal: dict, market_data: dict) -> tuple[bool, str]:
        """
        取引すべきか判定

        Returns:
            (should_trade: bool, reason: str)
        """
        checks = [
            # 確信度チェック
            (signal['confidence'] >= 0.65,
             f"確信度不足: {signal['confidence']:.1%} < 65%"),

            # スプレッドチェック
            (market_data['spread_bps'] <= 5,
             f"スプレッド過大: {market_data['spread_bps']}bps > 5bps"),

            # 出来高チェック
            (market_data['volume_ratio'] >= 0.8,
             f"出来高不足: {market_data['volume_ratio']:.1%} < 80%"),

            # 日次取引回数制限
            (self.daily_trade_count < 5,
             f"日次取引上限: {self.daily_trade_count} >= 5"),

            # 日次損失制限
            (self.daily_loss < 0.03,
             f"日次損失上限: {self.daily_loss:.1%} >= 3%"),
        ]

        for passed, reason in checks:
            if not passed:
                return False, reason

        return True, "OK"
```

---

## 5. リスク管理

### 5.1 ポジションサイズ

```python
class PositionSizer:
    """ATRベースのポジションサイジング"""

    def __init__(self,
                 total_capital: float,
                 max_risk_per_trade: float = 0.02,    # 1トレードあたり最大リスク2%
                 max_position_ratio: float = 0.10):   # 最大ポジション10%
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_ratio = max_position_ratio

    def calculate(self, entry_price: float, stop_loss_price: float) -> float:
        """
        ポジションサイズ計算

        ロジック:
        1. リスク額 = 資金 × リスク率(2%)
        2. 価格リスク = |エントリー - ストップロス|
        3. ポジションサイズ = リスク額 / 価格リスク
        4. 最大ポジション制限でキャップ
        """
        risk_amount = self.total_capital * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)

        # リスクベースのポジションサイズ
        position_size = risk_amount / price_risk

        # 最大ポジション制限
        max_position = self.total_capital * self.max_position_ratio
        position_value = position_size * entry_price

        if position_value > max_position:
            position_size = max_position / entry_price

        return position_size
```

### 5.2 損切りルール

| ルール | 条件 | アクション |
|--------|------|-----------|
| **ATRストップ** | 価格 < エントリー - 2×ATR(14) | 即時決済 |
| **時間ストップ** | ポジション保有 > 4時間 | 強制決済 |
| **日次損失上限** | 日次損失 > 資本の3% | 当日取引停止 |
| **週次損失上限** | 週次損失 > 資本の7% | 当週取引停止 |
| **月次損失上限** | 月次損失 > 資本の15% | 当月取引停止（要レビュー）|

### 5.3 利確ルール（分割利確）

```python
class TakeProfitManager:
    """分割利確マネージャー"""

    def __init__(self):
        # (R倍率, 決済比率)
        self.tp_levels = [
            (1.5, 0.50),   # 1.5R到達で50%決済
            (2.5, 0.30),   # 2.5R到達で30%決済
            (4.0, 0.20),   # 4.0R到達で残り全決済
        ]

    def check_tp(self,
                 current_price: float,
                 entry_price: float,
                 stop_loss_price: float,
                 direction: str) -> tuple[bool, float, str]:
        """
        利確チェック

        Returns:
            (should_close: bool, close_ratio: float, level: str)
        """
        risk_unit = abs(entry_price - stop_loss_price)

        if direction == 'long':
            current_profit_r = (current_price - entry_price) / risk_unit
        else:
            current_profit_r = (entry_price - current_price) / risk_unit

        for r_target, close_ratio in self.tp_levels:
            if current_profit_r >= r_target:
                return True, close_ratio, f"TP{r_target}R"

        return False, 0.0, ""
```

---

## 6. 注文執行

### 6.1 Maker注文優先

GMOコインの手数料体系を活用：

| 注文タイプ | Maker | Taker |
|-----------|-------|-------|
| 現物 (BTC, ETH) | **-0.01%** (リベート) | 0.05% |
| レバレッジ | **無料** | **無料** |

```python
def calculate_limit_price(
    current_price: float,
    direction: str,
    spread_adjustment: float = 0.0001  # 0.01%
) -> float:
    """
    Maker注文用の指値価格を計算

    - Long: 現在価格より少し下（Ask寄り）
    - Short: 現在価格より少し上（Bid寄り）
    """
    if direction == 'long':
        # 少し下で指値（買い板に置く）
        return current_price * (1 - spread_adjustment)
    else:
        # 少し上で指値（売り板に置く）
        return current_price * (1 + spread_adjustment)
```

### 6.2 注文フロー

```
[シグナル承認受信]
        │
        ▼
┌───────────────┐
│ 最新価格取得   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ リスク再確認   │ ← 承認後の状況変化をチェック
│ - 日次損失    │
│ - ポジション数 │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ ポジションサイズ│ ← ATRベース計算
│ 計算          │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 指値価格計算   │ ← Maker注文でリベート獲得
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ 注文発行      │ ← GMO Private API
│ (LIMIT)      │
└───────┬───────┘
        │
        ▼
┌───────────────┐     [未約定30秒以上]
│ 約定待ち      │ ─────────────────→ 価格更新して再発行
│ (30秒)       │                    (最大3回)
└───────┬───────┘
        │ [約定]
        ▼
┌───────────────┐
│ ポジション登録 │ ← SQLite
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ ストップロス   │ ← 2×ATR
│ 注文設定      │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Telegram通知  │ ← 約定確認
└───────────────┘
```

---

## 7. 人間承認フロー

### 7.1 Telegram通知フォーマット

```
🔔 **新規シグナル検出**

📊 **BTC/JPY**
├ 方向: 🟢 LONG
├ 確信度: 72.5%
├ 現在価格: ¥14,250,000
├ 推奨エントリー: ¥14,248,575
├ ストップロス: ¥14,107,000 (-1.0%)
├ 利確目標1: ¥14,463,787 (1.5R)
├ 利確目標2: ¥14,606,281 (2.5R)
└ リスク額: ¥20,000 (2%)

📈 **主要指標**
├ RSI(14): 58.2
├ MACD: +0.12%
├ EMA比: 1.003
└ 出来高比: 1.25

⏰ 有効期限: 15分

/approve_abc123 - 承認
/reject_abc123 - 却下
```

### 7.2 承認フロー状態遷移

```
[シグナル生成]
      │
      ▼
   PENDING ──────[15分経過]──────→ TIMEOUT
      │                              │
      │                              ▼
      │                          (ログ記録)
      │
   [/approve]                    [/reject]
      │                              │
      ▼                              ▼
   APPROVED                      REJECTED
      │                              │
      ▼                              ▼
  (注文執行)                    (ログ記録)
      │
      ▼
   EXECUTED
```

---

## 8. 期待リターン分析

### 8.1 保守的見積もり

**前提条件**:
- 初期資本: 100万円
- 勝率: 55%
- 平均利益/損失比 (R): 1.5:1
- 1トレードあたりリスク: 2%
- 月間取引数: 30-40回

```
期待値 (1トレードあたり):
E = (勝率 × 平均利益) - (敗率 × 平均損失)
E = (0.55 × 3%) - (0.45 × 2%)
E = 1.65% - 0.90%
E = 0.75% / トレード

月間期待リターン:
月間E = 0.75% × 35トレード = 26.25%
※ただし複利効果と取引コストを考慮すると実質15-20%程度
```

### 8.2 シナリオ分析

| シナリオ | 勝率 | R比 | 月間トレード | 月間リターン |
|---------|------|-----|-------------|-------------|
| 楽観的 | 58% | 1.8:1 | 40 | +25% |
| 標準 | 55% | 1.5:1 | 35 | +15% |
| 保守的 | 52% | 1.3:1 | 30 | +5% |
| 悲観的 | 48% | 1.2:1 | 25 | -3% |

### 8.3 目標リスク指標

| 指標 | 目標値 | 許容下限 |
|------|--------|---------|
| Sharpe Ratio (年率) | > 2.0 | > 1.0 |
| Maximum Drawdown | < 15% | < 25% |
| Win Rate | > 55% | > 50% |
| Profit Factor | > 1.5 | > 1.2 |

---

## 9. パラメータ一覧

### 9.1 モデルパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 予測期間 | 4本 (1時間) | 15分足での予測ホライズン |
| 上昇閾値 | 0.3% | ラベル1の基準 |
| 確信度閾値 | 65% | 通知の最低確信度 |

### 9.2 リスクパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 1トレードリスク | 2% | 資金に対する最大損失 |
| 最大ポジション | 10% | 資金に対する最大ポジション |
| ストップロス | 2×ATR | ATRベースの損切り |
| 時間ストップ | 4時間 | 最大保有時間 |
| 日次損失上限 | 3% | 当日取引停止ライン |
| 週次損失上限 | 7% | 当週取引停止ライン |
| 月次損失上限 | 15% | 当月取引停止ライン |
| 日次取引上限 | 5回 | 1日の最大取引回数 |

### 9.3 利確パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| TP1 | 1.5R, 50% | 1.5倍リスクで半分決済 |
| TP2 | 2.5R, 30% | 2.5倍リスクで30%決済 |
| TP3 | 4.0R, 20% | 4倍リスクで残り決済 |

### 9.4 フィルターパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 最小確信度 | 65% | シグナル通知の閾値 |
| 最大スプレッド | 5bps | 取引可能スプレッド上限 |
| 最小出来高比 | 80% | 20MA比での最小出来高 |
| シグナル有効期限 | 15分 | 承認待ちタイムアウト |
