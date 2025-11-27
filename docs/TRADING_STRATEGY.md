# 売買戦略詳細設計

## 1. 戦略概要

資金を2つの戦略に分割して運用：

| 戦略 | 配分 | 特徴 |
|------|------|------|
| ペアトレード | 50% | 市場中立、低リスク |
| トレンドフォロー | 50% | 方向性、中リスク |

---

## 2. ペアトレード戦略

### 2.1 コンセプト

相関の高い2つの暗号資産間の価格乖離を利用。
価格が一時的に乖離しても、長期的には均衡に戻る性質（平均回帰）を活用。

### 2.2 ペア選定条件

```python
PAIR_SELECTION_CRITERIA = {
    # 相関条件
    "min_correlation": 0.7,          # 最小相関係数
    "correlation_window": 30,         # 相関計算期間（日）

    # コインテグレーション条件
    "cointegration_pvalue": 0.05,    # ADF検定のp値閾値

    # 流動性条件
    "min_daily_volume_jpy": 100_000_000,  # 最小日次出来高（1億円）

    # スプレッド条件
    "max_spread_pct": 0.5,           # 最大スプレッド（%）
}
```

### 2.3 候補ペア

| ペア | 理由 |
|------|------|
| BTC / ETH | 高流動性、高相関 |
| ETH / SOL | スマコン系、相関あり |
| XRP / XLM | 決済系、高相関 |
| LINK / DOT | インフラ系 |

### 2.4 エントリー条件

```python
# スプレッド計算
spread = log(price_A) - beta * log(price_B) - alpha

# Z-Score計算
z_score = (spread - spread_mean) / spread_std

# エントリー条件
ENTRY_CONDITIONS = {
    # ロングスプレッド（Aロング、Bショート）
    "long_spread": {
        "z_score": -2.0,             # Z-Score が -2.0 以下
        "confirmation_bars": 2,       # 2本連続で条件満たす
    },

    # ショートスプレッド（Aショート、Bロング）
    "short_spread": {
        "z_score": 2.0,              # Z-Score が 2.0 以上
        "confirmation_bars": 2,
    },
}
```

### 2.5 エグジット条件

```python
EXIT_CONDITIONS = {
    # 利確
    "take_profit": {
        "z_score_return": 0.0,       # Z-Score が0に回帰
    },

    # 損切り
    "stop_loss": {
        "z_score_extreme": 3.5,      # Z-Score が極端に拡大
        "max_holding_hours": 48,     # 最大保有時間
        "max_loss_pct": 2.0,         # 最大損失（%）
    },

    # 緊急エグジット
    "emergency": {
        "correlation_breakdown": 0.3, # 相関崩壊
    },
}
```

### 2.6 ポジションサイジング

```python
def calculate_pair_position_size(
    capital: float,
    z_score: float,
    volatility: float
) -> float:
    """
    ペアトレードのポジションサイズ計算

    - 基本サイズ: 資金の5%
    - Z-Score調整: 乖離が大きいほど増加（最大2倍）
    - ボラティリティ調整: 高ボラ時は減少
    """
    base_size = capital * 0.05

    # Z-Score調整（1.5〜2.0倍）
    z_multiplier = min(abs(z_score) / 2.0, 2.0)

    # ボラティリティ調整
    vol_adjustment = 1.0 / (1.0 + volatility)

    return base_size * z_multiplier * vol_adjustment
```

---

## 3. トレンドフォロー戦略

### 3.1 コンセプト

明確なトレンドに追従して利益を獲得。
複数の指標を組み合わせてフィルタリング。

### 3.2 使用指標

| 指標 | パラメータ | 用途 |
|------|-----------|------|
| EMA | 9, 21, 55 | トレンド方向 |
| RSI | 14 | 過熱感 |
| MACD | 12, 26, 9 | モメンタム |
| ATR | 14 | ボラティリティ |
| Volume MA | 20 | 出来高確認 |

### 3.3 エントリー条件（ロング）

```python
LONG_ENTRY_CONDITIONS = {
    # 必須条件（すべて満たす）
    "required": {
        # EMAの並び
        "ema_alignment": "EMA9 > EMA21 > EMA55",

        # 価格位置
        "price_above_ema21": True,

        # RSI範囲
        "rsi_range": (35, 65),        # 過熱していない

        # MACD
        "macd_positive": True,         # MACDライン > 0
        "macd_above_signal": True,     # MACDライン > シグナル
    },

    # トリガー条件（いずれか1つ）
    "trigger": [
        # EMA9がEMA21を上抜け
        {"type": "ema_cross", "fast": 9, "slow": 21, "direction": "up"},

        # 価格がEMA21にタッチして反発
        {"type": "ema_bounce", "ema": 21, "direction": "up"},

        # ブレイクアウト
        {"type": "breakout", "period": 20, "direction": "up"},
    ],

    # フィルター条件
    "filter": {
        # 出来高確認
        "volume_above_ma": 1.2,        # 20MA比120%以上

        # ATRフィルター
        "atr_min_pct": 1.0,            # 最小ボラティリティ
        "atr_max_pct": 8.0,            # 最大ボラティリティ
    },
}
```

### 3.4 エントリー条件（ショート）

```python
SHORT_ENTRY_CONDITIONS = {
    "required": {
        "ema_alignment": "EMA9 < EMA21 < EMA55",
        "price_below_ema21": True,
        "rsi_range": (35, 65),
        "macd_negative": True,
        "macd_below_signal": True,
    },

    "trigger": [
        {"type": "ema_cross", "fast": 9, "slow": 21, "direction": "down"},
        {"type": "ema_bounce", "ema": 21, "direction": "down"},
        {"type": "breakout", "period": 20, "direction": "down"},
    ],

    "filter": {
        "volume_above_ma": 1.2,
        "atr_min_pct": 1.0,
        "atr_max_pct": 8.0,
    },
}
```

### 3.5 エグジット条件

```python
EXIT_CONDITIONS = {
    # 利確
    "take_profit": {
        # ATRベース（エントリー価格 + 3 * ATR）
        "atr_multiple": 3.0,

        # または固定比率
        "fixed_pct": 4.0,
    },

    # 損切り
    "stop_loss": {
        # ATRベース（エントリー価格 - 1.5 * ATR）
        "atr_multiple": 1.5,

        # または固定比率
        "fixed_pct": 2.0,
    },

    # トレーリングストップ
    "trailing_stop": {
        "activation_pct": 2.0,         # 2%利益で発動
        "trail_pct": 1.5,              # 1.5%で追従
    },

    # トレンド反転エグジット
    "trend_reversal": {
        "ema_cross_against": True,     # EMA9がEMA21を逆方向クロス
        "rsi_extreme": (20, 80),       # RSI極端値
    },

    # 時間ベース
    "time_based": {
        "max_holding_hours": 24,       # デイトレなので24時間以内
    },
}
```

### 3.6 ポジションサイジング

```python
def calculate_trend_position_size(
    capital: float,
    atr: float,
    price: float,
    risk_per_trade: float = 0.01  # 1トレードあたりリスク1%
) -> float:
    """
    ATRベースのポジションサイズ計算

    - リスク額 = 資金 × リスク率
    - ストップ幅 = ATR × 1.5
    - ポジションサイズ = リスク額 / ストップ幅
    """
    risk_amount = capital * risk_per_trade
    stop_distance = atr * 1.5
    position_size = risk_amount / stop_distance

    # 最大サイズ制限（資金の5%）
    max_size = (capital * 0.05) / price

    return min(position_size, max_size)
```

---

## 4. シグナル統合

### 4.1 シグナル優先度

```python
SIGNAL_PRIORITY = {
    # 緊急シグナル（即時実行）
    "emergency_exit": 1,      # 損切り、緊急決済

    # 高優先度
    "stop_loss": 2,           # 通常損切り
    "take_profit": 3,         # 利確

    # 通常優先度
    "entry": 4,               # 新規エントリー
    "trail_update": 5,        # トレーリング更新
}
```

### 4.2 シグナル競合解決

```python
def resolve_signal_conflict(signals: list) -> Signal:
    """
    複数シグナルの競合を解決

    1. 優先度でソート
    2. リスク管理チェック
    3. 最終シグナル決定
    """
    # エグジット優先（損切りは必ず実行）
    exit_signals = [s for s in signals if s.is_exit]
    if exit_signals:
        return max(exit_signals, key=lambda s: s.priority)

    # エントリーシグナルの評価
    entry_signals = [s for s in signals if s.is_entry]
    if not entry_signals:
        return None

    # 両戦略からシグナルがある場合は強い方を選択
    return max(entry_signals, key=lambda s: s.confidence)
```

---

## 5. フローチャート

### 5.1 ペアトレードフロー

```
┌─────────────────┐
│   価格データ取得  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  スプレッド計算   │
│  Z-Score算出    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     No
│ Z-Score > 2.0   │────────────┐
│ or < -2.0 ?     │            │
└────────┬────────┘            │
         │ Yes                 │
         ▼                     │
┌─────────────────┐            │
│  確認バー待ち    │            │
│  (2本連続)      │            │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│ リスクチェック   │            │
│ - ポジション上限  │            │
│ - 日次損失上限   │            │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│   注文発行      │            │
│ - ペアA: ロング  │            │
│ - ペアB: ショート │            │
└────────┬────────┘            │
         │                     │
         ▼                     ▼
┌─────────────────┐     ┌──────────┐
│  ポジション監視   │     │  待機    │
│ - Z-Score回帰   │     └──────────┘
│ - 損切りチェック  │
└─────────────────┘
```

### 5.2 トレンドフォローフロー

```
┌─────────────────┐
│   価格データ取得  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  指標計算        │
│ - EMA, RSI      │
│ - MACD, ATR     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     No
│ 必須条件チェック  │────────────┐
│ (EMA並び等)     │            │
└────────┬────────┘            │
         │ Yes                 │
         ▼                     │
┌─────────────────┐     No     │
│ トリガー条件     │────────────┤
│ (クロス等)      │            │
└────────┬────────┘            │
         │ Yes                 │
         ▼                     │
┌─────────────────┐     No     │
│ フィルター条件   │────────────┤
│ (出来高等)      │            │
└────────┬────────┘            │
         │ Yes                 │
         ▼                     │
┌─────────────────┐            │
│  予測モデル確認   │            │
│ (方向性一致?)    │            │
└────────┬────────┘            │
         │                     │
         ▼                     │
┌─────────────────┐            │
│ リスクチェック   │            │
└────────┬────────┘            │
         │                     │
         ▼                     ▼
┌─────────────────┐     ┌──────────┐
│   注文発行      │     │  待機    │
└─────────────────┘     └──────────┘
```

---

## 6. パラメータ一覧

### 6.1 ペアトレード

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| Z-Score エントリー | ±2.0 | エントリー閾値 |
| Z-Score エグジット | 0.0 | 利確閾値 |
| Z-Score 損切り | ±3.5 | 損切り閾値 |
| 確認バー数 | 2 | 連続確認本数 |
| 相関最小値 | 0.7 | ペア選定基準 |
| 最大保有時間 | 48時間 | 時間ベース決済 |

### 6.2 トレンドフォロー

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| EMA期間 | 9, 21, 55 | トレンド判定 |
| RSI期間 | 14 | 過熱感判定 |
| RSI範囲 | 35-65 | エントリー許可範囲 |
| ATR期間 | 14 | ボラティリティ |
| 利確 (ATR倍率) | 3.0 | ATRの3倍 |
| 損切り (ATR倍率) | 1.5 | ATRの1.5倍 |
| トレーリング発動 | 2.0% | 利益2%で発動 |
| 最大保有時間 | 24時間 | デイトレ制限 |
