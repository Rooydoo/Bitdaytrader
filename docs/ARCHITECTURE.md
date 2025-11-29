# CryptoTrader V2 - アーキテクチャ設計書

## 1. システム概要

### 1.1 目的
低リソースVPS環境（2コア/8GB RAM、実効1コア/4GB）で運用可能な、機械学習ベースの仮想通貨デイトレードシステムを構築する。

### 1.2 設計思想
- **軽量性**: メモリ/CPU使用量の最小化
- **堅牢性**: リスク管理の厳格化による資本保全
- **透明性**: 人間による最終判断を維持（Telegram承認）

### 1.3 対象市場
- **主要ペア**: BTC/JPY, ETH/JPY
- **取引所**: GMOコイン（API経由）
- **取引時間**: 24時間365日（自動監視、人間判断による執行）
- **取引種別**: 取引所（現物取引）または レバレッジ取引

---

## 2. システムアーキテクチャ

### 2.1 ハードウェア要件

| リソース | 総量 | 実効割当 | 用途 |
|---------|------|----------|------|
| CPU | 2コア | 1コア | 推論・データ処理 |
| RAM | 8GB | 4GB | モデル+データバッファ |
| ストレージ | - | 10GB | ログ・モデル・履歴データ |
| ネットワーク | - | 安定接続 | API通信 |

### 2.2 ソフトウェアスタック

```
┌─────────────────────────────────────────────────────┐
│                    VPS (Ubuntu 22.04)               │
├─────────────────────────────────────────────────────┤
│  Python 3.11                                        │
│  ├── LightGBM 4.x        (推論エンジン)            │
│  ├── pandas 2.x          (データ処理)              │
│  ├── pandas-ta           (テクニカル指標)          │
│  ├── httpx               (GMOコインAPI)            │
│  ├── websockets          (リアルタイムデータ)      │
│  └── asyncio             (非同期処理)              │
├─────────────────────────────────────────────────────┤
│  SQLite                   (ローカルDB)              │
│  Telegram Bot API         (通知・承認)              │
│  cron                     (15分毎スケジュール)      │
└─────────────────────────────────────────────────────┘
```

### 2.3 処理フロー

```
[15分毎のcron]
      │
      ▼
┌─────────────┐
│ データ取得  │ ← GMOコイン Public API (Ticker + 板情報)
└─────┬───────┘
      ▼
┌─────────────┐
│ 特徴量計算  │ ← 12個のテクニカル指標
└─────┬───────┘
      ▼
┌─────────────┐
│ LightGBM   │ ← 学習済みモデル (.pkl, ~5MB)
│ 推論       │
└─────┬───────┘
      ▼
┌─────────────┐
│ シグナル   │ ← 確信度スコア算出
│ 生成       │
└─────┬───────┘
      ▼
┌─────────────┐     [LONG: 確信度 < 75%]
│ 閾値判定   │     [SHORT: 確信度 < 80%]
│            │ ──────────────────→ ログのみ、通知なし
└─────┬───────┘
      │ [LONG: 確信度 ≥ 75%]
      │ [SHORT: 確信度 ≥ 80%]
      ▼
┌─────────────┐
│ 取引       │     [フィルター不通過]
│ フィルター  │ ──────────────────→ ログのみ、通知なし
└─────┬───────┘
      │ [通過]
      ▼
┌─────────────┐
│ Telegram   │ ← シグナル詳細を送信
│ 通知       │
└─────┬───────┘
      ▼
┌─────────────┐
│ 人間承認   │ ← Telegramで承認/却下 (15分有効)
└─────┬───────┘
      │ [承認]
      ▼
┌─────────────┐
│ 注文執行   │ ← GMOコイン Private API (Maker注文優先)
└─────────────┘
```

---

## 3. コンポーネント詳細

### 3.1 GMO Coin API Connector

| 機能 | API種別 | 用途 |
|------|---------|------|
| ティッカー取得 | Public REST | 現在価格 |
| 板情報取得 | Public REST | スプレッド確認 |
| 約定履歴取得 | Public REST | 出来高確認 |
| 残高照会 | Private REST | 資産状況確認 |
| 注文発行 | Private REST | 売買注文 |
| 注文照会 | Private REST | 注文状況確認 |
| ポジション照会 | Private REST | 建玉確認 |
| リアルタイム価格 | WebSocket | ティックデータ（オプション）|

**API呼び出し制限**:
- 取引高に応じたTier制（週次更新）
- WebSocket: 1秒間1リクエストまで

### 3.2 LightGBM 予測エンジン

```python
# モデル仕様
MODEL_SPEC = {
    "type": "LightGBM",
    "objective": "binary",           # 上昇/下落の2クラス分類
    "target": "1時間後に+0.3%以上上昇するか",
    "features": 12,                  # 特徴量数
    "model_size": "~5MB",
    "inference_time": "<10ms",
}

# 予測出力
PREDICTION_OUTPUT = {
    "probability": 0.0-1.0,          # 上昇確率
    "direction": "long" | "short",   # 推奨方向
    "confidence": 0.0-1.0,           # 確信度
}
```

### 3.3 Risk Manager

```python
RISK_PARAMS = {
    # ポジションサイズ
    "max_risk_per_trade": 0.02,      # 1トレードあたり最大リスク2%
    "max_position_ratio": 0.10,      # 最大ポジション10%

    # 損失制限
    "max_daily_loss_pct": 0.03,      # 1日最大損失3%
    "max_weekly_loss_pct": 0.07,     # 週間最大損失7%
    "max_monthly_loss_pct": 0.15,    # 月間最大損失15%

    # 個別ポジション
    "stop_loss_atr_multiple": 2.0,   # ストップロス = 2×ATR
    "time_stop_hours": 4,            # 最大保有時間4時間

    # 取引制限（税金効率化のため削減）
    "max_daily_trades": 3,           # 1日最大3トレード
}
```

### 3.4 取引フィルター

```python
TRADE_FILTER = {
    # 確信度閾値（方向別）
    "long_min_confidence": 0.75,     # LONG: 確信度75%以上
    "short_min_confidence": 0.80,    # SHORT: 確信度80%以上
    "confidence_rank": 0.10,         # 上位10%の確信度
    "max_spread_bps": 5,             # スプレッド5bps以下
    "min_volume_ratio": 0.8,         # 出来高が平均の80%以上
    "avoid_news_time": True,         # 重要経済指標発表前後30分は除外
}
```

### 3.5 Telegram Bot

```
機能:
1. シグナル通知
   - 方向、確信度、推奨エントリー価格
   - ストップロス、利確目標
   - リスク額

2. 承認フロー
   - /approve_{id} - 承認
   - /reject_{id} - 却下
   - /modify_{id} - 修正
   - 15分タイムアウト

3. 日次レポート
   - 損益サマリー
   - 統計（勝率、PF、DD）
   - リスク状態

4. アラート
   - 損失上限到達
   - システムエラー
   - 異常価格変動
```

### 3.6 Execution Engine

```
注文フロー:
1. 人間承認受信
2. リスクチェック（最新状態で再確認）
3. ポジションサイズ計算（ATRベース）
4. Maker注文で指値発行（リベート獲得）
5. 約定確認 → ポジション登録
6. ストップロス・利確注文設定
7. Telegram通知
```

---

## 4. データフロー

```
[15分毎cron]
      │
      ▼
[GMO Public API] ─────────────────────────────────────┐
      │                                               │
      ▼                                               │
┌─────────────┐     ┌─────────────┐                  │
│ Ticker      │     │ Orderbook   │                  │
│ (価格)      │     │ (板情報)     │                  │
└─────┬───────┘     └─────┬───────┘                  │
      │                   │                          │
      └───────┬───────────┘                          │
              │                                       │
              ▼                                       │
       ┌─────────────┐                               │
       │ 特徴量計算  │ ← 12個のテクニカル指標        │
       └─────┬───────┘                               │
              │                                       │
              ▼                                       │
       ┌─────────────┐                               │
       │ LightGBM   │ ← model.pkl (~5MB)            │
       │ 推論       │                                │
       └─────┬───────┘                               │
              │                                       │
              ▼                                       │
       ┌─────────────┐                               │
       │ SQLite     │ ← signals, trades, summary    │
       └─────┬───────┘                               │
              │                                       │
              ▼                                       │
       ┌─────────────┐     ┌─────────────┐          │
       │ Telegram   │ ←→  │ 人間        │          │
       │ Bot        │     │ (承認/却下)  │          │
       └─────┬───────┘     └─────────────┘          │
              │ [承認]                                │
              ▼                                       │
       ┌─────────────┐                               │
       │ GMO        │ ← Private API                 │
       │ 注文執行   │                                │
       └─────────────┘                               │
```

---

## 5. データベース設計

### 5.1 SQLite スキーマ

```sql
-- 取引履歴
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,           -- 'long' or 'short'
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL NOT NULL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    position_size REAL NOT NULL,
    confidence REAL NOT NULL,
    entry_time DATETIME NOT NULL,
    exit_time DATETIME,
    exit_reason TEXT,                  -- 'tp1', 'tp2', 'stop_loss', 'time_stop', 'manual'
    pnl_amount REAL,
    pnl_percent REAL,
    status TEXT DEFAULT 'pending',     -- 'pending', 'open', 'closed', 'cancelled'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- シグナル履歴
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    features JSON,                     -- 予測時の特徴量
    prediction_time DATETIME NOT NULL,
    approved BOOLEAN,
    approved_by TEXT,                  -- 'human', 'timeout'
    trade_id INTEGER REFERENCES trades(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 日次サマリー
CREATE TABLE daily_summary (
    date DATE PRIMARY KEY,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    gross_profit REAL,
    gross_loss REAL,
    net_pnl REAL,
    max_drawdown REAL,
    sharpe_ratio REAL
);

-- モデルパフォーマンス追跡
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    model_version TEXT NOT NULL,
    predictions INTEGER,
    correct_predictions INTEGER,
    accuracy REAL,
    auc_roc REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. 計算資源設計

### 6.1 メモリ使用量

| コンポーネント | 推定メモリ |
|---------------|-----------|
| Python ランタイム | ~100 MB |
| LightGBM モデル | ~50 MB |
| pandas データバッファ | ~200 MB |
| SQLite | ~50 MB |
| その他 | ~100 MB |
| **合計** | **~500 MB** |

→ 4GB割当で十分な余裕

### 6.2 CPU使用量

| 処理 | 頻度 | CPU時間 |
|------|------|---------|
| データ取得 | 15分毎 | ~1秒 |
| 特徴量計算 | 15分毎 | ~2秒 |
| LightGBM推論 | 15分毎 | ~10ms |
| Telegram処理 | 随時 | ~100ms |

→ 1コアで十分（使用率 <5%）

### 6.3 ストレージ使用量

| 項目 | サイズ |
|------|--------|
| モデルファイル | ~5 MB |
| SQLiteデータベース | ~100 MB/年 |
| ログファイル | ~500 MB/年 |
| 履歴データ | ~200 MB |

→ 10GBで十分

---

## 7. セキュリティ設計

### 7.1 API認証情報
```
- 環境変数で管理（.env）
- .gitignore に追加
- GMO API: APIキー + シークレット
- Telegram: Botトークン + Chat ID
```

### 7.2 ネットワーク
```
- VPS のファイアウォール設定
- SSH キー認証のみ
- API通信はHTTPS
- WebSocketはWSS
```

### 7.3 監査ログ
```
- 全シグナルのログ記録
- 全注文のログ記録
- 承認/却下のログ記録
- エラー・例外のログ記録
- ログローテーション（7日保持）
```

---

## 8. 監視・アラート

### 8.1 システムヘルスチェック

| チェック項目 | 条件 | 重大度 |
|-------------|------|--------|
| API接続 | 3回連続エラー | Critical |
| メモリ使用率 | > 90% | Critical |
| メモリ使用率 | > 80% | Warning |
| モデル精度 | AUC < 0.52 | Warning |
| 日次損失 | > 3% | Critical |
| 価格変動 | > 5%/15分 | Warning |

### 8.2 アラートアクション

| 重大度 | アクション |
|--------|-----------|
| Critical | 即時Telegram通知、取引停止 |
| Warning | Telegram通知のみ |

---

## 9. 障害対策

### 9.1 自動復旧
```
- systemd によるプロセス監視・再起動
- API障害時のリトライ（指数バックオフ）
- WebSocket再接続ロジック
```

### 9.2 緊急停止
```
- Telegram /stop コマンド
- 自動停止条件:
  - 日次損失上限到達
  - API連続エラー（3回）
  - メモリ逼迫（90%超）
```

### 9.3 バックアップ
```
- SQLite日次バックアップ
- モデルファイルのバージョン管理
- ポジション状態の永続化
```

---

## 10. 開発・デプロイメントフェーズ

### Phase 1: 基盤構築 (2週間)
- [ ] プロジェクト構造作成
- [ ] GMOコインAPIクライアント実装
- [ ] データ取得モジュール実装
- [ ] 特徴量計算モジュール実装
- [ ] SQLiteデータベース構築

### Phase 2: モデル開発 (2週間)
- [ ] 過去データ収集
- [ ] 特徴量エンジニアリング検証
- [ ] LightGBMモデル学習
- [ ] バックテスト実装
- [ ] ハイパーパラメータ最適化

### Phase 3: 取引システム (2週間)
- [ ] ポジション管理モジュール
- [ ] リスク管理モジュール
- [ ] 注文執行モジュール
- [ ] 統合テスト

### Phase 4: 通知・承認 (1週間)
- [ ] Telegram Bot実装
- [ ] 承認フロー実装
- [ ] 日次レポート生成
- [ ] アラートシステム

### Phase 5: デプロイ・検証 (2週間)
- [ ] VPSセットアップ
- [ ] ペーパートレード運用 (1週間)
- [ ] パフォーマンス検証
- [ ] 本番移行

### Phase 6: 本番運用開始
- [ ] 少額資本での運用開始
- [ ] 週次レビュー
- [ ] 継続的改善
