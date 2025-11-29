# Bitdaytrader

GMOコインAPIを使用した暗号資産デイトレード完全自動売買システム

## 概要

LightGBMによる価格方向予測を使用した完全自動売買システムです。15分足データから1時間後の価格方向を予測し、自動で売買を行います。

### 主要特徴

- **LightGBM予測**: 15分足データから1時間後の価格方向を予測
- **完全自動売買**: 予測に基づき自動でエントリー・決済
- **Maker注文**: GMOコインのMaker手数料リベート（-0.01%）を活用
- **厳密なリスク管理**: 1トレード2%、日次3%損失制限
- **定期レポート**: 朝昼晩の日次レポート、週次・月次レポートをTelegramに送信
- **ウォークフォワード検証**: 2週間ごとにモデルを再学習・検証

## システム構成

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ GMO Coin    │────▶│ VPS          │────▶│ Telegram    │
│ API         │◀────│ (推論・実行) │     │ (レポート)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ SQLite DB   │
                    └─────────────┘
```

## 必要要件

### VPS（実行環境）
- Python 3.11+
- RAM: 1GB以上
- CPU: 1コア

### ローカルPC（学習環境）
- Python 3.11+
- RAM: 8GB以上推奨
- モデル学習・バックテスト用

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/bitdaytrader.git
cd bitdaytrader
```

### 2. 環境セットアップ

```bash
# venv作成と依存パッケージインストール
make setup

# または手動で
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. 環境変数設定

```bash
cp .env.example .env
# .envファイルを編集
```

必須設定項目：
```bash
# GMO Coin API
GMO_API_KEY=your_api_key
GMO_API_SECRET=your_api_secret

# Telegram Bot（レポート送信用）
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. モデル学習（ローカルPC）

```bash
# 依存パッケージインストール
pip install -e ".[training]"

# 過去データ取得（約14ヶ月分）
make fetch-data

# モデル学習（12ヶ月学習、2ヶ月バックテスト）
make train

# models/lightgbm_model.joblib をVPSにコピー
```

### 5. cron設定（VPS）

```bash
# 15分ごとに実行
crontab -e

# 以下を追加
*/15 * * * * cd /path/to/bitdaytrader && .venv/bin/python -m src.main >> logs/cron.log 2>&1
```

## インストールオプション

```bash
# VPS用（軽量、推論のみ）
pip install -e .

# ローカルPC用（モデル学習）
pip install -e ".[training]"

# 開発用
pip install -e ".[dev]"

# ダッシュボード付き（オプション）
pip install -e ".[dashboard]"
```

## プロジェクト構造

```
Bitdaytrader/
├── src/                  # メインソースコード
│   ├── api/              # GMO Coin APIクライアント
│   ├── models/           # LightGBM予測モデル
│   ├── features/         # 特徴量計算
│   ├── risk/             # リスク管理
│   ├── execution/        # 注文執行
│   ├── telegram/         # Telegram Bot（レポート）
│   ├── reports/          # レポート生成
│   ├── database/         # SQLiteモデル
│   └── core/             # コアエンジン
├── training/             # モデル学習コード（ローカルPC用）
├── config/               # 設定ファイル
├── tests/                # テスト
├── docs/                 # ドキュメント
├── models/               # 学習済みモデル（.joblibファイル）
├── scripts/              # 運用スクリプト
└── data/                 # データ・ログ
```

## 売買フロー

```
1. [15分ごと] GMO APIから15分足データ取得
2. [15分ごと] 12特徴量を計算
3. [15分ごと] LightGBMで方向予測
4. [信頼度≥75%/80%] 自動でMaker指値注文を発注（LONG:75%, SHORT:80%）
5. [約定後] ATRベースの損切り・分割利確を管理
```

## 特徴量（12個）

| 特徴量 | 説明 |
|--------|------|
| return_1 | 1期間リターン |
| return_5 | 5期間リターン |
| return_15 | 15期間リターン |
| volatility_20 | 20期間ボラティリティ |
| atr_14 | ATR（14期間） |
| rsi_14 | RSI（14期間） |
| macd_diff | MACD - シグナル |
| ema_ratio | 短期EMA / 長期EMA |
| bb_position | ボリンジャーバンド位置 |
| volume_ratio | 出来高比率 |
| hour | 時間（0-23） |
| day_of_week | 曜日（0-6） |

## リスク管理

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 1トレードリスク | 2% | 損切り時の最大損失 |
| 日次損失制限 | 3% | この損失で当日取引停止 |
| 最大ポジション | 10% | 総資金に対する上限 |
| 1日最大取引数 | 5回 | これを超えると新規エントリー停止 |

## 利確ルール（分割決済）

| R倍数 | 決済比率 |
|-------|---------|
| 1.5R | 50% |
| 2.5R | 30% |
| 4.0R | 残り20% |

※ 1R = 損切り幅（2×ATR）

## レポート機能

### 日次レポート
- **朝（8:00）**: ステータスレポート
- **昼（12:00）**: ステータスレポート
- **夕方（20:00）**: 日次サマリー

### 週次レポート
- **毎週月曜 8:00**: 週間パフォーマンス

### 月次レポート
- **毎月1日 8:00**: 月間パフォーマンス

## モデル学習スケジュール

| 項目 | 値 |
|------|-----|
| 学習データ | 12ヶ月 |
| バックテストデータ | 2ヶ月 |
| ウォークフォワード間隔 | 2週間 |

## コマンド

```bash
# セットアップ
make setup              # venv作成 + 依存インストール

# 実行（VPS）
make run                # メイン処理実行
make run-paper          # ペーパートレードモード

# モデル学習（ローカルPC）
make fetch-data         # GMOから過去データ取得
make train              # walk-forward学習

# 開発
make test               # テスト実行
make lint               # リンター
make format             # コード整形

# ユーティリティ
make logs               # ログ表示
make cron-setup         # cron設定ヘルプ
```

## ドキュメント

- [アーキテクチャ設計](docs/ARCHITECTURE.md)
- [売買戦略詳細](docs/TRADING_STRATEGY.md)
- [予測モデル設計](docs/PREDICTION_MODEL.md)
- [GMO API リファレンス](docs/GMO_API_REFERENCE.md)
- [リソース最適化](docs/RESOURCE_OPTIMIZATION.md)

## リソース使用量（VPS）

| 構成 | メモリ | 備考 |
|------|--------|------|
| 推論のみ | ~500MB | 通常運用 |
| +ログ保持 | ~700MB | 履歴データ含む |

## Telegramレポート例

### 日次レポート
```
📊 日次レポート
━━━━━━━━━━━━━━

📅 日付: 2024-01-15

📈 取引実績:
  • 取引数: 3回
  • 勝率: 66.7%
  • 勝ち: 2回 / 負け: 1回

💰 損益:
  • 本日損益: +¥15,230
  • 現在資金: ¥1,015,230

━━━━━━━━━━━━━━
```

### 新規ポジション通知
```
📈 新規ポジション

通貨: BTC/JPY
方向: LONG
価格: ¥6,234,500
数量: 0.05 BTC
損切: ¥6,184,500
信頼度: 72.3%
```

## 免責事項

- このシステムは投資助言を目的としていません
- 暗号資産取引にはリスクが伴います
- 損失が発生しても一切の責任を負いません
- 自己責任でご利用ください

## ライセンス

MIT License
