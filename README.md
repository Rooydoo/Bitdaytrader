# Bitdaytrader

GMOコインAPIを使用した暗号資産デイトレード自動売買システム

## 概要

このシステムは、GMOコインの暗号資産取引所APIを利用して、以下の戦略を組み合わせた自動売買を行います：

- **ペアトレード戦略（50%）**: 相関の高い通貨ペア間の価格乖離を利用
- **トレンドフォロー戦略（50%）**: テクニカル指標に基づくトレンド追従
- **機械学習予測**: LightGBM + Ridge + GARCHによる方向予測

## 主要機能

- 自動売買エンジン
- 軽量予測モデル（メモリ効率重視）
- バックテスト・ウォークフォワード分析
- 厳密なリスク管理
- Telegram通知
- 監視ダッシュボード（オプション）

## 必要要件

- Python 3.11+
- RAM: 3GB以上（システム割当）
- CPU: 1コア以上

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
# .envファイルを編集してAPIキーを設定
```

### 4. ボット起動

```bash
source .venv/bin/activate

# ペーパートレード（テスト）
make run-bot-paper

# 本番
make run-bot
```

## インストールオプション

```bash
# 基本（軽量）
pip install -e .

# 開発用
pip install -e ".[dev]"

# ダッシュボード付き
pip install -e ".[dashboard]"

# フル機能（PostgreSQL + Redis）
pip install -e ".[full]"
```

## プロジェクト構造

```
Bitdaytrader/
├── src/                # メインソースコード
│   ├── api/            # GMO Coin APIクライアント
│   ├── strategy/       # 取引戦略
│   ├── models/         # 予測モデル
│   ├── risk/           # リスク管理
│   ├── execution/      # 注文執行
│   ├── backtest/       # バックテスト
│   └── core/           # コアエンジン
├── dashboard/          # 監視ダッシュボード
├── config/             # 設定ファイル
├── tests/              # テスト
├── docs/               # ドキュメント
└── scripts/            # 運用スクリプト
```

## ドキュメント

- [アーキテクチャ設計](docs/ARCHITECTURE.md)
- [プロジェクト構造](docs/PROJECT_STRUCTURE.md)
- [技術スタック](docs/TECH_STACK.md)
- [GMO API リファレンス](docs/GMO_API_REFERENCE.md)
- [売買戦略詳細](docs/TRADING_STRATEGY.md)
- [予測モデル設計](docs/PREDICTION_MODEL.md)
- [リソース最適化](docs/RESOURCE_OPTIMIZATION.md)

## コマンド

```bash
# セットアップ
make setup              # venv作成 + 依存インストール

# 実行
make run-bot            # ボット起動
make run-bot-paper      # ペーパートレード
make run-backtest       # バックテスト
make run-dashboard      # ダッシュボード起動

# モデル
make train-model        # 予測モデル学習
make update-pairs       # ペア相関更新

# 開発
make test               # テスト実行
make lint               # リンター
make format             # コード整形
make clean              # キャッシュ削除
```

## 売買戦略

### ペアトレード
- Z-Score ≥ 2.0 でショートスプレッド
- Z-Score ≤ -2.0 でロングスプレッド
- Z-Score → 0 で利確

### トレンドフォロー
- EMA (9/21/55) の並びでトレンド判定
- RSI, MACD でフィルタリング
- ATRベースの損切り・利確

## リスク管理

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| 1日最大損失 | 2% | 日次損失がこの値に達すると取引停止 |
| 1ポジション最大サイズ | 5% | 総資金に対する1ポジションの上限 |
| 総エクスポージャー | 30% | 全ポジション合計の上限 |

## リソース使用量

| 構成 | メモリ | CPU |
|------|--------|-----|
| 最小（ボットのみ） | ~1.4 GB | 50% |
| 標準（+予測） | ~2.0 GB | 50% |
| フル（+バックテスト） | ~3.0 GB | 80% |

## 免責事項

- このシステムは投資助言を目的としていません
- 暗号資産取引にはリスクが伴います
- 損失が発生しても一切の責任を負いません
- 自己責任でご利用ください

## ライセンス

MIT License
