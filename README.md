# Bitdaytrader

GMOコインAPIを使用した暗号資産デイトレード自動売買システム

## 概要

このシステムは、GMOコインの暗号資産取引所APIを利用して、以下の戦略を組み合わせた自動売買を行います：

- **ペアトレード戦略（50%）**: 相関の高い通貨ペア間の価格乖離を利用
- **トレンドフォロー戦略（50%）**: テクニカル指標に基づくトレンド追従

## 主要機能

- 自動売買エンジン
- リアルタイム監視ダッシュボード
- バックテスト・ウォークフォワード分析
- 厳密なリスク管理
- Telegram/Discord通知

## 必要要件

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Docker & Docker Compose

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/bitdaytrader.git
cd bitdaytrader
```

### 2. 環境設定

```bash
cp .env.example .env
# .envファイルを編集してAPIキーなどを設定
```

### 3. セットアップ

```bash
make setup
```

### 4. ダッシュボード起動

```bash
make run-dashboard
```

## プロジェクト構造

```
Bitdaytrader/
├── src/                # メインソースコード
│   ├── api/            # GMO Coin APIクライアント
│   ├── strategy/       # 取引戦略
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

## コマンド

```bash
# 開発環境セットアップ
make install-dev

# データベース起動
make db-up

# テスト実行
make test

# リンター実行
make lint

# ボット起動（ペーパートレード）
make run-bot-paper

# バックテスト実行
make run-backtest

# ダッシュボード起動
make run-dashboard
```

## リスク管理

このシステムは以下のリスク管理機能を備えています：

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| 1日最大損失 | 2% | 日次損失がこの値に達すると取引停止 |
| 1ポジション最大サイズ | 5% | 総資金に対する1ポジションの上限 |
| 総エクスポージャー | 30% | 全ポジション合計の上限 |

## 免責事項

- このシステムは投資助言を目的としていません
- 暗号資産取引にはリスクが伴います
- 損失が発生しても一切の責任を負いません
- 自己責任でご利用ください

## ライセンス

MIT License
