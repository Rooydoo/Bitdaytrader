# プロジェクト構造

```
Bitdaytrader/
├── README.md                       # プロジェクト概要
├── pyproject.toml                  # Python プロジェクト設定
├── requirements.txt                # 依存パッケージ
├── requirements-dev.txt            # 開発用依存パッケージ
├── .env.example                    # 環境変数テンプレート
├── .gitignore                      # Git除外設定
├── docker-compose.yml              # Docker構成（PostgreSQL, Redis）
├── Makefile                        # よく使うコマンド集
│
├── config/                         # 設定ファイル
│   ├── __init__.py
│   ├── settings.py                 # 基本設定
│   ├── trading_params.py           # 取引パラメータ
│   └── logging_config.py           # ログ設定
│
├── src/                            # メインソースコード
│   ├── __init__.py
│   │
│   ├── api/                        # GMO Coin API クライアント
│   │   ├── __init__.py
│   │   ├── client.py               # APIクライアント基底クラス
│   │   ├── public.py               # Public API (価格取得等)
│   │   ├── private.py              # Private API (注文・残高)
│   │   ├── websocket.py            # WebSocket クライアント
│   │   ├── models.py               # APIレスポンスモデル
│   │   └── exceptions.py           # API例外クラス
│   │
│   ├── data/                       # データ管理
│   │   ├── __init__.py
│   │   ├── fetcher.py              # データ取得
│   │   ├── storage.py              # データ保存（PostgreSQL）
│   │   ├── cache.py                # キャッシュ（Redis）
│   │   └── models.py               # データモデル（SQLAlchemy）
│   │
│   ├── strategy/                   # 取引戦略
│   │   ├── __init__.py
│   │   ├── base.py                 # 戦略基底クラス
│   │   ├── pair_trading/           # ペアトレード戦略
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py         # 相関・コインテグレーション分析
│   │   │   ├── spread.py           # スプレッド計算
│   │   │   └── strategy.py         # ペアトレード戦略実装
│   │   │
│   │   ├── trend_following/        # トレンドフォロー戦略
│   │   │   ├── __init__.py
│   │   │   ├── indicators.py       # テクニカル指標
│   │   │   └── strategy.py         # トレンドフォロー戦略実装
│   │   │
│   │   └── signals.py              # シグナル統合
│   │
│   ├── risk/                       # リスク管理
│   │   ├── __init__.py
│   │   ├── manager.py              # リスクマネージャー
│   │   ├── position_sizer.py       # ポジションサイズ計算
│   │   ├── stop_loss.py            # ストップロス管理
│   │   └── exposure.py             # エクスポージャー管理
│   │
│   ├── execution/                  # 注文執行
│   │   ├── __init__.py
│   │   ├── engine.py               # 執行エンジン
│   │   ├── order_manager.py        # 注文管理
│   │   └── position_manager.py     # ポジション管理
│   │
│   ├── backtest/                   # バックテスト
│   │   ├── __init__.py
│   │   ├── engine.py               # バックテストエンジン
│   │   ├── data_loader.py          # データローダー
│   │   ├── metrics.py              # パフォーマンス指標
│   │   └── walk_forward.py         # ウォークフォワード分析
│   │
│   ├── core/                       # コアエンジン
│   │   ├── __init__.py
│   │   ├── engine.py               # メインエンジン
│   │   ├── scheduler.py            # スケジューラー
│   │   └── event_bus.py            # イベントバス
│   │
│   └── utils/                      # ユーティリティ
│       ├── __init__.py
│       ├── logger.py               # ロギング
│       ├── decorators.py           # デコレーター
│       └── helpers.py              # ヘルパー関数
│
├── dashboard/                      # 監視ダッシュボード
│   ├── __init__.py
│   ├── app.py                      # Streamlit アプリ
│   ├── pages/                      # ダッシュボードページ
│   │   ├── overview.py             # 概要ページ
│   │   ├── positions.py            # ポジション一覧
│   │   ├── performance.py          # パフォーマンス
│   │   ├── backtest.py             # バックテスト結果
│   │   └── settings.py             # 設定ページ
│   └── components/                 # UIコンポーネント
│       ├── charts.py               # チャート
│       └── tables.py               # テーブル
│
├── alerts/                         # アラート通知
│   ├── __init__.py
│   ├── manager.py                  # アラートマネージャー
│   ├── telegram.py                 # Telegram通知
│   └── discord.py                  # Discord通知（オプション）
│
├── scripts/                        # 運用スクリプト
│   ├── run_bot.py                  # ボット起動
│   ├── run_backtest.py             # バックテスト実行
│   ├── run_walkforward.py          # ウォークフォワード実行
│   ├── run_dashboard.py            # ダッシュボード起動
│   └── db_migrate.py               # DB マイグレーション
│
├── tests/                          # テストコード
│   ├── __init__.py
│   ├── conftest.py                 # pytest設定
│   ├── test_api/                   # APIテスト
│   ├── test_strategy/              # 戦略テスト
│   ├── test_risk/                  # リスク管理テスト
│   ├── test_execution/             # 執行テスト
│   └── test_backtest/              # バックテストテスト
│
├── notebooks/                      # Jupyter Notebooks（分析用）
│   ├── 01_data_exploration.ipynb
│   ├── 02_pair_analysis.ipynb
│   ├── 03_strategy_research.ipynb
│   └── 04_performance_analysis.ipynb
│
├── data/                           # ローカルデータ（gitignore）
│   ├── historical/                 # 過去データ
│   ├── backtest_results/           # バックテスト結果
│   └── logs/                       # ログファイル
│
└── docs/                           # ドキュメント
    ├── ARCHITECTURE.md             # アーキテクチャ設計
    ├── PROJECT_STRUCTURE.md        # プロジェクト構造（本ファイル）
    ├── API_REFERENCE.md            # API リファレンス
    ├── STRATEGY_GUIDE.md           # 戦略ガイド
    ├── DEPLOYMENT.md               # デプロイ手順
    └── TROUBLESHOOTING.md          # トラブルシューティング
```

## モジュール依存関係

```
                    ┌─────────────┐
                    │   config    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │  utils  │       │   api   │       │  data   │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                  │
        └────────────┬────┴──────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │strategy │  │  risk   │  │execution│
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
               ┌──────────┐
               │   core   │
               └────┬─────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │backtest │ │dashboard│ │ alerts  │
   └─────────┘ └─────────┘ └─────────┘
```

## 主要ファイルの責務

| ファイル | 責務 |
|----------|------|
| `src/core/engine.py` | システム全体の制御、各コンポーネントの連携 |
| `src/api/client.py` | GMO Coin APIとの通信抽象化 |
| `src/strategy/base.py` | 戦略の共通インターフェース定義 |
| `src/risk/manager.py` | リスクルールの適用、取引可否判断 |
| `src/execution/engine.py` | 注文の発行・管理・約定処理 |
| `src/backtest/engine.py` | 過去データでの戦略シミュレーション |
| `dashboard/app.py` | Streamlitダッシュボードのエントリーポイント |
