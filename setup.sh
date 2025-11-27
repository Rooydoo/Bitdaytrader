#!/bin/bash
# setup.sh - Bitdaytrader セットアップスクリプト

set -e

echo "=== Bitdaytrader Setup ==="

# Python バージョン確認
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Python 3.11+ is required (found: $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION"

# venv 作成
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# venv アクティベート
source .venv/bin/activate

# pip アップグレード
echo "Upgrading pip..."
pip install --upgrade pip

# 依存パッケージインストール
echo "Installing dependencies..."
pip install -e ".[dev]"

# .env ファイル作成（存在しない場合）
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env and add your API keys"
fi

# データディレクトリ作成
echo "Creating data directories..."
mkdir -p data/historical
mkdir -p data/backtest_results
mkdir -p data/logs
mkdir -p data/models

# PostgreSQL/Redis インストール確認
echo ""
echo "=== Database Setup ==="
echo "Please ensure the following are installed and running:"
echo "  - PostgreSQL 16+ (or SQLite for development)"
echo "  - Redis 7+"
echo ""
echo "Install on Ubuntu/Debian:"
echo "  sudo apt install postgresql postgresql-contrib redis-server"
echo ""
echo "Install on macOS:"
echo "  brew install postgresql@16 redis"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your GMO Coin API keys"
echo "  2. Start PostgreSQL and Redis"
echo "  3. Run: source .venv/bin/activate"
echo "  4. Run: make run-dashboard"
