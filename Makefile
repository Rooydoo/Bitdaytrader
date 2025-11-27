.PHONY: help venv install install-dev setup test lint format typecheck clean run-bot run-backtest run-dashboard

# Default target
help:
	@echo "Bitdaytrader - GMO Coin Day Trading System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  venv          Create virtual environment"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup         Full setup (venv + install)"
	@echo ""
	@echo "Development:"
	@echo "  test          Run tests"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (black + ruff)"
	@echo "  typecheck     Run type checker (mypy)"
	@echo "  clean         Remove cache and build files"
	@echo ""
	@echo "Run:"
	@echo "  run-bot       Start trading bot"
	@echo "  run-backtest  Run backtest"
	@echo "  run-dashboard Start monitoring dashboard"
	@echo ""
	@echo "Utilities:"
	@echo "  train-model   Train prediction models"
	@echo "  update-pairs  Update pair correlations"

# Virtual Environment
venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Run: source .venv/bin/activate"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"

setup: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	mkdir -p data/historical data/backtest_results data/logs data/models
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - please edit with your API keys"; fi
	@echo ""
	@echo "Setup complete!"
	@echo "  1. Run: source .venv/bin/activate"
	@echo "  2. Edit .env with your API keys"
	@echo "  3. Run: make run-dashboard"

# Development
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --ff

lint:
	ruff check src/ tests/ scripts/ dashboard/

format:
	black src/ tests/ scripts/ dashboard/ config/
	ruff check --fix src/ tests/ scripts/ dashboard/

typecheck:
	mypy src/

check: lint typecheck test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

clean-all: clean
	rm -rf .venv

# Run
run-bot:
	python scripts/run_bot.py

run-bot-paper:
	TRADING_MODE=paper python scripts/run_bot.py

run-backtest:
	python scripts/run_backtest.py

run-walkforward:
	python scripts/run_walkforward.py

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

# Model Training
train-model:
	python scripts/train_models.py

update-pairs:
	python scripts/update_pair_correlations.py

# Utilities
shell:
	ipython

notebook:
	jupyter notebook notebooks/

# Logs
logs:
	tail -f data/logs/*.log

# Database (SQLite for dev, PostgreSQL for prod)
db-init:
	python scripts/db_migrate.py

db-backup:
	@mkdir -p backups
	@if [ -f data/bitdaytrader.db ]; then \
		cp data/bitdaytrader.db backups/backup_$$(date +%Y%m%d_%H%M%S).db; \
		echo "Backup created in backups/"; \
	else \
		echo "No database file found"; \
	fi
