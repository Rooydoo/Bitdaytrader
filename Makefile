.PHONY: help venv install install-dev install-training setup test lint format typecheck clean run train fetch-data

# Default target
help:
	@echo "Bitdaytrader - GMO Coin Day Trading System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  venv             Create virtual environment"
	@echo "  install          Install production dependencies (VPS)"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-training Install training dependencies (local PC)"
	@echo "  setup            Full setup (venv + install)"
	@echo ""
	@echo "Development:"
	@echo "  test             Run tests"
	@echo "  lint             Run linter (ruff)"
	@echo "  format           Format code (black + ruff)"
	@echo "  typecheck        Run type checker (mypy)"
	@echo "  clean            Remove cache and build files"
	@echo ""
	@echo "Run:"
	@echo "  run              Run trading cycle (VPS)"
	@echo "  run-paper        Run in paper trading mode"
	@echo ""
	@echo "Training (local PC):"
	@echo "  fetch-data       Fetch historical data from GMO"
	@echo "  train            Train LightGBM model"
	@echo ""
	@echo "Utilities:"
	@echo "  logs             Tail log files"
	@echo "  shell            Start IPython shell"

# Virtual Environment
venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Run: source .venv/bin/activate"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-training:
	pip install -e ".[training]"

setup: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	mkdir -p data logs models
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - please edit with your API keys"; fi
	@echo ""
	@echo "Setup complete!"
	@echo "  1. Run: source .venv/bin/activate"
	@echo "  2. Edit .env with your API keys"
	@echo "  3. Set up cron: */15 * * * * cd $(pwd) && .venv/bin/python -m src.main"

# Development
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --ff

lint:
	ruff check src/ tests/ scripts/ config/ training/

format:
	black src/ tests/ scripts/ config/ training/
	ruff check --fix src/ tests/ scripts/ config/ training/

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

# Run (VPS)
run:
	python -m src.main

run-paper:
	MODE=paper python -m src.main

# Training (local PC)
fetch-data:
	python scripts/fetch_data.py --days 420 --output data/historical.csv

train:
	python scripts/train_model.py --data data/historical.csv --output models/lightgbm_model.joblib

# Utilities
shell:
	ipython

logs:
	tail -f logs/*.log

# Database
db-backup:
	@mkdir -p backups
	@if [ -f data/trading.db ]; then \
		cp data/trading.db backups/backup_$$(date +%Y%m%d_%H%M%S).db; \
		echo "Backup created in backups/"; \
	else \
		echo "No database file found"; \
	fi

# Cron setup helper
cron-setup:
	@echo "Add this line to your crontab (crontab -e):"
	@echo ""
	@echo "*/15 * * * * cd $$(pwd) && .venv/bin/python -m src.main >> logs/cron.log 2>&1"
	@echo ""
