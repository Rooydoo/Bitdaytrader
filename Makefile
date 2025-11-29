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
	@echo "Paper Trading Test:"
	@echo "  paper-test-setup Setup 1-month paper trading test"
	@echo "  paper-cron-setup Show cron setup for paper trading"
	@echo "  paper-status     Show current paper trading status"
	@echo "  paper-logs       Show recent paper trading logs"
	@echo "  paper-reset      Reset paper trading state"
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

# Paper Trading Test (1 month)
paper-test-setup:
	@echo "=== Paper Trading 1-Month Test Setup ==="
	@echo ""
	@echo "This will start a simulated trading test with:"
	@echo "  - Initial capital: 1,000,000 JPY (virtual)"
	@echo "  - Duration: 1 month continuous"
	@echo "  - Mode: Paper (no real money)"
	@echo ""
	@mkdir -p data logs
	@rm -f data/paper_trading_state.json
	@echo "Ready for paper trading test."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Set up cron: make paper-cron-setup"
	@echo "  2. Check status: make paper-status"
	@echo "  3. View logs: make paper-logs"

paper-cron-setup:
	@echo "Add this line to your crontab (crontab -e):"
	@echo ""
	@echo "# Bitdaytrader Paper Trading Test (every 15 min)"
	@echo "*/15 * * * * cd $$(pwd) && MODE=paper .venv/bin/python -m src.main >> logs/paper_test.log 2>&1"
	@echo ""
	@echo "To edit crontab, run: crontab -e"

paper-status:
	@echo "=== Paper Trading Status ==="
	@if [ -f data/paper_trading_state.json ]; then \
		python3 -c "import json; d=json.load(open('data/paper_trading_state.json')); \
		print(f\"Session start: {d['session_start']}\"); \
		print(f\"Initial capital: ¥{d['initial_capital']:,.0f}\"); \
		print(f\"Current capital: ¥{d['current_capital']:,.0f}\"); \
		print(f\"Peak capital: ¥{d['peak_capital']:,.0f}\"); \
		print(f\"Total PnL: ¥{d['total_pnl']:+,.0f}\"); \
		print(f\"Total trades: {d['total_trades']}\"); \
		print(f\"Win/Loss: {d['winning_trades']}/{d['losing_trades']}\"); \
		wr = d['winning_trades']/d['total_trades']*100 if d['total_trades']>0 else 0; \
		print(f\"Win rate: {wr:.1f}%\"); \
		print(f\"Commission: ¥{d['total_commission']:,.0f}\"); \
		"; \
	else \
		echo "No paper trading data found."; \
		echo "Run 'make paper-test-setup' then set up cron."; \
	fi

paper-logs:
	@if [ -f logs/paper_test.log ]; then \
		tail -50 logs/paper_test.log; \
	else \
		echo "No paper trading logs found."; \
	fi

paper-reset:
	@echo "Resetting paper trading state..."
	@rm -f data/paper_trading_state.json
	@echo "Done. Paper trading will start fresh on next cycle."

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

# Systemd service (recommended for production)
service-install:
	@echo "Installing systemd service..."
	cp deploy/bitdaytrader.service /etc/systemd/system/
	systemctl daemon-reload
	@echo "Service installed. Use 'make service-start' to start."

service-start:
	systemctl start bitdaytrader
	systemctl enable bitdaytrader
	@echo "Service started and enabled on boot."

service-stop:
	systemctl stop bitdaytrader
	@echo "Service stopped."

service-restart:
	systemctl restart bitdaytrader
	@echo "Service restarted."

service-status:
	systemctl status bitdaytrader

service-logs:
	journalctl -u bitdaytrader -f

service-uninstall:
	systemctl stop bitdaytrader || true
	systemctl disable bitdaytrader || true
	rm -f /etc/systemd/system/bitdaytrader.service
	systemctl daemon-reload
	@echo "Service uninstalled."
