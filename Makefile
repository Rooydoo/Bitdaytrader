.PHONY: help install install-dev setup db-up db-down db-logs test lint format typecheck clean run-bot run-backtest run-dashboard

# Default target
help:
	@echo "Bitdaytrader - GMO Coin Day Trading System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup         Full setup (install + db)"
	@echo ""
	@echo "Database:"
	@echo "  db-up         Start PostgreSQL and Redis containers"
	@echo "  db-down       Stop database containers"
	@echo "  db-logs       Show database logs"
	@echo "  db-tools      Start with pgAdmin and Redis Commander"
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

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"
	pre-commit install

setup: install-dev db-up
	@echo "Setup complete! Run 'make run-dashboard' to start."

# Database
db-up:
	docker-compose up -d postgres redis
	@echo "Waiting for databases to be ready..."
	@sleep 5
	@docker-compose ps

db-down:
	docker-compose down

db-logs:
	docker-compose logs -f postgres redis

db-tools:
	docker-compose --profile tools up -d
	@echo "pgAdmin: http://localhost:5050"
	@echo "Redis Commander: http://localhost:8081"

db-reset:
	docker-compose down -v
	docker-compose up -d postgres redis
	@sleep 5
	@echo "Database reset complete"

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

# Utilities
migrate:
	alembic upgrade head

migrate-create:
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

shell:
	ipython

notebook:
	jupyter notebook notebooks/

# Logs
logs:
	tail -f data/logs/*.log

# Backup
backup-db:
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U trader bitdaytrader > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/"

restore-db:
	@read -p "Backup file: " file; \
	docker-compose exec -T postgres psql -U trader bitdaytrader < $$file
