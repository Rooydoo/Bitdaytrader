#!/usr/bin/env python3
"""Run the trading bot.

This script is the main entry point for running Bitdaytrader.
It can be invoked via:
  - python scripts/run_bot.py
  - python -m scripts.run_bot
  - bitdaytrader (if installed via pip install -e .)

Environment variables:
  MODE: 'paper' or 'live' (default: from .env, fallback 'paper')
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import main as run_main


def main() -> None:
    """Run the trading bot."""
    asyncio.run(run_main())


if __name__ == "__main__":
    main()
