#!/usr/bin/env python3
"""Run backtesting.

Usage:
  python scripts/run_backtest.py --data data/historical.csv --months 2
  bitdaytrader-backtest --data data/historical.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run backtest."""
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--data", type=str, default="data/historical.csv",
                        help="Historical data CSV file")
    parser.add_argument("--months", type=int, default=2,
                        help="Number of months to backtest")
    parser.add_argument("--output", type=str, default="data/backtest_results",
                        help="Output directory for results")

    args = parser.parse_args()

    from training.trainer import ModelTrainer

    print(f"Running backtest on {args.data}")
    print(f"Period: {args.months} months")

    trainer = ModelTrainer()

    # Load data
    import pandas as pd
    df = pd.read_csv(args.data, parse_dates=["timestamp"])

    # Run backtest
    results = trainer.backtest(
        df,
        model_path="models/lightgbm_model.joblib",
        months=args.months,
    )

    print("\n=== Backtest Results ===")
    print(f"Total trades: {results.get('total_trades', 0)}")
    print(f"Win rate: {results.get('win_rate', 0):.2%}")
    print(f"Profit factor: {results.get('profit_factor', 0):.2f}")
    print(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Total return: {results.get('total_return', 0):.2%}")


if __name__ == "__main__":
    main()
