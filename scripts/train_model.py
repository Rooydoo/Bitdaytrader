#!/usr/bin/env python3
"""Script to train the LightGBM model."""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import train_and_save


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to OHLCV CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/lightgbm_model.joblib",
        help="Output model path",
    )
    parser.add_argument(
        "--training-months",
        type=int,
        default=12,
        help="Months of data for training",
    )
    parser.add_argument(
        "--backtest-months",
        type=int,
        default=2,
        help="Months of data for backtesting",
    )

    args = parser.parse_args()

    logger.info(f"Training model with {args.training_months} months training, {args.backtest_months} months backtest")

    train_and_save(
        data_path=args.data,
        output_path=args.output,
        training_months=args.training_months,
        backtest_months=args.backtest_months,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
