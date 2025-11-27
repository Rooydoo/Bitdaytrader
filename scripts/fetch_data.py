#!/usr/bin/env python3
"""Script to fetch historical OHLCV data from GMO Coin."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.api.gmo_client import GMOCoinClient


def fetch_historical_data(
    client: GMOCoinClient,
    symbol: str,
    interval: str,
    days: int,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data.

    Args:
        client: GMO Coin client
        symbol: Trading symbol
        interval: Candle interval
        days: Number of days to fetch

    Returns:
        DataFrame with OHLCV data
    """
    all_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        logger.info(f"Fetching data for {date_str}")

        try:
            df = client.get_klines(
                symbol=symbol,
                interval=interval,
                date=date_str,
            )
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch {date_str}: {e}")

        current_date += timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    # Combine and sort
    result = pd.concat(all_data).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to fetch (default: 365)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC_JPY",
        help="Trading symbol (default: BTC_JPY)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15min",
        help="Candle interval (default: 15min)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    settings = get_settings()

    with GMOCoinClient(
        api_key=settings.gmo_api_key,
        api_secret=settings.gmo_api_secret,
    ) as client:
        logger.info(f"Fetching {args.days} days of {args.symbol} {args.interval} data")

        df = fetch_historical_data(
            client=client,
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
        )

        if df.empty:
            logger.error("No data fetched")
            return

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} candles to {output_path}")


if __name__ == "__main__":
    main()
