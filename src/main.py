"""Main entry point for the trading bot."""

import asyncio
import sys

from loguru import logger

from config.settings import get_settings
from src.core.engine import TradingEngine
from src.utils.timezone import now_jst


def setup_logging() -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )


async def main() -> None:
    """Main trading loop."""
    setup_logging()
    logger.info("Starting Bitdaytrader")

    settings = get_settings()
    engine = TradingEngine(settings)

    try:
        # Get current hour for report scheduling (JST)
        now = now_jst()
        hour = now.hour

        # Morning report (8:00)
        if hour == settings.report_morning_hour:
            await engine.send_status_report("朝")

        # Noon report (12:00)
        elif hour == settings.report_noon_hour:
            await engine.send_status_report("昼")

        # Evening report (20:00)
        elif hour == settings.report_evening_hour:
            await engine.send_daily_report()

        # Weekly report (Monday 8:00)
        if now.weekday() == settings.report_weekly_day and hour == settings.report_morning_hour:
            await engine.send_weekly_report()

        # Bi-weekly model analysis report (every other Monday 8:00)
        # Based on ISO week number: send on even weeks
        if (
            now.weekday() == settings.report_weekly_day
            and hour == settings.report_morning_hour
            and now.isocalendar()[1] % 2 == 0
        ):
            await engine.send_model_analysis_report()

        # Monthly report (1st of month, 8:00)
        if now.day == settings.report_monthly_day and hour == settings.report_morning_hour:
            await engine.send_monthly_report()

        # Run main trading cycle
        await engine.run_cycle()

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        engine.close()
        logger.info("Bitdaytrader stopped")


def run() -> None:
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
