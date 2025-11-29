#!/usr/bin/env python3
"""Run the trading bot as a daemon with Telegram command support.

This script runs continuously and:
1. Executes trading cycles every 15 minutes
2. Listens for Telegram commands
3. Handles graceful shutdown
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from config.settings import get_settings
from src.core.engine import TradingEngine
from src.telegram.commands import TelegramCommandHandler
from src.utils.timezone import now_jst


class TradingDaemon:
    """Trading daemon with Telegram command support."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine: TradingEngine | None = None
        self.telegram_handler: TelegramCommandHandler | None = None
        self.running = False
        self._last_cycle: datetime | None = None
        self._cycle_interval = timedelta(minutes=15)

    async def start(self) -> None:
        """Start the daemon."""
        logger.info("Starting Trading Daemon")

        # Initialize engine
        self.engine = TradingEngine(self.settings)

        # Initialize Telegram command handler
        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            self.telegram_handler = TelegramCommandHandler(
                token=self.settings.telegram_bot_token,
                chat_id=self.settings.telegram_chat_id,
            )
            self.telegram_handler.set_engine(self.engine)
            logger.info("Telegram command handler initialized")

        self.running = True

        # Run both tasks concurrently
        tasks = [
            asyncio.create_task(self._trading_loop()),
        ]

        if self.telegram_handler:
            tasks.append(asyncio.create_task(self._telegram_loop()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Daemon tasks cancelled")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        logger.info("Stopping Trading Daemon")
        self.running = False

        if self.engine:
            self.engine.close()

    async def _trading_loop(self) -> None:
        """Main trading loop - runs cycles every 15 minutes."""
        logger.info("Trading loop started")

        while self.running:
            try:
                now = now_jst()

                # Check if it's time for a cycle (aligned to 15-min intervals)
                should_run = False
                if self._last_cycle is None:
                    should_run = True
                elif now - self._last_cycle >= self._cycle_interval:
                    should_run = True

                # Also run at specific minutes (0, 15, 30, 45)
                if now.minute in [0, 15, 30, 45] and now.second < 30:
                    if self._last_cycle is None or self._last_cycle.minute != now.minute:
                        should_run = True

                if should_run:
                    logger.info(f"Running trading cycle at {now}")
                    await self._run_cycle()
                    self._last_cycle = now

                # Sleep for a short interval
                await asyncio.sleep(10)

            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _run_cycle(self) -> None:
        """Run a single trading cycle with reports."""
        if not self.engine:
            return

        try:
            now = now_jst()
            hour = now.hour

            # Morning report (8:00)
            if hour == self.settings.report_morning_hour and now.minute < 15:
                await self.engine.send_status_report("朝")

            # Noon report (12:00)
            elif hour == self.settings.report_noon_hour and now.minute < 15:
                await self.engine.send_status_report("昼")

            # Evening report (20:00)
            elif hour == self.settings.report_evening_hour and now.minute < 15:
                await self.engine.send_daily_report()

            # Weekly report (Monday 8:00)
            if (
                now.weekday() == self.settings.report_weekly_day
                and hour == self.settings.report_morning_hour
                and now.minute < 15
            ):
                await self.engine.send_weekly_report()

            # Bi-weekly model analysis report
            if (
                now.weekday() == self.settings.report_weekly_day
                and hour == self.settings.report_morning_hour
                and now.minute < 15
                and now.isocalendar()[1] % 2 == 0
            ):
                await self.engine.send_model_analysis_report()

            # Monthly report (1st of month, 8:00)
            if (
                now.day == self.settings.report_monthly_day
                and hour == self.settings.report_morning_hour
                and now.minute < 15
            ):
                await self.engine.send_monthly_report()

            # Run main trading cycle
            await self.engine.run_cycle()

        except Exception as e:
            logger.exception(f"Error in cycle: {e}")

    async def _telegram_loop(self) -> None:
        """Telegram bot polling loop."""
        if not self.telegram_handler:
            return

        logger.info("Starting Telegram bot polling")

        app = self.telegram_handler.build_application()

        try:
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)

            # Keep running while daemon is active
            while self.running:
                await asyncio.sleep(1)

            await app.updater.stop()
            await app.stop()
            await app.shutdown()

        except Exception as e:
            logger.exception(f"Error in Telegram loop: {e}")


def setup_logging() -> None:
    """Configure logging for daemon mode."""
    logger.remove()

    # Console output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # File output
    logger.add(
        "/root/Bitdaytrader/logs/daemon_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )


async def main() -> None:
    """Main entry point."""
    setup_logging()

    daemon = TradingDaemon()

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(daemon.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await daemon.stop()
        logger.info("Daemon stopped")


if __name__ == "__main__":
    asyncio.run(main())
