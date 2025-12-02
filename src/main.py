"""Main entry point for the trading bot."""

import asyncio
import sys

from loguru import logger

from config.settings import get_settings
from src.core.engine import TradingEngine
from src.core.safety import (
    ProcessLock,
    StartupValidator,
    PositionReconciler,
    create_crash_indicator,
    remove_crash_indicator,
)
from src.utils.timezone import now_jst


# Global process lock
_process_lock: ProcessLock | None = None


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


def acquire_process_lock() -> bool:
    """
    Acquire process lock to prevent multiple instances.

    Returns:
        True if lock acquired, False if another instance is running
    """
    global _process_lock
    _process_lock = ProcessLock()

    if not _process_lock.acquire():
        owner_pid = _process_lock.get_owner_pid()
        if _process_lock.is_owner_alive():
            logger.error(
                f"Another instance is already running (PID: {owner_pid}). "
                "Kill the other instance or wait for it to finish."
            )
            return False
        else:
            # Stale lock from crashed process
            logger.warning(f"Found stale lock from dead process (PID: {owner_pid}), cleaning up")
            if _process_lock.force_release():
                return _process_lock.acquire()
            return False

    return True


def run_startup_validation(settings) -> bool:
    """
    Run startup validation checks.

    Returns:
        True if all critical checks pass
    """
    validator = StartupValidator(
        db_path=settings.db_path,
        model_path=settings.model_path,
    )

    result = validator.validate()

    # Log results
    for check in result.checks_passed:
        logger.debug(f"✓ {check}")

    for warning in result.warnings:
        logger.warning(f"⚠ {warning}")

    for check in result.checks_failed:
        logger.error(f"✗ {check}")

    for rec in result.recommendations:
        logger.info(f"💡 Recommendation: {rec}")

    if not result.is_valid:
        logger.error("Startup validation failed - some critical checks did not pass")
        return False

    logger.info("Startup validation passed")
    return True


async def run_position_reconciliation(engine: TradingEngine) -> None:
    """
    Reconcile positions between database and exchange.

    Args:
        engine: TradingEngine instance
    """
    if engine.settings.mode != "live":
        logger.debug("Position reconciliation skipped (not in live mode)")
        return

    try:
        reconciler = PositionReconciler(
            gmo_client=engine.client,
            trade_repository=engine.trade_repo,
        )

        result = reconciler.reconcile(engine.settings.symbol)

        if not result.is_consistent:
            # Log issues
            for orphan in result.orphan_local:
                logger.warning(f"Orphan local position: {orphan}")

            for orphan in result.orphan_exchange:
                logger.warning(f"Orphan exchange position: {orphan}")

            for mismatch in result.size_mismatches:
                logger.warning(f"Size mismatch: {mismatch}")

            # Notify via Telegram
            await engine.telegram.notify_error(
                f"Position reconciliation found issues:\n"
                f"- Orphan local: {len(result.orphan_local)}\n"
                f"- Orphan exchange: {len(result.orphan_exchange)}\n"
                f"- Size mismatches: {len(result.size_mismatches)}",
                "position_reconciliation",
            )
        else:
            logger.info("Position reconciliation: all positions match")

    except Exception as e:
        logger.error(f"Position reconciliation failed: {e}")


async def main() -> None:
    """Main trading loop."""
    global _process_lock

    setup_logging()
    logger.info("Starting Bitdaytrader")

    # Acquire process lock
    if not acquire_process_lock():
        sys.exit(1)

    settings = get_settings()

    # Run startup validation
    if not run_startup_validation(settings):
        logger.error("Startup validation failed, exiting")
        if _process_lock:
            _process_lock.release()
        sys.exit(1)

    # Create crash indicator (removed on clean shutdown)
    crash_indicator = create_crash_indicator()

    engine = TradingEngine(settings)

    try:
        # Run position reconciliation on startup (for live mode)
        await run_position_reconciliation(engine)

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

        # Clean shutdown - remove crash indicator
        remove_crash_indicator()
        logger.info("Trading cycle completed successfully")

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        remove_crash_indicator()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        # Don't remove crash indicator on error - helps with debugging
        raise
    finally:
        engine.close()
        if _process_lock:
            _process_lock.release()
        logger.info("Bitdaytrader stopped")


def run() -> None:
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
