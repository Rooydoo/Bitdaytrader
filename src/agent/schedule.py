"""Schedule management for Meta AI Agent."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine

from loguru import logger

from src.utils.timezone import now_jst, JST


class TaskFrequency(str, Enum):
    """Task frequency types."""

    INTERVAL = "interval"  # Run at fixed intervals
    DAILY = "daily"        # Run at specific time each day
    WEEKLY = "weekly"      # Run at specific time each week
    MONTHLY = "monthly"    # Run at specific time each month


@dataclass
class ScheduledTask:
    """Definition of a scheduled task."""

    name: str
    task_func: Callable[[], Coroutine[Any, Any, Any]]
    frequency: TaskFrequency
    enabled: bool = True

    # For interval-based tasks
    interval: timedelta | None = None

    # For time-based tasks
    run_time: time | None = None  # Time of day to run (JST)
    run_day: int | None = None    # Day of week (0=Mon) or day of month

    # Tracking
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str | None = None

    def calculate_next_run(self) -> datetime:
        """Calculate the next run time."""
        now = now_jst()

        match self.frequency:
            case TaskFrequency.INTERVAL:
                if self.interval is None:
                    raise ValueError(f"Interval task {self.name} has no interval set")
                if self.last_run is None:
                    return now
                return self.last_run + self.interval

            case TaskFrequency.DAILY:
                if self.run_time is None:
                    raise ValueError(f"Daily task {self.name} has no run_time set")

                # Get today's run time
                today_run = datetime.combine(now.date(), self.run_time, tzinfo=JST)

                # If we've already passed today's run time, schedule for tomorrow
                if now >= today_run:
                    return today_run + timedelta(days=1)
                return today_run

            case TaskFrequency.WEEKLY:
                if self.run_time is None or self.run_day is None:
                    raise ValueError(f"Weekly task {self.name} needs run_time and run_day")

                # Find next occurrence of run_day
                days_ahead = self.run_day - now.weekday()
                if days_ahead < 0:
                    days_ahead += 7

                next_date = now.date() + timedelta(days=days_ahead)
                next_run = datetime.combine(next_date, self.run_time, tzinfo=JST)

                # If it's the same day but time has passed, go to next week
                if next_run <= now:
                    next_run += timedelta(days=7)

                return next_run

            case TaskFrequency.MONTHLY:
                if self.run_time is None or self.run_day is None:
                    raise ValueError(f"Monthly task {self.name} needs run_time and run_day")

                # Get this month's run date
                try:
                    this_month_run = datetime(
                        now.year, now.month, self.run_day,
                        self.run_time.hour, self.run_time.minute,
                        tzinfo=JST
                    )
                except ValueError:
                    # Day doesn't exist in this month (e.g., 31st in February)
                    # Use last day of month
                    import calendar
                    last_day = calendar.monthrange(now.year, now.month)[1]
                    this_month_run = datetime(
                        now.year, now.month, last_day,
                        self.run_time.hour, self.run_time.minute,
                        tzinfo=JST
                    )

                if this_month_run > now:
                    return this_month_run

                # Schedule for next month
                if now.month == 12:
                    next_month = datetime(now.year + 1, 1, 1, tzinfo=JST)
                else:
                    next_month = datetime(now.year, now.month + 1, 1, tzinfo=JST)

                try:
                    return datetime(
                        next_month.year, next_month.month, self.run_day,
                        self.run_time.hour, self.run_time.minute,
                        tzinfo=JST
                    )
                except ValueError:
                    import calendar
                    last_day = calendar.monthrange(next_month.year, next_month.month)[1]
                    return datetime(
                        next_month.year, next_month.month, last_day,
                        self.run_time.hour, self.run_time.minute,
                        tzinfo=JST
                    )

            case _:
                raise ValueError(f"Unknown frequency: {self.frequency}")

    def is_due(self) -> bool:
        """Check if task is due to run."""
        if not self.enabled:
            return False

        if self.next_run is None:
            self.next_run = self.calculate_next_run()

        return now_jst() >= self.next_run

    def mark_run(self, success: bool = True, error: str | None = None) -> None:
        """Mark task as run."""
        self.last_run = now_jst()
        self.run_count += 1

        if not success:
            self.error_count += 1
            self.last_error = error

        # Calculate next run
        self.next_run = self.calculate_next_run()


class Scheduler:
    """
    Scheduler for managing periodic tasks.
    Runs within the agent's main loop.
    """

    def __init__(self) -> None:
        """Initialize scheduler."""
        self.tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._task_lock = asyncio.Lock()

    def add_task(
        self,
        name: str,
        task_func: Callable[[], Coroutine[Any, Any, Any]],
        frequency: TaskFrequency,
        interval: timedelta | None = None,
        run_time: time | None = None,
        run_day: int | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Add a scheduled task.

        Args:
            name: Unique task name
            task_func: Async function to call
            frequency: How often to run
            interval: For interval tasks, the timedelta between runs
            run_time: For daily/weekly/monthly, the time to run
            run_day: For weekly (0-6) or monthly (1-31)
            enabled: Whether task is enabled
        """
        task = ScheduledTask(
            name=name,
            task_func=task_func,
            frequency=frequency,
            interval=interval,
            run_time=run_time,
            run_day=run_day,
            enabled=enabled,
        )

        # Calculate initial next_run
        task.next_run = task.calculate_next_run()

        self.tasks[name] = task
        logger.info(f"Scheduled task added: {name} ({frequency.value}), next run: {task.next_run}")

    def remove_task(self, name: str) -> bool:
        """Remove a task."""
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Scheduled task removed: {name}")
            return True
        return False

    def enable_task(self, name: str) -> bool:
        """Enable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = True
            self.tasks[name].next_run = self.tasks[name].calculate_next_run()
            logger.info(f"Scheduled task enabled: {name}")
            return True
        return False

    def disable_task(self, name: str) -> bool:
        """Disable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = False
            logger.info(f"Scheduled task disabled: {name}")
            return True
        return False

    async def check_and_run(self) -> list[str]:
        """
        Check for due tasks and run them.

        Returns:
            List of task names that were run
        """
        async with self._task_lock:
            run_tasks = []

            for name, task in self.tasks.items():
                if task.is_due():
                    logger.info(f"Running scheduled task: {name}")

                    try:
                        await task.task_func()
                        task.mark_run(success=True)
                        run_tasks.append(name)
                        logger.info(f"Scheduled task completed: {name}, next run: {task.next_run}")

                    except Exception as e:
                        error_msg = str(e)
                        task.mark_run(success=False, error=error_msg)
                        logger.error(f"Scheduled task failed: {name} - {error_msg}")

            return run_tasks

    def get_task_status(self, name: str) -> dict | None:
        """Get status of a specific task."""
        if name not in self.tasks:
            return None

        task = self.tasks[name]
        return {
            "name": task.name,
            "frequency": task.frequency.value,
            "enabled": task.enabled,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "error_count": task.error_count,
            "last_error": task.last_error,
        }

    def get_all_status(self) -> list[dict]:
        """Get status of all tasks."""
        return [self.get_task_status(name) for name in self.tasks]

    def get_upcoming_tasks(self, hours: int = 24) -> list[dict]:
        """Get tasks scheduled to run in the next N hours."""
        cutoff = now_jst() + timedelta(hours=hours)
        upcoming = []

        for task in self.tasks.values():
            if task.enabled and task.next_run and task.next_run <= cutoff:
                upcoming.append({
                    "name": task.name,
                    "next_run": task.next_run.isoformat(),
                    "frequency": task.frequency.value,
                })

        return sorted(upcoming, key=lambda x: x["next_run"])


def create_default_scheduler() -> Scheduler:
    """Create scheduler with default task definitions (without actual task functions)."""
    scheduler = Scheduler()

    # Note: Task functions will be added by the MetaAgent
    # These are just the schedule definitions

    return scheduler


# Default task schedule definitions
DEFAULT_TASKS = {
    "market_check": {
        "frequency": TaskFrequency.INTERVAL,
        "interval": timedelta(minutes=1),
        "description": "市場状態の確認",
    },
    "signal_verification": {
        "frequency": TaskFrequency.INTERVAL,
        "interval": timedelta(minutes=15),
        "description": "シグナルの事後検証",
    },
    "performance_snapshot": {
        "frequency": TaskFrequency.INTERVAL,
        "interval": timedelta(hours=1),
        "description": "パフォーマンススナップショット",
    },
    "daily_review": {
        "frequency": TaskFrequency.DAILY,
        "run_time": time(21, 0),  # 21:00 JST
        "description": "日次レビュー（反省会）",
    },
    "morning_prep": {
        "frequency": TaskFrequency.DAILY,
        "run_time": time(8, 0),  # 08:00 JST
        "description": "朝の準備・状態確認",
    },
    "weekly_summary": {
        "frequency": TaskFrequency.WEEKLY,
        "run_time": time(20, 0),  # 20:00 JST
        "run_day": 6,  # Sunday
        "description": "週次サマリー",
    },
    "model_evaluation": {
        "frequency": TaskFrequency.WEEKLY,
        "run_time": time(3, 0),  # 03:00 JST (low activity time)
        "run_day": 0,  # Monday
        "description": "モデルパフォーマンス評価",
    },
}
