"""Japan Standard Time (JST) utilities.

This module provides timezone-aware datetime utilities for consistent
time handling across the trading system. All times should be in JST
for Japanese market operations and reporting.
"""

from datetime import datetime, timedelta, timezone

# Japan Standard Time (UTC+9)
JST = timezone(timedelta(hours=9), name="JST")


def now_jst() -> datetime:
    """
    Get current time in JST (Japan Standard Time).

    Returns:
        datetime: Current time in JST with timezone info

    Example:
        >>> now = now_jst()
        >>> print(now.tzname())  # 'JST'
    """
    return datetime.now(JST)


def to_jst(dt: datetime) -> datetime:
    """
    Convert a datetime to JST.

    Args:
        dt: datetime to convert (can be naive or aware)

    Returns:
        datetime: Time in JST with timezone info

    Note:
        - If dt is naive (no timezone), it's assumed to be UTC
        - If dt is aware, it's converted to JST
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST)


def today_jst() -> datetime:
    """
    Get today's date at midnight in JST.

    Returns:
        datetime: Today's date at 00:00:00 JST
    """
    now = now_jst()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def format_jst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime in JST.

    Args:
        dt: datetime to format
        fmt: strftime format string

    Returns:
        str: Formatted datetime string
    """
    jst_dt = to_jst(dt) if dt.tzinfo != JST else dt
    return jst_dt.strftime(fmt)


def parse_jst(date_str: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse a datetime string as JST.

    Args:
        date_str: datetime string to parse
        fmt: strftime format string

    Returns:
        datetime: Parsed datetime in JST
    """
    dt = datetime.strptime(date_str, fmt)
    return dt.replace(tzinfo=JST)


def is_trading_hours(dt: datetime | None = None) -> bool:
    """
    Check if current time is within typical trading hours.

    Crypto markets are 24/7, but this can be used to
    avoid high-volatility periods or maintenance windows.

    Args:
        dt: datetime to check (default: now in JST)

    Returns:
        bool: True if within trading hours

    Note:
        GMO Coin has maintenance:
        - Every Wednesday 4:00-4:30 AM JST
        - Saturdays: specific times may vary
    """
    if dt is None:
        dt = now_jst()
    else:
        dt = to_jst(dt)

    # Check for GMO Coin maintenance (Wednesday 4:00-4:30 JST)
    if dt.weekday() == 2 and 4 <= dt.hour < 5:  # Wednesday, 4:00-4:59
        return False

    return True


def get_report_schedule_check(dt: datetime | None = None) -> dict:
    """
    Check if it's time for scheduled reports.

    Args:
        dt: datetime to check (default: now in JST)

    Returns:
        dict with keys:
            - morning: True if 8:00 JST
            - noon: True if 12:00 JST
            - evening: True if 20:00 JST
            - weekly: True if Monday 8:00 JST
            - monthly: True if 1st of month 8:00 JST
    """
    if dt is None:
        dt = now_jst()
    else:
        dt = to_jst(dt)

    hour = dt.hour

    return {
        "morning": hour == 8,
        "noon": hour == 12,
        "evening": hour == 20,
        "weekly": dt.weekday() == 0 and hour == 8,  # Monday 8:00
        "monthly": dt.day == 1 and hour == 8,
        "biweekly_analysis": (
            dt.weekday() == 0 and
            hour == 8 and
            dt.isocalendar()[1] % 2 == 0
        ),
    }
