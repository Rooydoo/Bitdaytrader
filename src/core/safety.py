"""System safety and production readiness features."""

import atexit
import fcntl
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    is_consistent: bool
    local_positions: list[dict[str, Any]] = field(default_factory=list)
    exchange_positions: list[dict[str, Any]] = field(default_factory=list)
    orphan_local: list[dict[str, Any]] = field(default_factory=list)  # In DB but not on exchange
    orphan_exchange: list[dict[str, Any]] = field(default_factory=list)  # On exchange but not in DB
    size_mismatches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StartupValidation:
    """Result of startup validation."""

    is_valid: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ProcessLock:
    """File-based process locking to prevent multiple instances."""

    def __init__(self, lock_file: str = "data/.trading_bot.lock") -> None:
        """
        Initialize process lock.

        Args:
            lock_file: Path to lock file
        """
        self.lock_file = Path(lock_file)
        self.lock_fd: int | None = None
        self._acquired = False

    def acquire(self) -> bool:
        """
        Try to acquire the process lock.

        Returns:
            True if lock acquired, False if another instance is running
        """
        try:
            # Create directory if needed
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)

            # Open lock file
            self.lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR)

            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                # Lock is held by another process
                os.close(self.lock_fd)
                self.lock_fd = None
                return False

            # Write PID to lock file
            os.ftruncate(self.lock_fd, 0)
            os.write(self.lock_fd, f"{os.getpid()}\n".encode())
            os.fsync(self.lock_fd)

            self._acquired = True

            # Register cleanup on exit
            atexit.register(self.release)

            # Handle signals
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            logger.info(f"Process lock acquired (PID: {os.getpid()})")
            return True

        except Exception as e:
            logger.error(f"Failed to acquire process lock: {e}")
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
            return False

    def release(self) -> None:
        """Release the process lock."""
        if self.lock_fd is not None and self._acquired:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = None
                self._acquired = False

                # Remove lock file
                if self.lock_file.exists():
                    self.lock_file.unlink()

                logger.info("Process lock released")
            except Exception as e:
                logger.error(f"Error releasing process lock: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, releasing lock and exiting")
        self.release()
        sys.exit(0)

    def get_owner_pid(self) -> int | None:
        """
        Get PID of process holding the lock.

        Returns:
            PID if lock is held, None otherwise
        """
        if not self.lock_file.exists():
            return None

        try:
            content = self.lock_file.read_text().strip()
            return int(content)
        except (ValueError, IOError):
            return None

    def is_owner_alive(self) -> bool:
        """Check if the lock owner process is still alive."""
        pid = self.get_owner_pid()
        if pid is None:
            return False

        try:
            # Check if process exists
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def force_release(self) -> bool:
        """
        Force release a stale lock (use with caution).

        Returns:
            True if lock was released, False otherwise
        """
        if not self.lock_file.exists():
            return True

        if self.is_owner_alive():
            logger.warning("Cannot force release - owner process is still alive")
            return False

        try:
            self.lock_file.unlink()
            logger.info("Stale lock file removed")
            return True
        except Exception as e:
            logger.error(f"Failed to force release lock: {e}")
            return False


class PositionReconciler:
    """Reconcile local database positions with exchange positions."""

    def __init__(self, gmo_client: Any, trade_repository: Any) -> None:
        """
        Initialize reconciler.

        Args:
            gmo_client: GMO API client
            trade_repository: Trade database repository
        """
        self.client = gmo_client
        self.trade_repo = trade_repository

    def reconcile(self, symbol: str = "BTC_JPY") -> ReconciliationResult:
        """
        Reconcile positions between local DB and exchange.

        Args:
            symbol: Trading symbol to reconcile

        Returns:
            ReconciliationResult with details
        """
        result = ReconciliationResult(is_consistent=True)

        try:
            # Get local open positions from DB
            local_trades = self.trade_repo.get_open_trades()
            local_positions = [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "size": t.size,
                    "entry_price": t.entry_price,
                    "is_paper": t.is_paper,
                }
                for t in local_trades
                if t.symbol == symbol and not t.is_paper
            ]
            result.local_positions = local_positions

            # Get exchange positions
            try:
                exchange_positions = self.client.get_positions(symbol)
                result.exchange_positions = [
                    {
                        "position_id": p["positionId"],
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "size": float(p["size"]),
                        "price": float(p["price"]),
                    }
                    for p in exchange_positions
                ]
            except Exception as e:
                result.errors.append(f"Failed to fetch exchange positions: {e}")
                result.is_consistent = False
                return result

            # Compare positions
            local_sides = {}
            for lp in local_positions:
                key = lp["side"]
                if key not in local_sides:
                    local_sides[key] = 0
                local_sides[key] += lp["size"]

            exchange_sides = {}
            for ep in result.exchange_positions:
                # Map exchange side to our format (BUY -> LONG, SELL -> SHORT)
                side = "LONG" if ep["side"] == "BUY" else "SHORT"
                if side not in exchange_sides:
                    exchange_sides[side] = 0
                exchange_sides[side] += ep["size"]

            # Check for orphan positions
            all_sides = set(local_sides.keys()) | set(exchange_sides.keys())

            for side in all_sides:
                local_size = local_sides.get(side, 0)
                exchange_size = exchange_sides.get(side, 0)

                if local_size > 0 and exchange_size == 0:
                    # Position in DB but not on exchange
                    result.orphan_local.append({
                        "side": side,
                        "size": local_size,
                        "issue": "Position in database but not on exchange",
                    })
                    result.is_consistent = False

                elif exchange_size > 0 and local_size == 0:
                    # Position on exchange but not in DB
                    result.orphan_exchange.append({
                        "side": side,
                        "size": exchange_size,
                        "issue": "Position on exchange but not in database",
                    })
                    result.is_consistent = False

                elif abs(local_size - exchange_size) > 0.00001:
                    # Size mismatch
                    result.size_mismatches.append({
                        "side": side,
                        "local_size": local_size,
                        "exchange_size": exchange_size,
                        "difference": exchange_size - local_size,
                    })
                    result.is_consistent = False

            if result.is_consistent:
                logger.info(f"Position reconciliation OK for {symbol}")
            else:
                logger.warning(f"Position reconciliation found issues for {symbol}")

        except Exception as e:
            result.errors.append(f"Reconciliation error: {e}")
            result.is_consistent = False
            logger.exception(f"Position reconciliation failed: {e}")

        return result

    def auto_fix_orphans(self, result: ReconciliationResult) -> dict[str, Any]:
        """
        Attempt to automatically fix orphan positions.

        Args:
            result: ReconciliationResult to fix

        Returns:
            Dict with fix results
        """
        fixes = {"local_closed": 0, "exchange_synced": 0, "errors": []}

        # Close orphan local positions (mark as closed in DB)
        for orphan in result.orphan_local:
            try:
                # Find and close matching trades
                trades = self.trade_repo.get_open_trades()
                for trade in trades:
                    if trade.side == orphan["side"] and not trade.is_paper:
                        self.trade_repo.update(trade.id, {
                            "status": "RECONCILED",
                            "notes": "Auto-closed by reconciliation - not found on exchange",
                        })
                        fixes["local_closed"] += 1
                        logger.info(f"Auto-closed orphan local trade {trade.id}")
            except Exception as e:
                fixes["errors"].append(f"Failed to close local orphan: {e}")

        # For orphan exchange positions, we create tracking records
        # (but don't close them - that requires manual review)
        for orphan in result.orphan_exchange:
            logger.warning(
                f"Orphan exchange position found: {orphan['side']} {orphan['size']} - "
                "manual review required"
            )
            fixes["exchange_synced"] += 1

        return fixes


class StartupValidator:
    """Validate system state on startup."""

    def __init__(
        self,
        db_path: str,
        model_path: str,
        gmo_client: Any | None = None,
    ) -> None:
        """
        Initialize validator.

        Args:
            db_path: Path to database
            model_path: Path to ML model
            gmo_client: Optional GMO client for API checks
        """
        self.db_path = Path(db_path)
        self.model_path = Path(model_path)
        self.client = gmo_client

    def validate(self) -> StartupValidation:
        """
        Run all startup validation checks.

        Returns:
            StartupValidation result
        """
        result = StartupValidation(is_valid=True)

        # Check database
        self._check_database(result)

        # Check model file
        self._check_model(result)

        # Check data directories
        self._check_directories(result)

        # Check API connectivity (if client provided)
        if self.client:
            self._check_api_connectivity(result)

        # Check for crash recovery
        self._check_crash_recovery(result)

        # Determine overall validity
        result.is_valid = len(result.checks_failed) == 0

        return result

    def _check_database(self, result: StartupValidation) -> None:
        """Check database integrity."""
        if not self.db_path.exists():
            result.checks_failed.append(f"Database file not found: {self.db_path}")
            result.recommendations.append("Run initial setup to create database")
            return

        try:
            import sqlite3

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]

            if integrity != "ok":
                result.checks_failed.append(f"Database integrity check failed: {integrity}")
                result.recommendations.append("Database may need to be restored from backup")
            else:
                result.checks_passed.append("Database integrity check passed")

            # Check WAL mode
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]

            if journal_mode.lower() != "wal":
                result.warnings.append(f"Database not in WAL mode (current: {journal_mode})")
                result.recommendations.append("Enable WAL mode for better concurrency")

            # Check for orphan open trades (potential crash recovery needed)
            cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE status = 'OPEN' AND is_paper = 0"
            )
            open_trades = cursor.fetchone()[0]

            if open_trades > 0:
                result.warnings.append(f"Found {open_trades} open trades in database")
                result.recommendations.append(
                    "Run position reconciliation to verify exchange state"
                )

            conn.close()

        except Exception as e:
            result.checks_failed.append(f"Database check error: {e}")

    def _check_model(self, result: StartupValidation) -> None:
        """Check ML model file."""
        if not self.model_path.exists():
            result.warnings.append(f"Model file not found: {self.model_path}")
            result.recommendations.append("Train or download a model before starting")
            return

        # Check model age
        mtime = datetime.fromtimestamp(self.model_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days

        if age_days > 30:
            result.warnings.append(f"Model is {age_days} days old")
            result.recommendations.append("Consider retraining the model")
        else:
            result.checks_passed.append("Model file exists and is recent")

    def _check_directories(self, result: StartupValidation) -> None:
        """Check required directories."""
        required_dirs = ["data", "logs", "data/memory", "data/backups"]

        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    result.checks_passed.append(f"Created directory: {dir_path}")
                except Exception as e:
                    result.checks_failed.append(f"Failed to create {dir_path}: {e}")
            else:
                result.checks_passed.append(f"Directory exists: {dir_path}")

    def _check_api_connectivity(self, result: StartupValidation) -> None:
        """Check API connectivity."""
        try:
            # Try to get ticker (public endpoint)
            ticker = self.client.get_ticker("BTC_JPY")
            if ticker:
                result.checks_passed.append("GMO API connection OK")
            else:
                result.checks_failed.append("GMO API returned no data")
        except Exception as e:
            result.checks_failed.append(f"GMO API connection failed: {e}")
            result.recommendations.append("Check network connectivity and API credentials")

    def _check_crash_recovery(self, result: StartupValidation) -> None:
        """Check if crash recovery is needed."""
        # Check for incomplete operations (lock files, temp files)
        crash_indicators = [
            Path("data/.trading_bot.crash"),
            Path("data/.retraining_in_progress"),
        ]

        for indicator in crash_indicators:
            if indicator.exists():
                result.warnings.append(f"Crash indicator found: {indicator}")
                result.recommendations.append(
                    "Previous session may have crashed - check logs"
                )
                # Clean up indicator
                try:
                    indicator.unlink()
                except Exception:
                    pass


def enable_database_wal_mode(db_path: str) -> bool:
    """
    Enable WAL (Write-Ahead Logging) mode for SQLite database.

    WAL mode improves:
    - Concurrent read/write performance
    - Crash recovery
    - Database durability

    Args:
        db_path: Path to SQLite database

    Returns:
        True if WAL mode enabled successfully
    """
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Enable WAL mode
        cursor.execute("PRAGMA journal_mode=WAL")
        result = cursor.fetchone()[0]

        # Set synchronous mode to NORMAL (good balance of safety and speed)
        cursor.execute("PRAGMA synchronous=NORMAL")

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")

        conn.close()

        if result.lower() == "wal":
            logger.info(f"WAL mode enabled for {db_path}")
            return True
        else:
            logger.warning(f"Failed to enable WAL mode, got: {result}")
            return False

    except Exception as e:
        logger.error(f"Failed to enable WAL mode: {e}")
        return False


def check_database_integrity(db_path: str) -> tuple[bool, str]:
    """
    Run SQLite integrity check on database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Tuple of (is_ok, message)
    """
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]

        conn.close()

        if result == "ok":
            return True, "Database integrity check passed"
        else:
            return False, f"Database integrity check failed: {result}"

    except Exception as e:
        return False, f"Database integrity check error: {e}"


def create_crash_indicator(indicator_type: str = "crash") -> Path:
    """
    Create a crash indicator file.

    Args:
        indicator_type: Type of indicator (crash, retraining, etc.)

    Returns:
        Path to indicator file
    """
    indicator_path = Path(f"data/.trading_bot.{indicator_type}")
    indicator_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        indicator_path.write_text(
            f"{datetime.utcnow().isoformat()}\n{os.getpid()}\n"
        )
    except IOError as e:
        logger.warning(f"Failed to create crash indicator: {e}")

    return indicator_path


def remove_crash_indicator(indicator_type: str = "crash") -> None:
    """
    Remove crash indicator file on clean shutdown.

    Args:
        indicator_type: Type of indicator to remove
    """
    indicator_path = Path(f"data/.trading_bot.{indicator_type}")

    try:
        if indicator_path.exists():
            indicator_path.unlink()
    except IOError as e:
        logger.warning(f"Failed to remove crash indicator: {e}")
