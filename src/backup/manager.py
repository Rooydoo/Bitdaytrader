"""Backup manager for automated data backup."""

import asyncio
import gzip
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger


@dataclass
class BackupConfig:
    """Configuration for backup behavior."""

    # Backup directory
    backup_dir: str = "backups"

    # Files and directories to backup
    include_database: bool = True
    include_models: bool = True
    include_config: bool = True
    include_logs: bool = False

    # Specific paths (relative to project root)
    database_path: str = "data/trading.db"
    models_dir: str = "models"
    config_files: list[str] = field(default_factory=lambda: [
        ".env",
        "data/runtime_settings.json",
        "data/walkforward_results.json",
    ])
    logs_dir: str = "logs"

    # Backup schedule
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 6  # Backup every 6 hours

    # Retention settings
    max_backups: int = 10  # Keep last 10 backups
    compress_backups: bool = True

    # Pre/post backup hooks
    pre_backup_hook: Callable | None = None
    post_backup_hook: Callable | None = None


@dataclass
class BackupResult:
    """Result of a backup operation."""

    success: bool
    backup_path: Path | None
    timestamp: datetime
    size_bytes: int
    files_backed_up: list[str]
    error_message: str | None = None
    duration_seconds: float = 0.0


class BackupManager:
    """Manages automated backups of trading data."""

    def __init__(
        self,
        config: BackupConfig | None = None,
        project_root: Path | None = None,
    ) -> None:
        """
        Initialize the backup manager.

        Args:
            config: Backup configuration
            project_root: Project root directory (auto-detected if None)
        """
        self.config = config or BackupConfig()

        # Detect project root
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root

        self.backup_dir = self.project_root / self.config.backup_dir

        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Async task for scheduled backups
        self._backup_task: asyncio.Task | None = None
        self._running = False

        logger.info(f"BackupManager initialized. Backup dir: {self.backup_dir}")

    def create_backup(self, name_suffix: str = "") -> BackupResult:
        """
        Create a backup of configured files.

        Args:
            name_suffix: Optional suffix for backup name

        Returns:
            BackupResult with details of the backup
        """
        start_time = datetime.now()
        files_backed_up = []
        error_message = None

        # Generate backup name
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        if name_suffix:
            backup_name += f"_{name_suffix}"

        # Create temp directory for this backup
        temp_backup_dir = self.backup_dir / f"temp_{backup_name}"
        temp_backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run pre-backup hook
            if self.config.pre_backup_hook:
                try:
                    self.config.pre_backup_hook()
                except Exception as e:
                    logger.warning(f"Pre-backup hook failed: {e}")

            # Backup database
            if self.config.include_database:
                db_path = self.project_root / self.config.database_path
                if db_path.exists():
                    dest = temp_backup_dir / "database" / db_path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(db_path, dest)
                    files_backed_up.append(str(db_path.relative_to(self.project_root)))
                    logger.debug(f"Backed up database: {db_path}")

            # Backup models
            if self.config.include_models:
                models_dir = self.project_root / self.config.models_dir
                if models_dir.exists():
                    dest = temp_backup_dir / "models"
                    dest.mkdir(parents=True, exist_ok=True)
                    for model_file in models_dir.glob("*"):
                        if model_file.is_file():
                            shutil.copy2(model_file, dest / model_file.name)
                            files_backed_up.append(str(model_file.relative_to(self.project_root)))
                    logger.debug(f"Backed up models: {models_dir}")

            # Backup config files
            if self.config.include_config:
                config_dest = temp_backup_dir / "config"
                config_dest.mkdir(parents=True, exist_ok=True)
                for config_file in self.config.config_files:
                    src = self.project_root / config_file
                    if src.exists():
                        # Preserve directory structure
                        rel_path = Path(config_file)
                        dest_file = config_dest / rel_path.name
                        shutil.copy2(src, dest_file)
                        files_backed_up.append(config_file)
                        logger.debug(f"Backed up config: {src}")

            # Backup logs (optional)
            if self.config.include_logs:
                logs_dir = self.project_root / self.config.logs_dir
                if logs_dir.exists():
                    dest = temp_backup_dir / "logs"
                    shutil.copytree(logs_dir, dest, dirs_exist_ok=True)
                    for log_file in logs_dir.glob("**/*"):
                        if log_file.is_file():
                            files_backed_up.append(str(log_file.relative_to(self.project_root)))

            # Create metadata file
            metadata = {
                "timestamp": start_time.isoformat(),
                "files_count": len(files_backed_up),
                "files": files_backed_up,
            }
            import json
            with open(temp_backup_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Create archive
            if self.config.compress_backups:
                archive_path = self.backup_dir / f"{backup_name}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(temp_backup_dir, arcname=backup_name)
                backup_path = archive_path
            else:
                # Just move the directory
                final_path = self.backup_dir / backup_name
                shutil.move(temp_backup_dir, final_path)
                backup_path = final_path

            # Get size
            if backup_path.is_file():
                size_bytes = backup_path.stat().st_size
            else:
                size_bytes = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())

            # Clean up old backups
            self._cleanup_old_backups()

            # Run post-backup hook
            if self.config.post_backup_hook:
                try:
                    self.config.post_backup_hook()
                except Exception as e:
                    logger.warning(f"Post-backup hook failed: {e}")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Backup completed: {backup_path.name}, "
                f"{len(files_backed_up)} files, "
                f"{size_bytes / 1024 / 1024:.2f} MB, "
                f"{duration:.1f}s"
            )

            return BackupResult(
                success=True,
                backup_path=backup_path,
                timestamp=start_time,
                size_bytes=size_bytes,
                files_backed_up=files_backed_up,
                duration_seconds=duration,
            )

        except Exception as e:
            error_message = str(e)
            logger.error(f"Backup failed: {e}")
            return BackupResult(
                success=False,
                backup_path=None,
                timestamp=start_time,
                size_bytes=0,
                files_backed_up=files_backed_up,
                error_message=error_message,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        finally:
            # Clean up temp directory
            if temp_backup_dir.exists():
                try:
                    shutil.rmtree(temp_backup_dir)
                except Exception:
                    pass

    def _cleanup_old_backups(self) -> int:
        """
        Remove old backups exceeding max_backups limit.

        Returns:
            Number of backups removed
        """
        # Get all backups
        backups = []
        for item in self.backup_dir.iterdir():
            if item.name.startswith("backup_"):
                backups.append(item)

        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove excess backups
        removed = 0
        for backup in backups[self.config.max_backups:]:
            try:
                if backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup)
                removed += 1
                logger.debug(f"Removed old backup: {backup.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {backup.name}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old backups")

        return removed

    def list_backups(self) -> list[dict]:
        """
        List all available backups.

        Returns:
            List of backup info dictionaries
        """
        backups = []
        for item in self.backup_dir.iterdir():
            if item.name.startswith("backup_"):
                try:
                    stat = item.stat()
                    backups.append({
                        "name": item.name,
                        "path": str(item),
                        "size_bytes": stat.st_size if item.is_file() else sum(
                            f.stat().st_size for f in item.rglob("*") if f.is_file()
                        ),
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_compressed": item.suffix in [".gz", ".tar.gz"],
                    })
                except Exception as e:
                    logger.warning(f"Failed to get backup info for {item.name}: {e}")

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def restore_backup(self, backup_name: str, restore_dir: Path | None = None) -> bool:
        """
        Restore a backup.

        Args:
            backup_name: Name of the backup to restore
            restore_dir: Directory to restore to (default: project root)

        Returns:
            True if restore was successful
        """
        if restore_dir is None:
            restore_dir = self.project_root

        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            # Try with .tar.gz extension
            backup_path = self.backup_dir / f"{backup_name}.tar.gz"
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_name}")
                return False

        try:
            if backup_path.suffix == ".gz" or backup_path.name.endswith(".tar.gz"):
                # Extract compressed backup
                with tarfile.open(backup_path, "r:gz") as tar:
                    # Get the root folder name in the archive
                    members = tar.getmembers()
                    if not members:
                        logger.error("Empty backup archive")
                        return False

                    root_name = members[0].name.split("/")[0]
                    temp_extract = self.backup_dir / f"temp_restore_{datetime.now().timestamp()}"
                    tar.extractall(temp_extract)

                    extracted_dir = temp_extract / root_name
                    self._restore_from_dir(extracted_dir, restore_dir)

                    # Cleanup
                    shutil.rmtree(temp_extract)
            else:
                # Restore from uncompressed directory
                self._restore_from_dir(backup_path, restore_dir)

            logger.info(f"Backup restored successfully: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def _restore_from_dir(self, backup_dir: Path, restore_dir: Path) -> None:
        """Restore files from a backup directory."""
        # Restore database
        db_backup = backup_dir / "database"
        if db_backup.exists():
            for db_file in db_backup.iterdir():
                dest = restore_dir / self.config.database_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db_file, dest)
                logger.debug(f"Restored database: {dest}")

        # Restore models
        models_backup = backup_dir / "models"
        if models_backup.exists():
            dest_dir = restore_dir / self.config.models_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            for model_file in models_backup.iterdir():
                shutil.copy2(model_file, dest_dir / model_file.name)
                logger.debug(f"Restored model: {model_file.name}")

        # Restore config files
        config_backup = backup_dir / "config"
        if config_backup.exists():
            for config_file in config_backup.iterdir():
                # Find original path
                for original_path in self.config.config_files:
                    if Path(original_path).name == config_file.name:
                        dest = restore_dir / original_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(config_file, dest)
                        logger.debug(f"Restored config: {dest}")
                        break

    async def start_scheduled_backups(self) -> None:
        """Start the scheduled backup task."""
        if not self.config.auto_backup_enabled:
            logger.info("Auto backup is disabled")
            return

        if self._backup_task is not None and not self._backup_task.done():
            logger.warning("Scheduled backups already running")
            return

        self._running = True
        self._backup_task = asyncio.create_task(self._backup_loop())
        logger.info(
            f"Started scheduled backups (every {self.config.backup_interval_hours} hours)"
        )

    async def stop_scheduled_backups(self) -> None:
        """Stop the scheduled backup task."""
        self._running = False
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
        logger.info("Stopped scheduled backups")

    async def _backup_loop(self) -> None:
        """Internal loop for scheduled backups."""
        interval_seconds = self.config.backup_interval_hours * 3600

        # Run initial backup
        await asyncio.get_event_loop().run_in_executor(
            None, self.create_backup, "scheduled"
        )

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                if self._running:
                    # Run backup in thread pool to avoid blocking
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.create_backup, "scheduled"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled backup error: {e}")

    def get_backup_stats(self) -> dict:
        """
        Get backup statistics.

        Returns:
            Dict with backup statistics
        """
        backups = self.list_backups()
        total_size = sum(b["size_bytes"] for b in backups)

        return {
            "backup_count": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "newest_backup": backups[0] if backups else None,
            "oldest_backup": backups[-1] if backups else None,
            "auto_backup_enabled": self.config.auto_backup_enabled,
            "backup_interval_hours": self.config.backup_interval_hours,
            "max_backups": self.config.max_backups,
        }
