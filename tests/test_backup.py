"""Tests for the BackupManager class."""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.backup.manager import BackupManager, BackupConfig, BackupResult


class TestBackupConfig:
    """Tests for BackupConfig class."""

    def test_default_values(self):
        """Test default backup configuration."""
        config = BackupConfig()

        assert config.backup_dir == "backups"
        assert config.include_database is True
        assert config.include_models is True
        assert config.include_config is True
        assert config.include_logs is False
        assert config.auto_backup_enabled is True
        assert config.backup_interval_hours == 6
        assert config.max_backups == 10
        assert config.compress_backups is True

    def test_custom_values(self):
        """Test custom backup configuration."""
        config = BackupConfig(
            backup_dir="custom_backups",
            max_backups=5,
            backup_interval_hours=12,
            compress_backups=False,
        )

        assert config.backup_dir == "custom_backups"
        assert config.max_backups == 5
        assert config.backup_interval_hours == 12
        assert config.compress_backups is False


class TestBackupManager:
    """Tests for BackupManager class."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory with test files."""
        temp_dir = tempfile.mkdtemp()
        project_root = Path(temp_dir)

        # Create test structure
        (project_root / "data").mkdir()
        (project_root / "models").mkdir()
        (project_root / "logs").mkdir()

        # Create test files
        db_path = project_root / "data" / "trading.db"
        db_path.write_text("test database content")

        runtime_settings = project_root / "data" / "runtime_settings.json"
        runtime_settings.write_text(json.dumps({"test": "settings"}))

        model_file = project_root / "models" / "lightgbm_model.joblib"
        model_file.write_text("test model content")

        env_file = project_root / ".env"
        env_file.write_text("TEST_VAR=value")

        yield project_root

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def backup_manager(self, temp_project):
        """Create a BackupManager with test configuration."""
        config = BackupConfig(
            backup_dir="backups",
            database_path="data/trading.db",
            models_dir="models",
            config_files=[
                ".env",
                "data/runtime_settings.json",
            ],
            max_backups=3,
            compress_backups=True,
        )
        return BackupManager(config=config, project_root=temp_project)

    def test_initialization(self, backup_manager, temp_project):
        """Test BackupManager initialization."""
        assert backup_manager.project_root == temp_project
        assert backup_manager.backup_dir.exists()
        assert backup_manager.config.max_backups == 3

    def test_create_backup_compressed(self, backup_manager):
        """Test creating a compressed backup."""
        result = backup_manager.create_backup("test")

        assert result.success is True
        assert result.backup_path is not None
        assert result.backup_path.exists()
        assert result.backup_path.suffix == ".gz"
        assert len(result.files_backed_up) > 0
        assert result.size_bytes > 0
        assert result.duration_seconds >= 0

    def test_create_backup_uncompressed(self, temp_project):
        """Test creating an uncompressed backup."""
        config = BackupConfig(
            compress_backups=False,
            database_path="data/trading.db",
        )
        manager = BackupManager(config=config, project_root=temp_project)
        result = manager.create_backup("uncompressed")

        assert result.success is True
        assert result.backup_path is not None
        assert result.backup_path.is_dir()

    def test_list_backups(self, backup_manager):
        """Test listing backups."""
        # Create a few backups
        backup_manager.create_backup("backup1")
        backup_manager.create_backup("backup2")

        backups = backup_manager.list_backups()

        assert len(backups) >= 2
        assert all("name" in b for b in backups)
        assert all("size_bytes" in b for b in backups)
        assert all("created_at" in b for b in backups)

        # Should be sorted newest first
        if len(backups) >= 2:
            assert backups[0]["created_at"] >= backups[1]["created_at"]

    def test_backup_rotation(self, backup_manager):
        """Test that old backups are cleaned up."""
        # Create more backups than max_backups
        for i in range(5):
            backup_manager.create_backup(f"rotation_{i}")

        backups = backup_manager.list_backups()

        # Should only have max_backups (3) backups
        assert len(backups) <= backup_manager.config.max_backups

    def test_backup_stats(self, backup_manager):
        """Test getting backup statistics."""
        backup_manager.create_backup("stats_test")

        stats = backup_manager.get_backup_stats()

        assert "backup_count" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "newest_backup" in stats
        assert "auto_backup_enabled" in stats
        assert stats["backup_count"] >= 1

    def test_restore_backup(self, backup_manager, temp_project):
        """Test restoring a backup."""
        # Modify original files
        db_path = temp_project / "data" / "trading.db"
        original_content = db_path.read_text()

        # Create backup
        result = backup_manager.create_backup("restore_test")
        assert result.success

        # Modify the file
        db_path.write_text("modified content")
        assert db_path.read_text() != original_content

        # Restore backup
        backup_name = result.backup_path.name
        restore_success = backup_manager.restore_backup(backup_name)

        assert restore_success is True
        assert db_path.read_text() == original_content

    def test_restore_nonexistent_backup(self, backup_manager):
        """Test restoring a backup that doesn't exist."""
        success = backup_manager.restore_backup("nonexistent_backup")
        assert success is False

    def test_backup_with_missing_files(self, temp_project):
        """Test backup handles missing optional files gracefully."""
        # Create config with files that don't exist
        config = BackupConfig(
            database_path="data/nonexistent.db",
            config_files=["nonexistent.env"],
        )
        manager = BackupManager(config=config, project_root=temp_project)

        result = manager.create_backup("missing_files")

        # Should still succeed, but may include model file if present
        assert result.success is True
        # Missing db and config files, but models may still be backed up
        for f in result.files_backed_up:
            assert "nonexistent" not in f

    def test_backup_hooks(self, temp_project):
        """Test pre and post backup hooks."""
        hook_calls = []

        def pre_hook():
            hook_calls.append("pre")

        def post_hook():
            hook_calls.append("post")

        config = BackupConfig(
            pre_backup_hook=pre_hook,
            post_backup_hook=post_hook,
        )
        manager = BackupManager(config=config, project_root=temp_project)

        manager.create_backup("hooks_test")

        assert "pre" in hook_calls
        assert "post" in hook_calls
        assert hook_calls.index("pre") < hook_calls.index("post")


class TestBackupResult:
    """Tests for BackupResult class."""

    def test_success_result(self):
        """Test successful backup result."""
        result = BackupResult(
            success=True,
            backup_path=Path("/backups/test.tar.gz"),
            timestamp=datetime.now(),
            size_bytes=1024,
            files_backed_up=["file1.txt", "file2.txt"],
        )

        assert result.success is True
        assert result.error_message is None
        assert len(result.files_backed_up) == 2

    def test_failure_result(self):
        """Test failed backup result."""
        result = BackupResult(
            success=False,
            backup_path=None,
            timestamp=datetime.now(),
            size_bytes=0,
            files_backed_up=[],
            error_message="Permission denied",
        )

        assert result.success is False
        assert result.error_message == "Permission denied"
        assert result.backup_path is None


class TestBackupManagerAsync:
    """Tests for async backup functionality."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        project_root = Path(temp_dir)
        (project_root / "data").mkdir()

        db_path = project_root / "data" / "trading.db"
        db_path.write_text("test")

        yield project_root
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_start_stop_scheduled_backups(self, temp_project):
        """Test starting and stopping scheduled backups."""
        import asyncio

        config = BackupConfig(
            auto_backup_enabled=True,
            backup_interval_hours=1,  # Short for testing
        )
        manager = BackupManager(config=config, project_root=temp_project)

        # Start
        await manager.start_scheduled_backups()
        assert manager._running is True
        assert manager._backup_task is not None

        # Wait a tiny bit
        await asyncio.sleep(0.1)

        # Stop
        await manager.stop_scheduled_backups()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_disabled_auto_backup(self, temp_project):
        """Test that disabled auto backup doesn't start task."""
        config = BackupConfig(auto_backup_enabled=False)
        manager = BackupManager(config=config, project_root=temp_project)

        await manager.start_scheduled_backups()

        assert manager._backup_task is None
