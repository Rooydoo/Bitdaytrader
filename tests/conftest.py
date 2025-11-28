"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_capital():
    """Standard capital for testing."""
    return 1_000_000  # 1M JPY


@pytest.fixture
def sample_btc_price():
    """Standard BTC price for testing."""
    return 10_000_000  # 10M JPY


@pytest.fixture
def sample_atr():
    """Standard ATR value for testing."""
    return 200_000  # 200K JPY
