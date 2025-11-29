"""Tests for the RiskManager class."""

import pytest
from datetime import date

from src.risk.manager import (
    RiskManager,
    RiskCheck,
    DrawdownState,
    LosingStreakConfig,
    DirectionConfig,
    PositionSize,
)


class TestRiskManagerBasics:
    """Basic RiskManager functionality tests."""

    def test_initialization_defaults(self):
        """Test RiskManager initializes with correct defaults."""
        rm = RiskManager()

        # Check global limits
        assert rm.daily_loss_limit == 0.03
        assert rm.weekly_loss_limit == 0.07
        assert rm.monthly_loss_limit == 0.12
        assert rm.max_drawdown_limit == 0.15
        assert rm.max_daily_trades == 5
        assert rm.max_consecutive_losses == 5

        # Check LONG config (more aggressive)
        assert rm.long_config.risk_per_trade == 0.02
        assert rm.long_config.max_position_size == 0.10
        assert rm.long_config.confidence_threshold == 0.65

        # Check SHORT config (more conservative)
        assert rm.short_config.risk_per_trade == 0.015
        assert rm.short_config.max_position_size == 0.07
        assert rm.short_config.confidence_threshold == 0.70

    def test_get_config_by_side(self):
        """Test getting correct config by side."""
        rm = RiskManager()

        long_config = rm.get_config("BUY")
        short_config = rm.get_config("SELL")

        assert long_config == rm.long_config
        assert short_config == rm.short_config


class TestDailyLimits:
    """Test daily loss and trade limits."""

    def test_daily_loss_limit_blocks_trading(self):
        """Test that daily loss limit blocks new trades."""
        rm = RiskManager(daily_loss_limit=0.03)
        capital = 1_000_000

        # Simulate 3% loss
        rm.add_trade_result(-30_000, "BUY", capital - 30_000)

        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Daily loss limit" in result.reason

    def test_max_daily_trades_blocks_trading(self):
        """Test that max daily trades blocks new trades."""
        rm = RiskManager(max_daily_trades=2)
        capital = 1_000_000

        # Add 2 trades
        rm.add_trade_result(1000, "BUY", capital)
        rm.add_trade_result(1000, "SELL", capital)

        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Max daily trades" in result.reason

    def test_direction_specific_max_trades(self):
        """Test direction-specific trade limits."""
        rm = RiskManager(
            long_max_daily_trades=2,
            short_max_daily_trades=1,
            max_daily_trades=10,  # High global limit
        )
        capital = 1_000_000

        # Add 2 LONG trades
        rm.add_trade_result(1000, "BUY", capital)
        rm.add_trade_result(1000, "BUY", capital)

        # LONG should be blocked
        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Max LONG trades" in result.reason

        # SHORT should still be allowed
        result = rm.check_can_trade(capital, "SELL")
        assert result.allowed


class TestDrawdownLimits:
    """Test weekly, monthly, and max drawdown limits."""

    def test_weekly_loss_limit(self):
        """Test weekly loss limit blocks trading."""
        rm = RiskManager(weekly_loss_limit=0.05)
        capital = 1_000_000

        # Initialize capital tracking
        rm.update_capital(capital)

        # Simulate 5% weekly loss
        for _ in range(5):
            rm.add_trade_result(-10_000, "BUY", capital - 50_000)

        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Weekly loss limit" in result.reason

    def test_monthly_loss_limit(self):
        """Test monthly loss limit blocks trading."""
        rm = RiskManager(monthly_loss_limit=0.10)
        capital = 1_000_000

        rm.update_capital(capital)

        # Simulate 10% monthly loss
        rm.add_trade_result(-100_000, "BUY", capital - 100_000)

        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Monthly loss limit" in result.reason

    def test_max_drawdown_limit(self):
        """Test max drawdown from peak blocks trading."""
        rm = RiskManager(max_drawdown_limit=0.15)
        capital = 1_000_000

        # Set peak capital
        rm.update_capital(capital)

        # Drop by 15%
        rm.update_capital(850_000)

        result = rm.check_can_trade(850_000, "BUY")
        assert not result.allowed
        assert "Max drawdown limit" in result.reason


class TestConsecutiveLosses:
    """Test consecutive loss tracking and limits."""

    def test_consecutive_losses_tracked(self):
        """Test that consecutive losses are tracked correctly."""
        rm = RiskManager()
        capital = 1_000_000

        rm.add_trade_result(-1000, "BUY", capital)
        assert rm._drawdown.consecutive_losses == 1

        rm.add_trade_result(-1000, "BUY", capital)
        assert rm._drawdown.consecutive_losses == 2

        rm.add_trade_result(-1000, "BUY", capital)
        assert rm._drawdown.consecutive_losses == 3

    def test_consecutive_losses_reset_on_win(self):
        """Test that consecutive losses reset on winning trade."""
        rm = RiskManager()
        capital = 1_000_000

        rm.add_trade_result(-1000, "BUY", capital)
        rm.add_trade_result(-1000, "BUY", capital)
        assert rm._drawdown.consecutive_losses == 2

        rm.add_trade_result(1000, "BUY", capital)
        assert rm._drawdown.consecutive_losses == 0
        assert rm._drawdown.consecutive_wins == 1

    def test_max_consecutive_losses_blocks_trading(self):
        """Test max consecutive losses blocks trading."""
        rm = RiskManager(max_consecutive_losses=3)
        capital = 1_000_000

        for _ in range(3):
            rm.add_trade_result(-1000, "BUY", capital)

        result = rm.check_can_trade(capital, "BUY")
        assert not result.allowed
        assert "Max consecutive losses" in result.reason


class TestLosingStreakRiskReduction:
    """Test losing streak risk reduction mechanism."""

    def test_no_reduction_below_threshold(self):
        """Test no risk reduction below threshold."""
        config = LosingStreakConfig(start_reduction_at=3)
        rm = RiskManager(losing_streak_config=config)
        capital = 1_000_000

        # 2 losses - below threshold
        rm.add_trade_result(-1000, "BUY", capital)
        rm.add_trade_result(-1000, "BUY", capital)

        multiplier = rm.get_losing_streak_multiplier()
        assert multiplier == 1.0

    def test_reduction_at_threshold(self):
        """Test risk reduction starts at threshold."""
        config = LosingStreakConfig(
            start_reduction_at=2,
            reduction_per_loss=0.20,
        )
        rm = RiskManager(losing_streak_config=config)
        capital = 1_000_000

        # 2 losses - at threshold
        rm.add_trade_result(-1000, "BUY", capital)
        rm.add_trade_result(-1000, "BUY", capital)

        multiplier = rm.get_losing_streak_multiplier()
        assert multiplier == 0.80  # 1.0 - 0.20

    def test_progressive_reduction(self):
        """Test progressive risk reduction with more losses."""
        config = LosingStreakConfig(
            start_reduction_at=2,
            reduction_per_loss=0.20,
            min_risk_multiplier=0.30,
        )
        rm = RiskManager(losing_streak_config=config)
        capital = 1_000_000

        # 4 losses
        for _ in range(4):
            rm.add_trade_result(-1000, "BUY", capital)

        # 3 losses beyond threshold (2, 3, 4) = 3 * 0.20 = 0.60 reduction
        multiplier = rm.get_losing_streak_multiplier()
        assert multiplier == 0.40  # 1.0 - 0.60

    def test_minimum_multiplier_floor(self):
        """Test minimum multiplier floor is respected."""
        config = LosingStreakConfig(
            start_reduction_at=2,
            reduction_per_loss=0.25,
            min_risk_multiplier=0.30,
        )
        rm = RiskManager(losing_streak_config=config)
        capital = 1_000_000

        # 10 losses - should hit floor
        for _ in range(10):
            rm.add_trade_result(-1000, "BUY", capital)

        multiplier = rm.get_losing_streak_multiplier()
        assert multiplier == 0.30  # Floor

    def test_recovery_after_wins(self):
        """Test risk recovery after consecutive wins."""
        config = LosingStreakConfig(
            start_reduction_at=2,
            reduction_per_loss=0.20,
            wins_to_recover=2,
        )
        rm = RiskManager(losing_streak_config=config)
        capital = 1_000_000

        # 3 losses
        for _ in range(3):
            rm.add_trade_result(-1000, "BUY", capital)

        # Verify reduction
        assert rm.get_losing_streak_multiplier() < 1.0

        # 2 wins to recover
        rm.add_trade_result(1000, "BUY", capital)
        rm.add_trade_result(1000, "BUY", capital)

        # Should be fully recovered
        multiplier = rm.get_losing_streak_multiplier()
        assert multiplier == 1.0


class TestPositionSizing:
    """Test position size calculations."""

    def test_basic_position_size(self):
        """Test basic position size calculation."""
        rm = RiskManager(
            long_risk_per_trade=0.02,
            long_max_position_size=0.10,
            long_sl_atr_multiple=2.0,
        )
        capital = 1_000_000
        entry_price = 10_000_000  # 10M JPY (BTC)
        atr = 200_000  # 200K JPY ATR

        position = rm.calculate_position_size(capital, entry_price, atr, "BUY")

        # Risk amount = 1M * 0.02 = 20K JPY
        # Stop distance = 200K * 2 = 400K JPY
        # Size by risk = 20K / 400K = 0.05 BTC
        assert position.size > 0
        assert position.stop_loss == entry_price - (atr * 2.0)
        assert position.risk_amount > 0

    def test_position_size_respects_max_limit(self):
        """Test position size respects max position size limit."""
        rm = RiskManager(
            long_risk_per_trade=0.10,  # Very high risk
            long_max_position_size=0.05,  # But low max position
        )
        capital = 1_000_000
        entry_price = 10_000_000
        atr = 100_000

        position = rm.calculate_position_size(capital, entry_price, atr, "BUY")

        # Position value should not exceed 5% of capital
        max_value = capital * 0.05
        assert position.position_value <= max_value

    def test_losing_streak_reduces_position_size(self):
        """Test losing streak reduces position size."""
        config = LosingStreakConfig(
            start_reduction_at=2,
            reduction_per_loss=0.50,  # 50% reduction per loss
        )
        rm = RiskManager(
            losing_streak_config=config,
            long_risk_per_trade=0.02,
        )
        capital = 1_000_000
        entry_price = 10_000_000
        atr = 200_000

        # Get normal position size
        normal_position = rm.calculate_position_size(capital, entry_price, atr, "BUY")

        # Add 2 losses
        rm.add_trade_result(-1000, "BUY", capital)
        rm.add_trade_result(-1000, "BUY", capital)

        # Get reduced position size
        reduced_position = rm.calculate_position_size(capital, entry_price, atr, "BUY")

        # Should be 50% of normal
        assert reduced_position.size == pytest.approx(normal_position.size * 0.5, rel=0.01)


class TestConservativeMode:
    """Test conservative mode functionality."""

    def test_enable_conservative_mode(self):
        """Test enabling conservative mode."""
        rm = RiskManager()
        original_risk = rm.long_config.risk_per_trade

        rm.enable_conservative_mode(multiplier=0.5)

        assert rm.is_conservative_mode
        assert rm.long_config.risk_per_trade == original_risk * 0.5

    def test_disable_conservative_mode(self):
        """Test disabling conservative mode restores settings."""
        rm = RiskManager()
        original_risk = rm.long_config.risk_per_trade

        rm.enable_conservative_mode(multiplier=0.5)
        rm.disable_conservative_mode()

        assert not rm.is_conservative_mode
        assert rm.long_config.risk_per_trade == original_risk


class TestConfidenceCheck:
    """Test confidence threshold checks."""

    def test_confidence_below_threshold_rejected(self):
        """Test low confidence is rejected."""
        rm = RiskManager(long_confidence_threshold=0.70)

        result = rm.check_confidence(0.65, "BUY")
        assert not result.allowed
        assert "confidence" in result.reason.lower()

    def test_confidence_above_threshold_accepted(self):
        """Test high confidence is accepted."""
        rm = RiskManager(long_confidence_threshold=0.70)

        result = rm.check_confidence(0.75, "BUY")
        assert result.allowed


class TestDrawdownState:
    """Test DrawdownState dataclass."""

    def test_current_drawdown_calculation(self):
        """Test current drawdown calculation."""
        state = DrawdownState(peak_capital=1_000_000, current_capital=900_000)
        assert state.current_drawdown == 0.10  # 10%

    def test_weekly_drawdown_calculation(self):
        """Test weekly drawdown calculation."""
        state = DrawdownState(
            weekly_start_capital=1_000_000,
            weekly_pnl=-50_000,
        )
        assert state.weekly_drawdown == 0.05  # 5%

    def test_no_drawdown_when_positive(self):
        """Test no drawdown reported when positive."""
        state = DrawdownState(
            weekly_start_capital=1_000_000,
            weekly_pnl=50_000,  # Positive
        )
        assert state.weekly_drawdown == 0.0
