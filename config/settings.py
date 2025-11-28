"""Application settings using Pydantic."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GMO Coin API
    gmo_api_key: str = Field(default="", description="GMO Coin API Key")
    gmo_api_secret: str = Field(default="", description="GMO Coin API Secret")
    gmo_base_url: str = Field(
        default="https://api.coin.z.com/public",
        description="GMO Coin Public API URL",
    )
    gmo_private_url: str = Field(
        default="https://api.coin.z.com/private",
        description="GMO Coin Private API URL",
    )

    # Telegram Bot
    telegram_bot_token: str = Field(default="", description="Telegram Bot Token")
    telegram_chat_id: str = Field(default="", description="Telegram Chat ID")

    # Trading Settings
    symbol: str = Field(default="BTC_JPY", description="Primary trading symbol (legacy)")
    timeframe: str = Field(default="15min", description="Candle timeframe")
    prediction_horizon: int = Field(default=4, description="Prediction horizon (periods)")

    # Portfolio Allocation - Multi-asset support
    # Format: "SYMBOL:allocation_pct,SYMBOL:allocation_pct" (must sum to 1.0 or less)
    symbols_config: str = Field(
        default="BTC_JPY:0.50,ETH_JPY:0.30,XRP_JPY:0.20",
        description="Symbols with allocation percentages (e.g., BTC_JPY:0.50,ETH_JPY:0.30)"
    )

    # Capital Utilization - How much of total capital to actually use
    total_capital_utilization: float = Field(
        default=0.80,
        description="Total capital to use for trading (80% = keep 20% cash reserve)"
    )

    # LONG/SHORT Allocation - How to split between long and short positions
    long_allocation_ratio: float = Field(
        default=0.60,
        description="Portion of capital allocated to LONG positions (60%)"
    )
    short_allocation_ratio: float = Field(
        default=0.40,
        description="Portion of capital allocated to SHORT positions (40%)"
    )

    # Leverage Settings (GMO Coin margin trading)
    use_leverage: bool = Field(default=True, description="Use leverage/margin trading")
    leverage: float = Field(default=1.2, description="Leverage ratio (1.0-2.0)")
    margin_call_threshold: float = Field(default=0.75, description="Margin maintenance ratio")

    # Risk Management - LONG positions (default/less risky)
    long_risk_per_trade: float = Field(default=0.02, description="Risk per trade for LONG (2%)")
    long_max_position_size: float = Field(default=0.10, description="Max position size for LONG (10%)")
    long_max_daily_trades: int = Field(default=3, description="Max LONG trades per day")
    long_confidence_threshold: float = Field(default=0.65, description="Min confidence for LONG")
    long_sl_atr_multiple: float = Field(default=2.0, description="Stop loss ATR multiple for LONG")

    # Risk Management - SHORT positions (stricter/higher risk)
    short_risk_per_trade: float = Field(default=0.015, description="Risk per trade for SHORT (1.5%)")
    short_max_position_size: float = Field(default=0.07, description="Max position size for SHORT (7%)")
    short_max_daily_trades: int = Field(default=2, description="Max SHORT trades per day")
    short_confidence_threshold: float = Field(default=0.70, description="Min confidence for SHORT (stricter)")
    short_sl_atr_multiple: float = Field(default=1.5, description="Stop loss ATR multiple for SHORT (tighter)")

    # Global Risk Limits
    daily_loss_limit: float = Field(default=0.03, description="Daily loss limit (3%)")
    max_daily_trades: int = Field(default=5, description="Max total trades per day")

    # Model Settings
    model_path: str = Field(default="models/lightgbm_model.joblib", description="Model file path")
    training_data_path: str = Field(default="data/ohlcv_history.csv", description="Training data CSV path")

    # Database
    db_path: str = Field(default="data/trading.db", description="SQLite database path")

    # Training Settings
    training_months: int = Field(default=12, description="Training data months")
    backtest_months: int = Field(default=2, description="Backtest data months")
    walkforward_interval_days: int = Field(default=14, description="Walk-forward interval")

    # Report Schedule (JST hours)
    report_morning_hour: int = Field(default=8, description="Morning report hour")
    report_noon_hour: int = Field(default=12, description="Noon report hour")
    report_evening_hour: int = Field(default=20, description="Evening report hour")
    report_weekly_day: int = Field(default=0, description="Weekly report day (0=Monday)")
    report_monthly_day: int = Field(default=1, description="Monthly report day")

    # Execution Mode
    mode: Literal["live", "paper", "backtest"] = Field(
        default="paper", description="Execution mode"
    )

    # Take Profit Levels - LONG (R multiples, can let profits run)
    long_tp_level_1: float = Field(default=1.5, description="First TP level for LONG (R)")
    long_tp_ratio_1: float = Field(default=0.5, description="First TP ratio for LONG")
    long_tp_level_2: float = Field(default=2.5, description="Second TP level for LONG (R)")
    long_tp_ratio_2: float = Field(default=0.3, description="Second TP ratio for LONG")
    long_tp_level_3: float = Field(default=4.0, description="Third TP level for LONG (R)")
    long_tp_ratio_3: float = Field(default=0.2, description="Third TP ratio for LONG")

    # Take Profit Levels - SHORT (R multiples, take profits faster)
    short_tp_level_1: float = Field(default=1.0, description="First TP level for SHORT (R)")
    short_tp_ratio_1: float = Field(default=0.5, description="First TP ratio for SHORT")
    short_tp_level_2: float = Field(default=1.5, description="Second TP level for SHORT (R)")
    short_tp_ratio_2: float = Field(default=0.3, description="Second TP ratio for SHORT")
    short_tp_level_3: float = Field(default=2.5, description="Third TP level for SHORT (R)")
    short_tp_ratio_3: float = Field(default=0.2, description="Third TP ratio for SHORT")


    def get_symbol_allocations(self) -> dict[str, float]:
        """
        Parse symbols_config and return dict of symbol -> allocation percentage.

        Example: "BTC_JPY:0.50,ETH_JPY:0.30" -> {"BTC_JPY": 0.50, "ETH_JPY": 0.30}
        """
        allocations = {}
        for item in self.symbols_config.split(","):
            item = item.strip()
            if ":" in item:
                symbol, alloc = item.split(":")
                allocations[symbol.strip()] = float(alloc.strip())
        return allocations

    def get_capital_for_symbol(self, symbol: str, total_capital: float) -> float:
        """
        Calculate capital allocated to a specific symbol.

        Args:
            symbol: Trading symbol
            total_capital: Total available capital

        Returns:
            Capital allocated to this symbol
        """
        allocations = self.get_symbol_allocations()
        symbol_pct = allocations.get(symbol, 0.0)
        return total_capital * self.total_capital_utilization * symbol_pct

    def get_capital_for_direction(
        self, symbol: str, direction: str, total_capital: float
    ) -> float:
        """
        Calculate capital allocated for a specific symbol and direction.

        Args:
            symbol: Trading symbol
            direction: "LONG" or "SHORT"
            total_capital: Total available capital

        Returns:
            Capital allocated for this symbol and direction
        """
        symbol_capital = self.get_capital_for_symbol(symbol, total_capital)
        if direction == "LONG":
            return symbol_capital * self.long_allocation_ratio
        else:  # SHORT
            return symbol_capital * self.short_allocation_ratio


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
