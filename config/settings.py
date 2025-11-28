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
    symbol: str = Field(default="BTC_JPY", description="Trading symbol")
    timeframe: str = Field(default="15min", description="Candle timeframe")
    prediction_horizon: int = Field(default=4, description="Prediction horizon (periods)")

    # Leverage Settings (GMO Coin margin trading)
    use_leverage: bool = Field(default=True, description="Use leverage/margin trading")
    leverage: float = Field(default=1.2, description="Leverage ratio (1.0-2.0)")
    margin_call_threshold: float = Field(default=0.75, description="Margin maintenance ratio")

    # Risk Management
    risk_per_trade: float = Field(default=0.02, description="Risk per trade (2%)")
    daily_loss_limit: float = Field(default=0.03, description="Daily loss limit (3%)")
    max_position_size: float = Field(default=0.10, description="Max position size (10%)")
    max_daily_trades: int = Field(default=5, description="Max trades per day")

    # Model Settings
    confidence_threshold: float = Field(default=0.65, description="Min confidence to trade")
    model_path: str = Field(default="models/lightgbm_model.joblib", description="Model file path")

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

    # Take Profit Levels (R multiples)
    tp_level_1: float = Field(default=1.5, description="First TP level (R)")
    tp_ratio_1: float = Field(default=0.5, description="First TP ratio")
    tp_level_2: float = Field(default=2.5, description="Second TP level (R)")
    tp_ratio_2: float = Field(default=0.3, description="Second TP ratio")
    tp_level_3: float = Field(default=4.0, description="Third TP level (R)")
    tp_ratio_3: float = Field(default=0.2, description="Third TP ratio")

    # Stop Loss (ATR multiple)
    sl_atr_multiple: float = Field(default=2.0, description="Stop loss ATR multiple")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
