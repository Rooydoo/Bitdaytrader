"""Main trading engine."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from src.api.gmo_client import GMOCoinClient
from src.database.models import (
    DailyPnLRepository,
    Signal,
    Trade,
    TradeRepository,
    init_db,
)
from src.execution.order_manager import OrderManager, PositionStatus
from src.features.calculator import FeatureCalculator
from src.models.predictor import Predictor
from src.reports.generator import ReportGenerator
from src.risk.manager import RiskManager
from src.telegram.bot import TelegramBot


class TradingEngine:
    """Main trading engine that orchestrates all components."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize trading engine.

        Args:
            settings: Application settings
        """
        self.settings = settings

        # Initialize database
        self.session = init_db(settings.db_path)
        self.trade_repo = TradeRepository(self.session)
        self.daily_pnl_repo = DailyPnLRepository(self.session)

        # Initialize API client
        self.client = GMOCoinClient(
            api_key=settings.gmo_api_key,
            api_secret=settings.gmo_api_secret,
            base_url=settings.gmo_base_url,
            private_url=settings.gmo_private_url,
        )

        # Initialize components
        self.feature_calc = FeatureCalculator()
        self.predictor = Predictor(settings.model_path)
        self.risk_manager = RiskManager(
            # LONG settings
            long_risk_per_trade=settings.long_risk_per_trade,
            long_max_position_size=settings.long_max_position_size,
            long_max_daily_trades=settings.long_max_daily_trades,
            long_confidence_threshold=settings.long_confidence_threshold,
            long_sl_atr_multiple=settings.long_sl_atr_multiple,
            long_tp_levels=[
                (settings.long_tp_level_1, settings.long_tp_ratio_1),
                (settings.long_tp_level_2, settings.long_tp_ratio_2),
                (settings.long_tp_level_3, settings.long_tp_ratio_3),
            ],
            # SHORT settings
            short_risk_per_trade=settings.short_risk_per_trade,
            short_max_position_size=settings.short_max_position_size,
            short_max_daily_trades=settings.short_max_daily_trades,
            short_confidence_threshold=settings.short_confidence_threshold,
            short_sl_atr_multiple=settings.short_sl_atr_multiple,
            short_tp_levels=[
                (settings.short_tp_level_1, settings.short_tp_ratio_1),
                (settings.short_tp_level_2, settings.short_tp_ratio_2),
                (settings.short_tp_level_3, settings.short_tp_ratio_3),
            ],
            # Global settings
            daily_loss_limit=settings.daily_loss_limit,
            max_daily_trades=settings.max_daily_trades,
        )
        self.order_manager = OrderManager(self.client, settings.symbol)

        # Initialize Telegram bot
        self.telegram = TelegramBot(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.report_generator = ReportGenerator(
            trade_repo=self.trade_repo,
            daily_pnl_repo=self.daily_pnl_repo,
            telegram_bot=self.telegram,
        )

        # Cache for OHLCV data
        self._ohlcv_cache: pd.DataFrame | None = None
        self._last_fetch_time: datetime | None = None

    async def run_cycle(self) -> None:
        """Run one trading cycle (called every 15 minutes)."""
        logger.info("Starting trading cycle")

        try:
            # 1. Fetch latest data
            df = await self._fetch_ohlcv_data()

            if df.empty:
                logger.warning("No OHLCV data available")
                return

            # 2. Check existing position
            if self.order_manager.has_position():
                await self._manage_existing_position(df)
            else:
                await self._check_new_signal(df)

            logger.info("Trading cycle completed")

        except Exception as e:
            logger.exception(f"Error in trading cycle: {e}")
            await self.telegram.notify_error(str(e), "trading_cycle")

    async def _fetch_ohlcv_data(self) -> pd.DataFrame:
        """Fetch and cache OHLCV data."""
        now = datetime.now()

        # Use cache if recent
        if (
            self._ohlcv_cache is not None
            and self._last_fetch_time is not None
            and now - self._last_fetch_time < timedelta(minutes=5)
        ):
            return self._ohlcv_cache

        # Fetch new data
        try:
            # Get today's and yesterday's data
            today = now.strftime("%Y%m%d")
            yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

            df_today = self.client.get_klines(
                symbol=self.settings.symbol,
                interval=self.settings.timeframe,
                date=today,
            )

            df_yesterday = self.client.get_klines(
                symbol=self.settings.symbol,
                interval=self.settings.timeframe,
                date=yesterday,
            )

            # Combine and sort
            df = pd.concat([df_yesterday, df_today]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp").reset_index(drop=True)

            self._ohlcv_cache = df
            self._last_fetch_time = now

            logger.debug(f"Fetched {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return pd.DataFrame()

    async def _check_new_signal(self, df: pd.DataFrame) -> None:
        """Check for new trading signal (long or short)."""
        # Get capital
        capital = self._get_capital()

        # Apply leverage to effective capital
        if self.settings.use_leverage:
            effective_capital = capital * self.settings.leverage
            logger.debug(f"Leverage {self.settings.leverage}x: {capital:.0f} -> {effective_capital:.0f}")
        else:
            effective_capital = capital

        # Calculate features
        features = self.feature_calc.get_latest_features(df)
        if features is None:
            logger.warning("Could not calculate features")
            return

        # Make prediction
        if not self.predictor.is_loaded():
            logger.warning("Model not loaded")
            return

        prediction, confidence = self.predictor.predict(features)

        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

        # Record signal
        self._record_signal(prediction, confidence, df["close"].iloc[-1], features)

        # Determine trade direction
        # prediction == 1: price will go UP -> LONG (BUY)
        # prediction == 0: price will go DOWN -> SHORT (SELL)
        if prediction == 1:
            side = "BUY"
            direction = "LONG"
        else:
            side = "SELL"
            direction = "SHORT"

        # Check direction-specific risk limits
        risk_check = self.risk_manager.check_can_trade(capital, side)
        if not risk_check.allowed:
            logger.info(f"{direction} trading not allowed: {risk_check.reason}")
            return

        # Check direction-specific confidence threshold
        confidence_check = self.risk_manager.check_confidence(confidence, side)
        if not confidence_check.allowed:
            logger.info(confidence_check.reason)
            return

        logger.info(f"Signal: {direction} with confidence {confidence:.4f}")

        # Execute trade
        await self._execute_entry(df, confidence, effective_capital, side)

    async def _execute_entry(
        self,
        df: pd.DataFrame,
        confidence: float,
        capital: float,
        side: str,
    ) -> None:
        """
        Execute entry trade (long or short).

        Args:
            df: OHLCV data
            confidence: Prediction confidence
            capital: Available capital (with leverage applied)
            side: "BUY" for long, "SELL" for short
        """
        # Get current price and ATR
        current_price = df["close"].iloc[-1]
        atr = self._calculate_current_atr(df)

        if atr is None or atr <= 0:
            logger.warning("Invalid ATR")
            return

        # Calculate position size
        position = self.risk_manager.calculate_position_size(
            capital=capital,
            entry_price=current_price,
            atr=atr,
            side=side,
        )

        # Calculate take profit levels (direction-specific)
        tp_levels = self.risk_manager.calculate_take_profit_prices(
            entry_price=current_price,
            stop_loss=position.stop_loss,
            side=side,
        )

        direction = "LONG" if side == "BUY" else "SHORT"
        logger.info(
            f"Opening {direction} position: size={position.size:.6f}, "
            f"entry={current_price}, sl={position.stop_loss}"
        )

        # Place order
        if self.settings.mode == "live":
            ticker = self.client.get_ticker(self.settings.symbol)

            # For maker orders:
            # LONG: use bid (slightly below market to get filled as maker)
            # SHORT: use ask (slightly above market to get filled as maker)
            if side == "BUY":
                entry_price = ticker.bid
            else:
                entry_price = ticker.ask

            # Use margin order if leverage is enabled
            if self.settings.use_leverage:
                pos = self.order_manager.open_margin_position(
                    side=side,
                    size=position.size,
                    entry_price=entry_price,
                    stop_loss=position.stop_loss,
                    take_profit_levels=tp_levels,
                )
            else:
                pos = self.order_manager.open_position(
                    side=side,
                    size=position.size,
                    entry_price=entry_price,
                    stop_loss=position.stop_loss,
                    take_profit_levels=tp_levels,
                )

            if pos:
                # Record trade
                self._record_trade_entry(pos, confidence)

                # Send notification
                await self.telegram.notify_trade_opened(
                    symbol=self.settings.symbol,
                    side=side,
                    price=entry_price,
                    size=position.size,
                    stop_loss=position.stop_loss,
                    confidence=confidence,
                )
        else:
            logger.info(f"Paper trading: Would open {direction} position at {current_price}")

    async def _manage_existing_position(self, df: pd.DataFrame) -> None:
        """Manage existing position."""
        position = self.order_manager.current_position
        if position is None:
            return

        current_price = df["close"].iloc[-1]

        # Check if entry order is filled
        if position.status == PositionStatus.PENDING:
            if self.order_manager.check_entry_filled():
                # Place take profit orders
                self.order_manager.place_take_profit_orders()
                logger.info("Entry filled, TP orders placed")
            return

        # Check stop loss
        if self.order_manager.check_stop_loss(current_price):
            pnl = self.order_manager.execute_stop_loss()

            # Record and notify (track by direction)
            self._record_trade_exit(position, pnl, "STOPPED")
            self.risk_manager.add_trade_result(pnl, position.side.value)

            await self.telegram.notify_stop_loss(
                symbol=self.settings.symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                stop_price=position.stop_loss,
                pnl=pnl,
            )
            return

        # Check take profit fills
        pnl = self.order_manager.check_take_profit_fills()
        if pnl != 0:
            logger.info(f"Take profit executed: PnL={pnl}")

            if not self.order_manager.has_position():
                # Position fully closed (track by direction)
                self._record_trade_exit(position, position.realized_pnl, "TP")
                self.risk_manager.add_trade_result(position.realized_pnl, position.side.value)

                await self.telegram.notify_trade_closed(
                    symbol=self.settings.symbol,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    pnl=position.realized_pnl,
                    pnl_percent=position.realized_pnl / (position.entry_price * position.size),
                    reason="TP",
                )

    def _calculate_current_atr(self, df: pd.DataFrame, period: int = 14) -> float | None:
        """Calculate current ATR value."""
        if len(df) < period + 1:
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        return float(atr.iloc[-1])

    def _get_capital(self) -> float:
        """Get current available capital."""
        if self.settings.mode == "live":
            return self.client.get_jpy_balance()
        else:
            # Paper trading: use fixed amount
            return 1_000_000  # 1M JPY

    def _record_signal(
        self,
        direction: int,
        confidence: float,
        price: float,
        features: Any,
    ) -> None:
        """Record signal to database."""
        import json

        signal = Signal(
            symbol=self.settings.symbol,
            direction=direction,
            confidence=confidence,
            price=price,
            features=json.dumps(features.tolist()) if features is not None else None,
        )
        self.session.add(signal)
        self.session.commit()

    def _record_trade_entry(self, position: Any, confidence: float) -> None:
        """Record trade entry to database."""
        trade = Trade(
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            size=position.size,
            stop_loss=position.stop_loss,
            status="OPEN",
            entry_time=position.entry_time,
            confidence=confidence,
        )
        self.session.add(trade)
        self.session.commit()

    def _record_trade_exit(self, position: Any, pnl: float, status: str) -> None:
        """Record trade exit to database."""
        # Find open trade
        trades = self.trade_repo.get_open_trades()
        for trade in trades:
            if trade.symbol == position.symbol:
                self.trade_repo.update(
                    trade.id,
                    {
                        "exit_price": position.entry_price,  # Approximate
                        "pnl": pnl,
                        "pnl_percent": pnl / (position.entry_price * position.size),
                        "exit_time": datetime.now(),
                        "status": status,
                    },
                )
                break

    # Report Methods

    async def send_status_report(self, report_type: str = "朝") -> None:
        """Send status report (morning/noon/evening)."""
        capital = self._get_capital()
        position_info = self.order_manager.get_position_info()
        direction_stats = self.risk_manager.get_daily_stats()

        await self.report_generator.generate_status_report(
            position_info=position_info,
            capital=capital,
            report_type=report_type,
            direction_stats=direction_stats,
        )

    async def send_daily_report(self) -> None:
        """Send daily report."""
        capital = self._get_capital()
        direction_stats = self.risk_manager.get_daily_stats()
        await self.report_generator.generate_daily_report(
            capital=capital,
            direction_stats=direction_stats,
        )

    async def send_weekly_report(self) -> None:
        """Send weekly report."""
        capital = self._get_capital()
        await self.report_generator.generate_weekly_report(capital)

    async def send_monthly_report(self) -> None:
        """Send monthly report."""
        capital = self._get_capital()
        await self.report_generator.generate_monthly_report(capital)

    def close(self) -> None:
        """Clean up resources."""
        self.client.close()
        self.session.close()
