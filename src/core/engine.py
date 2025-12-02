"""Main trading engine."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings
from src.utils.timezone import now_jst, JST
from src.agent.long_term_memory import LongTermMemory
from src.api.gmo_client import GMOCoinClient
from src.database.models import (
    DailyPnLRepository,
    Signal,
    Trade,
    TradeRepository,
    WalkForwardRepository,
    init_db,
)
from src.execution.order_manager import OrderManager, PositionStatus
from src.execution.paper_executor import PaperTradingExecutor, PaperTradingConfig
from src.features.calculator import FeatureCalculator
from src.features.registry import FeatureRegistry
from src.models.predictor import Predictor
from src.reports.generator import ReportGenerator
from src.risk.manager import RiskManager, DynamicLeverageConfig
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
        self.walkforward_repo = WalkForwardRepository(self.session)

        # Initialize API client
        self.client = GMOCoinClient(
            api_key=settings.gmo_api_key,
            api_secret=settings.gmo_api_secret,
            base_url=settings.gmo_base_url,
            private_url=settings.gmo_private_url,
        )

        # Initialize feature registry (shared with agent for dynamic feature management)
        self.feature_registry = FeatureRegistry(config_path="data/feature_registry.json")

        # Initialize components with feature registry
        self.feature_calc = FeatureCalculator(registry=self.feature_registry)
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

        # Configure portfolio allocation
        self.risk_manager.configure_allocation(
            symbol_allocations=settings.get_symbol_allocations(),
            total_capital_utilization=settings.total_capital_utilization,
            long_allocation_ratio=settings.long_allocation_ratio,
            short_allocation_ratio=settings.short_allocation_ratio,
        )

        # Initialize long-term memory (shared with MetaAgent via same directory)
        self.long_term_memory = LongTermMemory(memory_dir="data/memory")

        # Connect long-term memory to RiskManager for rule-based adjustments
        # This allows learned rules like "reduce position in high volatility" to be applied
        self.risk_manager.set_long_term_memory(
            self.long_term_memory,
            enable_adjustments=getattr(settings, 'enable_memory_adjustments', True),
        )

        # Enable dynamic leverage if leverage is configured
        if settings.use_leverage:
            dynamic_leverage_config = DynamicLeverageConfig(
                base_leverage=settings.leverage,
                min_leverage=1.0,
                max_leverage=min(settings.leverage * 2, 10.0),  # Up to 2x base, max 10x
                # Confidence thresholds aligned with risk manager thresholds
                high_confidence_threshold=0.85,
                medium_confidence_threshold=settings.long_confidence_threshold,
                low_confidence_threshold=0.65,
            )
            self.risk_manager.enable_dynamic_leverage(dynamic_leverage_config)

        # Store symbols for multi-asset trading
        self.symbols = list(settings.get_symbol_allocations().keys())
        self.primary_symbol = settings.symbol  # Legacy fallback

        # Order managers for each symbol
        self.order_managers: dict[str, OrderManager] = {}
        for symbol in self.symbols:
            self.order_managers[symbol] = OrderManager(self.client, symbol)

        # Legacy single order manager (for backward compatibility)
        self.order_manager = OrderManager(self.client, settings.symbol)

        # Paper trading executor (for paper/test mode)
        self.paper_executor: PaperTradingExecutor | None = None
        if settings.mode == "paper":
            paper_config = PaperTradingConfig(
                initial_capital=1_000_000,  # 1M JPY virtual capital
                slippage_bps=5.0,
                limit_fill_probability=0.95,
                maker_fee=-0.0001,
                taker_fee=0.0004,
            )
            self.paper_executor = PaperTradingExecutor(paper_config)

            # Load saved state if exists
            paper_state_path = Path("data/paper_trading_state.json")
            if paper_state_path.exists():
                self.paper_executor.load_state(str(paper_state_path))

        # Initialize Telegram bot
        self.telegram = TelegramBot(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.report_generator = ReportGenerator(
            trade_repo=self.trade_repo,
            daily_pnl_repo=self.daily_pnl_repo,
            telegram_bot=self.telegram,
            walkforward_repo=self.walkforward_repo,
        )

        # Cache for OHLCV data
        self._ohlcv_cache: pd.DataFrame | None = None
        self._last_fetch_time: datetime | None = None

        # Overfitting check state
        self._last_overfit_check: datetime | None = None
        self._overfit_check_interval = timedelta(hours=24)  # Check once daily

    async def run_cycle(self) -> None:
        """Run one trading cycle (called every 15 minutes)."""
        logger.info("Starting trading cycle")

        try:
            # 0a. Check if feature registry has been updated externally (by agent)
            if self.feature_registry.reload_if_changed():
                logger.info("Feature registry reloaded with updated settings")

            # 0b. Check for overfitting and adjust risk parameters
            await self._check_and_respond_to_overfitting()

            # 1. Fetch latest data
            df = await self._fetch_ohlcv_data()

            if df.empty:
                logger.warning("No OHLCV data available")
                return

            # 2. Check existing position
            if self._has_open_position():
                await self._manage_existing_position(df)
            else:
                await self._check_new_signal(df)

            logger.info("Trading cycle completed")

        except Exception as e:
            logger.exception(f"Error in trading cycle: {e}")
            await self.telegram.notify_error(str(e), "trading_cycle")

    async def _fetch_ohlcv_data(self) -> pd.DataFrame:
        """Fetch and cache OHLCV data."""
        now = now_jst()

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

    async def _check_and_respond_to_overfitting(self) -> None:
        """Check for overfitting and enable conservative mode if needed."""
        now = now_jst()

        # Skip if checked recently
        if (
            self._last_overfit_check is not None
            and now - self._last_overfit_check < self._overfit_check_interval
        ):
            return

        self._last_overfit_check = now

        # Check for degradation
        degradation = self.walkforward_repo.check_degradation()

        if degradation.get("degraded"):
            if not self.risk_manager.is_conservative_mode:
                # Enable conservative mode
                gap = degradation.get("gap", 0)
                if gap > 0.15:
                    multiplier = 0.3  # Very conservative for severe overfitting
                elif gap > 0.10:
                    multiplier = 0.5  # Moderate conservative
                else:
                    multiplier = 0.7  # Mild conservative

                self.risk_manager.enable_conservative_mode(multiplier)

                # Notify via Telegram
                await self.telegram.notify_overfitting_detected(
                    reason=degradation.get("reason", "Unknown"),
                    gap=gap,
                    action=f"Conservative mode enabled (risk reduced to {multiplier:.0%})",
                )

                logger.warning(
                    f"Overfitting detected: {degradation.get('reason')}. "
                    f"Conservative mode enabled with multiplier {multiplier}"
                )
        else:
            # Disable conservative mode if previously enabled and performance is now stable
            if self.risk_manager.is_conservative_mode:
                self.risk_manager.disable_conservative_mode()

                await self.telegram.notify_conservative_mode_disabled(
                    reason="Model performance stabilized"
                )

                logger.info("Conservative mode disabled - model performance stabilized")

    async def _check_new_signal(self, df: pd.DataFrame) -> None:
        """Check for new trading signal (long or short)."""
        # Get capital
        capital = self._get_capital()

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

        # Check if long-term memory rules suggest skipping this trade
        should_skip, skip_reason = self.risk_manager.should_skip_trade(
            side=side,
            market_conditions=self._get_market_conditions(df),
        )
        if should_skip:
            logger.info(f"{direction} trade skipped by memory rule: {skip_reason}")
            return

        logger.info(f"Signal: {direction} with confidence {confidence:.4f}")

        # Calculate dynamic leverage based on prediction confidence and market conditions
        current_price = df["close"].iloc[-1]
        atr = self._calculate_current_atr(df)

        if self.settings.use_leverage:
            leverage_calc = self.risk_manager.calculate_dynamic_leverage(
                confidence=confidence,
                side=side,
                current_price=current_price,
                atr=atr,
                market_conditions=self._get_market_conditions(df),
            )
            effective_capital = capital * leverage_calc.adjusted_leverage
            logger.info(
                f"Dynamic leverage: {leverage_calc.base_leverage:.1f}x → {leverage_calc.adjusted_leverage:.1f}x "
                f"(capital: ¥{capital:,.0f} → ¥{effective_capital:,.0f})"
            )
            for reason in leverage_calc.reasons:
                logger.debug(f"  - {reason}")
        else:
            effective_capital = capital

        # Execute trade
        await self._execute_entry(df, confidence, effective_capital, side)

    def _get_market_conditions(self, df: pd.DataFrame) -> dict:
        """Extract current market conditions from OHLCV data."""
        if len(df) < 20:
            return {}

        current_price = df["close"].iloc[-1]
        atr = self._calculate_current_atr(df)

        # Calculate simple trend using 20-period SMA
        sma20 = df["close"].tail(20).mean()
        if current_price > sma20 * 1.01:
            trend = "up"
        elif current_price < sma20 * 0.99:
            trend = "down"
        else:
            trend = "neutral"

        # Calculate volatility level
        if atr and current_price > 0:
            vol_pct = atr / current_price
            if vol_pct >= 0.03:
                volatility = "high"
            elif vol_pct <= 0.01:
                volatility = "low"
            else:
                volatility = "normal"
        else:
            volatility = "normal"

        return {
            "trend": trend,
            "volatility": volatility,
            "price": current_price,
            "atr": atr,
        }

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
            try:
                ticker = self.client.get_ticker(self.settings.symbol)
                if ticker is None:
                    logger.error("Failed to get ticker data - API returned None")
                    await self.telegram.notify_error("Ticker API returned None", "execute_entry")
                    return
            except Exception as e:
                logger.error(f"Failed to get ticker: {e}")
                await self.telegram.notify_error(f"Ticker API error: {e}", "execute_entry")
                return

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
            # Paper trading mode - simulate trade execution
            if self.paper_executor:
                paper_pos = self.paper_executor.open_position(
                    symbol=self.settings.symbol,
                    side=side,
                    size=position.size,
                    entry_price=current_price,
                    stop_loss=position.stop_loss,
                    take_profit_levels=tp_levels,
                    confidence=confidence,
                    use_limit_order=True,
                )

                if paper_pos:
                    # Record to database with paper flag
                    self._record_paper_trade_entry(paper_pos)

                    # Send notification with [PAPER] prefix
                    await self.telegram.notify_trade_opened(
                        symbol=self.settings.symbol,
                        side=side,
                        price=paper_pos.entry_price,
                        size=position.size,
                        stop_loss=position.stop_loss,
                        confidence=confidence,
                        is_paper=True,
                    )

                    # Save paper state periodically
                    self._save_paper_state()

                    logger.info(
                        f"[PAPER] {direction} position opened: "
                        f"size={position.size:.6f} @ ¥{paper_pos.entry_price:,.0f}"
                    )
            else:
                logger.info(f"Paper trading: Would open {direction} position at {current_price}")

    async def _manage_existing_position(self, df: pd.DataFrame) -> None:
        """Manage existing position."""
        current_price = df["close"].iloc[-1]

        # Paper trading mode
        if self.settings.mode == "paper" and self.paper_executor:
            await self._manage_paper_positions(current_price)
            return

        # Live trading mode
        position = self.order_manager.current_position
        if position is None:
            return

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

                # Calculate PnL percent safely (avoid division by zero)
                position_value = position.entry_price * position.size
                pnl_percent = (
                    position.realized_pnl / position_value
                    if position_value > 0
                    else 0.0
                )

                await self.telegram.notify_trade_closed(
                    symbol=self.settings.symbol,
                    side=position.side.value,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    pnl=position.realized_pnl,
                    pnl_percent=pnl_percent,
                    reason="TP",
                )

    async def _manage_paper_positions(self, current_price: float) -> None:
        """Manage paper trading positions."""
        if not self.paper_executor:
            return

        current_prices = {self.settings.symbol: current_price}

        # Check stop losses
        sl_triggers = self.paper_executor.check_stop_losses(current_prices)
        for trigger in sl_triggers:
            # Update database
            self._record_paper_trade_exit(trigger["position_id"], trigger["pnl"], "SL")

            # Track with risk manager
            side = "BUY" if trigger["side"] == "LONG" else "SELL"
            self.risk_manager.add_trade_result(trigger["pnl"], side)

            # Notify
            await self.telegram.notify_stop_loss(
                symbol=trigger["symbol"],
                side=side,
                entry_price=0,  # We don't have this info readily available
                stop_price=current_price,
                pnl=trigger["pnl"],
                is_paper=True,
            )

        # Check take profits
        tp_triggers = self.paper_executor.check_take_profits(current_prices)
        for trigger in tp_triggers:
            # Update database
            self._record_paper_trade_exit(trigger["position_id"], trigger["pnl"], "TP")

            # Track with risk manager (only if position fully closed)
            if not self.paper_executor.get_position(trigger["position_id"]):
                side = "BUY" if trigger["side"] == "LONG" else "SELL"
                self.risk_manager.add_trade_result(trigger["pnl"], side)

                await self.telegram.notify_trade_closed(
                    symbol=trigger["symbol"],
                    side=side,
                    entry_price=0,
                    exit_price=trigger["tp_level"],
                    pnl=trigger["pnl"],
                    pnl_percent=0,
                    reason="TP",
                    is_paper=True,
                )

        # Save state after managing positions
        self._save_paper_state()

    def _record_paper_trade_exit(self, position_id: str, pnl: float, status: str) -> None:
        """Record paper trade exit to database."""
        # Find open paper trade by position ID
        trades = self.trade_repo.get_open_trades()
        for trade in trades:
            # Match by paper_position_id for precise tracking
            if trade.is_paper and trade.paper_position_id == position_id:
                self.trade_repo.update(
                    trade.id,
                    {
                        "pnl": pnl,
                        "exit_time": now_jst(),
                        "status": status,
                    },
                )
                logger.debug(f"Paper trade exit recorded: position_id={position_id}, pnl={pnl:.2f}")
                return

        # Fallback: match by symbol if position_id not found (legacy trades)
        for trade in trades:
            if trade.is_paper and trade.symbol == self.settings.symbol:
                self.trade_repo.update(
                    trade.id,
                    {
                        "pnl": pnl,
                        "exit_time": now_jst(),
                        "status": status,
                    },
                )
                logger.warning(f"Paper trade exit matched by symbol (legacy): {trade.symbol}")
                return

        logger.warning(f"No matching paper trade found for position_id={position_id}")

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
            # Paper trading: use paper executor's virtual capital
            if self.paper_executor:
                return self.paper_executor.get_capital()
            return 1_000_000  # 1M JPY fallback

    def _has_open_position(self) -> bool:
        """Check if there are any open positions."""
        if self.settings.mode == "live":
            return self.order_manager.has_position()
        else:
            if self.paper_executor:
                return self.paper_executor.has_open_position(self.settings.symbol)
            return self.order_manager.has_position()

    def _record_paper_trade_entry(self, paper_pos: Any) -> None:
        """Record paper trade entry to database."""
        trade = Trade(
            symbol=paper_pos.symbol,
            side=paper_pos.side.value,
            entry_price=paper_pos.entry_price,
            size=paper_pos.size,
            stop_loss=paper_pos.stop_loss,
            status="OPEN",
            entry_time=paper_pos.entry_time,
            confidence=paper_pos.confidence,
            is_paper=True,  # Mark as paper trade
            paper_position_id=paper_pos.position_id,  # Store position ID for precise tracking
        )
        self.session.add(trade)
        self.session.commit()

    def _save_paper_state(self) -> None:
        """Save paper trading state to file."""
        if self.paper_executor:
            self.paper_executor.save_state("data/paper_trading_state.json")

    def _record_signal(
        self,
        direction: int,
        confidence: float,
        price: float,
        features: Any,
    ) -> None:
        """Record signal to database."""
        import json

        # Serialize features safely (handles NumPy arrays and other types)
        features_json = None
        if features is not None:
            try:
                if hasattr(features, 'tolist'):
                    features_json = json.dumps(features.tolist())
                else:
                    features_json = json.dumps(features)
            except (TypeError, AttributeError) as e:
                logger.warning(f"Failed to serialize features: {e}")

        signal = Signal(
            symbol=self.settings.symbol,
            direction=direction,
            confidence=confidence,
            price=price,
            features=features_json,
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
                # Calculate PnL percent safely (avoid division by zero)
                position_value = position.entry_price * position.size
                pnl_percent = pnl / position_value if position_value > 0 else 0.0

                self.trade_repo.update(
                    trade.id,
                    {
                        "exit_price": position.entry_price,  # Approximate
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "exit_time": now_jst(),
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

    async def send_model_analysis_report(self) -> None:
        """Send model analysis report (bi-weekly walk-forward analysis)."""
        await self.report_generator.generate_model_analysis_report()

        # Check if retraining is needed and trigger if so
        await self._check_and_trigger_retraining()

    async def _check_and_trigger_retraining(self) -> None:
        """Check if model retraining is needed and trigger if so."""
        from training.retrainer import AutoRetrainer

        try:
            retrainer = AutoRetrainer(
                walkforward_repo=self.walkforward_repo,
                data_path=self.settings.training_data_path,
                model_output_path=self.settings.model_path,
            )

            # Check if retraining is needed
            check = retrainer.check_retraining_needed()

            if not check["needed"]:
                logger.info(f"Retraining not needed: {check['reason']}")
                return

            logger.info(f"Retraining triggered: {check['reason']}")

            # Get recommended params
            params_info = retrainer.get_recommended_params()

            # Notify start
            await self.telegram.notify_retraining_triggered(
                reason=check["reason"],
                params_adjusted=params_info.get("params"),
            )

            # Run retraining (this may take a while)
            result = retrainer.retrain_model(severity=check["severity"])

            if result.get("success"):
                # Reload model in predictor
                self.predictor.reload_model(self.settings.model_path)

                # Disable conservative mode since we have a fresh model
                if self.risk_manager.is_conservative_mode:
                    self.risk_manager.disable_conservative_mode()

                # Get old metrics for comparison
                history = self.walkforward_repo.get_history(limit=2)
                old_accuracy = history[1].test_accuracy_mean if len(history) > 1 else None
                new_accuracy = result.get("metrics", {}).get("accuracy_mean")

                # Notify completion
                await self.telegram.send_retraining_notification(
                    reason=check["reason"],
                    old_accuracy=old_accuracy,
                    new_accuracy=new_accuracy,
                    improvement=(
                        new_accuracy - old_accuracy
                        if old_accuracy and new_accuracy
                        else None
                    ),
                )

                logger.info(f"Retraining completed: {result.get('model_version')}")
            else:
                logger.error(f"Retraining failed: {result.get('error')}")
                await self.telegram.notify_error(
                    f"Model retraining failed: {result.get('error')}",
                    "retraining",
                )

        except Exception as e:
            logger.exception(f"Error in retraining check: {e}")

    async def send_paper_trading_report(self) -> None:
        """Send paper trading performance report."""
        if not self.paper_executor:
            logger.warning("Paper executor not available for report")
            return

        summary = self.paper_executor.get_trade_summary()

        # Format report message
        message = (
            f"📊 <b>[PAPER] トレード実績レポート</b>\n\n"
            f"📅 期間: {summary['session_start'][:10]} 〜 現在\n"
            f"⏱ 経過時間: {summary['duration_hours']:.1f}時間\n\n"
            f"💰 <b>資金状況</b>\n"
            f"初期資金: ¥{summary['initial_capital']:,.0f}\n"
            f"現在資金: ¥{summary['current_capital']:,.0f}\n"
            f"最高資金: ¥{summary['peak_capital']:,.0f}\n"
            f"総損益: ¥{summary['total_pnl']:+,.0f} ({summary['total_return_pct']:+.2f}%)\n"
            f"最大DD: {summary['max_drawdown_pct']:.2f}%\n\n"
            f"📈 <b>トレード統計</b>\n"
            f"総取引数: {summary['total_trades']}\n"
            f"勝ち: {summary['winning_trades']} / 負け: {summary['losing_trades']}\n"
            f"勝率: {summary['win_rate']*100:.1f}%\n"
            f"PF: {summary['profit_factor']:.2f}\n"
            f"平均損益: ¥{summary['avg_trade_pnl']:+,.0f}\n\n"
            f"📊 <b>方向別</b>\n"
            f"LONG: {summary['long_trades']}回 ¥{summary['long_pnl']:+,.0f} "
            f"(勝率 {summary['long_win_rate']*100:.1f}%)\n"
            f"SHORT: {summary['short_trades']}回 ¥{summary['short_pnl']:+,.0f} "
            f"(勝率 {summary['short_win_rate']*100:.1f}%)\n\n"
            f"💹 オープンポジション: {summary['open_positions']}件\n"
            f"💳 累積手数料: ¥{summary['total_commission']:,.0f}"
        )

        await self.telegram.send_message(message)
        logger.info("Paper trading report sent")

    def get_paper_trading_stats(self) -> dict | None:
        """Get paper trading statistics for API."""
        if not self.paper_executor:
            return None
        return self.paper_executor.get_trade_summary()

    def close(self) -> None:
        """Clean up resources."""
        # Save paper trading state before closing
        if self.paper_executor:
            self._save_paper_state()
            logger.info("Paper trading state saved")

        self.client.close()
        self.session.close()
