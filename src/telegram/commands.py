"""Telegram command handlers for bot configuration."""

from typing import Any

from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.settings.runtime import get_runtime_settings


class TelegramCommandHandler:
    """Handles Telegram commands for configuration and status."""

    def __init__(self, token: str, chat_id: str) -> None:
        """
        Initialize command handler.

        Args:
            token: Telegram bot token
            chat_id: Authorized chat ID
        """
        self.token = token
        self.chat_id = chat_id
        self.runtime_settings = get_runtime_settings()
        self._engine: Any = None  # Set by engine after init

    def set_engine(self, engine: Any) -> None:
        """Set reference to trading engine."""
        self._engine = engine

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not self._check_authorized(update):
            return

        help_text = """
🤖 <b>GMO Coin Trading Bot</b>

<b>設定コマンド:</b>
/settings - 現在の設定を表示
/set [key] [value] - 設定を変更
/reset [key] - 設定をデフォルトに戻す
/allocation - 資金配分を表示

<b>設定変更例:</b>
/set symbols_config BTC_JPY:0.60,ETH_JPY:0.40
/set total_capital_utilization 0.75
/set long_allocation_ratio 0.70
/set short_allocation_ratio 0.30
/set mode paper

<b>取引コマンド:</b>
/status - 現在のステータス
/positions - ポジション一覧

<b>緊急停止:</b>
/stop - 新規取引を停止
/fullstop - 緊急停止（全決済）
/resume - 取引を再開

<b>レポート:</b>
/report - 本日のレポート
/weekly - 週次レポート
"""
        await update.message.reply_text(help_text, parse_mode="HTML")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command - show current settings."""
        if not self._check_authorized(update):
            return

        rs = self.runtime_settings

        # Get current effective settings
        from config.settings import Settings
        settings = Settings()

        # Apply runtime overrides for display
        symbols_config = rs.get("symbols_config", settings.symbols_config)
        utilization = rs.get("total_capital_utilization", settings.total_capital_utilization)
        long_ratio = rs.get("long_allocation_ratio", settings.long_allocation_ratio)
        short_ratio = rs.get("short_allocation_ratio", settings.short_allocation_ratio)
        mode = rs.get("mode", settings.mode)

        # Risk settings
        long_risk = rs.get("long_risk_per_trade", settings.long_risk_per_trade)
        short_risk = rs.get("short_risk_per_trade", settings.short_risk_per_trade)
        long_conf = rs.get("long_confidence_threshold", settings.long_confidence_threshold)
        short_conf = rs.get("short_confidence_threshold", settings.short_confidence_threshold)

        text = f"""
⚙️ <b>現在の設定</b>

📊 <b>ポートフォリオ配分:</b>
• コイン: {symbols_config}
• 資金使用率: {utilization:.0%}
• LONG配分: {long_ratio:.0%}
• SHORT配分: {short_ratio:.0%}

📈 <b>LONGリスク設定:</b>
• リスク/取引: {long_risk:.1%}
• 信頼度閾値: {long_conf:.0%}

📉 <b>SHORTリスク設定:</b>
• リスク/取引: {short_risk:.1%}
• 信頼度閾値: {short_conf:.0%}

🎮 <b>モード:</b> {mode}

{rs.get_display_summary()}
"""
        await update.message.reply_text(text.strip(), parse_mode="HTML")

    async def set_setting(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /set command - change a setting."""
        if not self._check_authorized(update):
            return

        args = context.args
        if not args or len(args) < 2:
            # Show available settings
            available = "\n".join([f"• {k}" for k in sorted(self.runtime_settings.MODIFIABLE_SETTINGS.keys())])
            await update.message.reply_text(
                f"使い方: /set [key] [value]\n\n<b>変更可能な設定:</b>\n{available}",
                parse_mode="HTML"
            )
            return

        key = args[0]
        value = " ".join(args[1:])  # Allow spaces in value (e.g., symbols_config)

        success, message = self.runtime_settings.set(key, value)

        if success:
            # Apply to engine if running
            if self._engine:
                self._apply_to_engine(key, value)

            await update.message.reply_text(f"✅ {message}")
            logger.info(f"Setting changed via Telegram: {key} = {value}")
        else:
            await update.message.reply_text(f"❌ {message}")

    async def reset_setting(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command - reset a setting to default."""
        if not self._check_authorized(update):
            return

        args = context.args
        if not args:
            await update.message.reply_text("使い方: /reset [key] または /reset all")
            return

        if args[0] == "all":
            self.runtime_settings.clear_all()
            await update.message.reply_text("✅ 全設定をデフォルトに戻しました")
            logger.info("All runtime settings reset via Telegram")
        else:
            key = args[0]
            success, message = self.runtime_settings.delete(key)
            if success:
                await update.message.reply_text(f"✅ {message}")
            else:
                await update.message.reply_text(f"❌ {message}")

    async def allocation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /allocation command - show current allocation status."""
        if not self._check_authorized(update):
            return

        if not self._engine:
            await update.message.reply_text("エンジンが初期化されていません")
            return

        try:
            # Get capital
            capital = self._engine._get_capital()
            summary = self._engine.risk_manager.get_allocation_summary(capital)

            text = f"""
💰 <b>資金配分状況</b>

総資金: ¥{summary['total_capital']:,.0f}
使用可能: ¥{summary['usable_capital']:,.0f} ({summary['utilization_rate']:.0%})

"""
            for symbol, data in summary["symbols"].items():
                text += f"""<b>{symbol}</b> ({data['allocation_pct']:.0%}):
  LONG: ¥{data['long_used']:,.0f} / ¥{data['long_allocated']:,.0f}
  SHORT: ¥{data['short_used']:,.0f} / ¥{data['short_allocated']:,.0f}

"""
            await update.message.reply_text(text.strip(), parse_mode="HTML")

        except Exception as e:
            logger.error(f"Error getting allocation: {e}")
            await update.message.reply_text(f"エラー: {e}")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command - show current status."""
        if not self._check_authorized(update):
            return

        if not self._engine:
            await update.message.reply_text("エンジンが初期化されていません")
            return

        try:
            capital = self._engine._get_capital()
            stats = self._engine.risk_manager.get_daily_stats()
            is_conservative = self._engine.risk_manager.is_conservative_mode

            mode = self.runtime_settings.get("mode", "paper")

            text = f"""
📋 <b>ステータス</b>

💼 資金: ¥{capital:,.0f}
🎮 モード: {mode}
🛡️ 保守モード: {"ON" if is_conservative else "OFF"}

📊 <b>本日の取引:</b>
• 総取引: {stats['total']['trades']}回
• 損益: ¥{stats['total']['pnl']:,.0f}

📈 LONG: {stats['long']['trades']}回 (勝率 {stats['long']['win_rate']:.0%})
📉 SHORT: {stats['short']['trades']}回 (勝率 {stats['short']['win_rate']:.0%})
"""
            await update.message.reply_text(text.strip(), parse_mode="HTML")

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            await update.message.reply_text(f"エラー: {e}")

    async def positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command - show open positions."""
        if not self._check_authorized(update):
            return

        if not self._engine:
            await update.message.reply_text("エンジンが初期化されていません")
            return

        try:
            open_trades = self._engine.trade_repo.get_open_trades()

            if not open_trades:
                await update.message.reply_text("📭 現在オープンポジションはありません")
                return

            text = "📍 <b>オープンポジション</b>\n\n"
            for trade in open_trades:
                text += f"""<b>{trade.symbol}</b> {trade.side}
• エントリー: ¥{trade.entry_price:,.0f}
• サイズ: {trade.size:.6f}
• SL: ¥{trade.stop_loss:,.0f}

"""
            await update.message.reply_text(text.strip(), parse_mode="HTML")

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            await update.message.reply_text(f"エラー: {e}")

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command - stop new positions."""
        if not self._check_authorized(update):
            return

        try:
            from src.api.main import get_emergency_stop, EmergencyStopMode, EmergencyStopReason

            emergency = get_emergency_stop()
            emergency.activate(
                mode=EmergencyStopMode.NO_NEW_POSITIONS,
                reason=EmergencyStopReason.MANUAL,
                message="Telegramから手動で停止",
            )

            await update.message.reply_text(
                "🛑 <b>新規取引を停止しました</b>\n\n"
                "既存のポジションは保持されます。\n"
                "再開するには /resume を使用してください。",
                parse_mode="HTML"
            )
            logger.warning("Trading stopped via Telegram (no new positions)")

        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            await update.message.reply_text(f"エラー: {e}")

    async def fullstop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /fullstop command - emergency stop with position closure."""
        if not self._check_authorized(update):
            return

        try:
            from src.api.main import get_emergency_stop, EmergencyStopMode, EmergencyStopReason

            emergency = get_emergency_stop()
            emergency.activate(
                mode=EmergencyStopMode.FULL_STOP,
                reason=EmergencyStopReason.MANUAL,
                message="Telegramから緊急停止",
            )

            await update.message.reply_text(
                "🚨 <b>緊急停止を実行しました</b>\n\n"
                "全ポジションの決済を試みます。\n"
                "再開するには /resume を使用してください。",
                parse_mode="HTML"
            )
            logger.warning("EMERGENCY STOP via Telegram (full stop)")

        except Exception as e:
            logger.error(f"Error with emergency stop: {e}")
            await update.message.reply_text(f"エラー: {e}")

    async def resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /resume command - resume trading."""
        if not self._check_authorized(update):
            return

        try:
            from src.api.main import get_emergency_stop

            emergency = get_emergency_stop()

            if not emergency.is_active():
                await update.message.reply_text("取引は既に稼働中です")
                return

            emergency.deactivate()

            await update.message.reply_text(
                "✅ <b>取引を再開しました</b>\n\n"
                "通常の取引が可能になりました。",
                parse_mode="HTML"
            )
            logger.info("Trading resumed via Telegram")

        except Exception as e:
            logger.error(f"Error resuming trading: {e}")
            await update.message.reply_text(f"エラー: {e}")

    def _check_authorized(self, update: Update) -> bool:
        """Check if the message is from authorized chat."""
        if str(update.effective_chat.id) != self.chat_id:
            logger.warning(f"Unauthorized access attempt from chat {update.effective_chat.id}")
            return False
        return True

    def _apply_to_engine(self, key: str, value: Any) -> None:
        """Apply a setting change to the running engine."""
        if not self._engine:
            return

        try:
            # Portfolio allocation changes
            if key in ["symbols_config", "total_capital_utilization", "long_allocation_ratio", "short_allocation_ratio"]:
                from config.settings import Settings
                settings = Settings()
                rs = self.runtime_settings

                # Get effective values
                symbols_str = rs.get("symbols_config", settings.symbols_config)
                allocations = {}
                for item in symbols_str.split(","):
                    item = item.strip()
                    if ":" in item:
                        sym, alloc = item.split(":")
                        allocations[sym.strip()] = float(alloc.strip())

                self._engine.risk_manager.configure_allocation(
                    symbol_allocations=allocations,
                    total_capital_utilization=rs.get("total_capital_utilization", settings.total_capital_utilization),
                    long_allocation_ratio=rs.get("long_allocation_ratio", settings.long_allocation_ratio),
                    short_allocation_ratio=rs.get("short_allocation_ratio", settings.short_allocation_ratio),
                )

            # Risk settings changes would require more complex updates
            # For now, they'll take effect on next restart or cycle

            logger.info(f"Applied setting {key} to engine")

        except Exception as e:
            logger.error(f"Failed to apply setting to engine: {e}")

    def build_application(self) -> Application:
        """Build the Telegram application with handlers."""
        app = Application.builder().token(self.token).build()

        # Add handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.start))
        app.add_handler(CommandHandler("settings", self.settings))
        app.add_handler(CommandHandler("set", self.set_setting))
        app.add_handler(CommandHandler("reset", self.reset_setting))
        app.add_handler(CommandHandler("allocation", self.allocation))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("positions", self.positions))
        # Emergency stop commands
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("fullstop", self.fullstop))
        app.add_handler(CommandHandler("resume", self.resume))

        return app


# Convenience function for quick settings changes
async def send_settings_update(bot: Any, chat_id: str, key: str, old_value: Any, new_value: Any) -> None:
    """Send notification about a settings change."""
    text = f"""
⚙️ <b>設定変更</b>

{key}: {old_value} → {new_value}
"""
    await bot.send_message(chat_id=chat_id, text=text.strip(), parse_mode="HTML")
