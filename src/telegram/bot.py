"""Telegram bot for notifications and reports."""

import asyncio
from typing import Any

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode


class TelegramBot:
    """Telegram bot for sending trading notifications and reports."""

    def __init__(self, token: str, chat_id: str) -> None:
        """
        Initialize Telegram bot.

        Args:
            token: Telegram bot token
            chat_id: Target chat ID for messages
        """
        self.token = token
        self.chat_id = chat_id
        self._bot: Bot | None = None

    @property
    def bot(self) -> Bot:
        """Get bot instance (lazy initialization)."""
        if self._bot is None:
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_message(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """
        Send a message to the configured chat.

        Args:
            text: Message text
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if sent successfully
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message_sync(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """
        Send a message synchronously.

        Args:
            text: Message text
            parse_mode: Parse mode

        Returns:
            True if sent successfully
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.send_message(text, parse_mode))

    # Trading Notifications

    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        stop_loss: float,
        confidence: float,
    ) -> bool:
        """Send notification when a trade is opened."""
        direction = "LONG" if side == "BUY" else "SHORT"
        emoji = "📈" if side == "BUY" else "📉"

        text = f"""
{emoji} <b>新規ポジション</b>

通貨: {symbol}
方向: {direction}
価格: ¥{price:,.0f}
数量: {size:.6f}
損切: ¥{stop_loss:,.0f}
信頼度: {confidence:.1%}
"""
        return await self.send_message(text.strip())

    async def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        reason: str = "TP",
    ) -> bool:
        """Send notification when a trade is closed."""
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_sign = "+" if pnl >= 0 else ""

        text = f"""
{emoji} <b>ポジション決済</b>

通貨: {symbol}
方向: {side}
エントリー: ¥{entry_price:,.0f}
決済: ¥{exit_price:,.0f}
損益: {pnl_sign}¥{pnl:,.0f} ({pnl_sign}{pnl_percent:.2%})
理由: {reason}
"""
        return await self.send_message(text.strip())

    async def notify_stop_loss(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        pnl: float,
    ) -> bool:
        """Send notification when stop loss is triggered."""
        text = f"""
🛑 <b>損切り発動</b>

通貨: {symbol}
方向: {side}
エントリー: ¥{entry_price:,.0f}
損切価格: ¥{stop_price:,.0f}
損失: ¥{pnl:,.0f}
"""
        return await self.send_message(text.strip())

    async def notify_signal_skipped(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        reason: str,
    ) -> bool:
        """Send notification when a signal is skipped."""
        text = f"""
⏭️ <b>シグナルスキップ</b>

通貨: {symbol}
方向: {direction}
信頼度: {confidence:.1%}
理由: {reason}
"""
        return await self.send_message(text.strip())

    async def notify_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        text = f"""
⚠️ <b>エラー発生</b>

{error}
{f'コンテキスト: {context}' if context else ''}
"""
        return await self.send_message(text.strip())

    # Report Methods

    async def send_daily_report(
        self,
        date: str,
        trades: int,
        wins: int,
        net_pnl: float,
        capital: float,
        report_type: str = "日次",
        direction_stats: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send daily trading report.

        Args:
            date: Report date
            trades: Number of trades
            wins: Number of winning trades
            net_pnl: Net profit/loss
            capital: Current capital
            report_type: Report type (朝/昼/夕方)
            direction_stats: Optional direction-specific statistics
        """
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_sign = "+" if net_pnl >= 0 else ""
        emoji = "📊" if report_type == "日次" else "📋"

        # Build direction breakdown if available
        direction_text = ""
        if direction_stats:
            long = direction_stats.get("long", {})
            short = direction_stats.get("short", {})

            long_trades = long.get("trades", 0)
            long_pnl = long.get("pnl", 0)
            long_wr = long.get("win_rate", 0)
            long_pnl_sign = "+" if long_pnl >= 0 else ""

            short_trades = short.get("trades", 0)
            short_pnl = short.get("pnl", 0)
            short_wr = short.get("win_rate", 0)
            short_pnl_sign = "+" if short_pnl >= 0 else ""

            direction_text = f"""
📈 LONG:
  • 取引: {long_trades}回 | 勝率: {long_wr:.0%}
  • 損益: {long_pnl_sign}¥{long_pnl:,.0f}

📉 SHORT:
  • 取引: {short_trades}回 | 勝率: {short_wr:.0%}
  • 損益: {short_pnl_sign}¥{short_pnl:,.0f}
"""

        text = f"""
{emoji} <b>{report_type}レポート</b>
━━━━━━━━━━━━━━

📅 日付: {date}

📊 全体実績:
  • 取引数: {trades}回
  • 勝率: {win_rate:.1f}%
  • 勝ち: {wins}回 / 負け: {trades - wins}回
{direction_text}
💰 損益:
  • 本日損益: {pnl_sign}¥{net_pnl:,.0f}
  • 現在資金: ¥{capital:,.0f}

━━━━━━━━━━━━━━
"""
        return await self.send_message(text.strip())

    async def send_weekly_report(
        self,
        week_start: str,
        week_end: str,
        trades: int,
        wins: int,
        net_pnl: float,
        capital_start: float,
        capital_end: float,
        best_trade: float,
        worst_trade: float,
    ) -> bool:
        """Send weekly trading report."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_sign = "+" if net_pnl >= 0 else ""
        capital_change = capital_end - capital_start
        capital_pct = (capital_change / capital_start * 100) if capital_start > 0 else 0

        text = f"""
📊 <b>週次レポート</b>
━━━━━━━━━━━━━━

📅 期間: {week_start} ~ {week_end}

📈 取引実績:
  • 総取引数: {trades}回
  • 勝率: {win_rate:.1f}%
  • 勝ち: {wins}回 / 負け: {trades - wins}回

💰 損益:
  • 週間損益: {pnl_sign}¥{net_pnl:,.0f}
  • 資金変動: {pnl_sign}¥{capital_change:,.0f} ({pnl_sign}{capital_pct:.2f}%)

📌 ハイライト:
  • ベスト: +¥{best_trade:,.0f}
  • ワースト: ¥{worst_trade:,.0f}

💼 資金状況:
  • 週初: ¥{capital_start:,.0f}
  • 週末: ¥{capital_end:,.0f}

━━━━━━━━━━━━━━
"""
        return await self.send_message(text.strip())

    async def send_monthly_report(
        self,
        month: str,
        trades: int,
        wins: int,
        net_pnl: float,
        capital_start: float,
        capital_end: float,
        max_drawdown: float,
        sharpe_ratio: float | None = None,
    ) -> bool:
        """Send monthly trading report."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_sign = "+" if net_pnl >= 0 else ""
        capital_change = capital_end - capital_start
        capital_pct = (capital_change / capital_start * 100) if capital_start > 0 else 0

        sharpe_text = f"  • シャープレシオ: {sharpe_ratio:.2f}" if sharpe_ratio else ""

        text = f"""
📊 <b>月次レポート</b>
━━━━━━━━━━━━━━

📅 期間: {month}

📈 取引実績:
  • 総取引数: {trades}回
  • 勝率: {win_rate:.1f}%
  • 勝ち: {wins}回 / 負け: {trades - wins}回

💰 損益:
  • 月間損益: {pnl_sign}¥{net_pnl:,.0f}
  • 資金変動: {pnl_sign}¥{capital_change:,.0f} ({pnl_sign}{capital_pct:.2f}%)

📉 リスク指標:
  • 最大ドローダウン: {max_drawdown:.2%}
{sharpe_text}

💼 資金状況:
  • 月初: ¥{capital_start:,.0f}
  • 月末: ¥{capital_end:,.0f}

━━━━━━━━━━━━━━
"""
        return await self.send_message(text.strip())

    async def send_status_report(
        self,
        position_info: dict[str, Any] | None,
        capital: float,
        daily_pnl: float,
        daily_trades: int,
        direction_stats: dict[str, Any] | None = None,
    ) -> bool:
        """Send current status report (morning/noon/evening)."""
        position_text = "なし"
        if position_info:
            position_text = f"""
  通貨: {position_info['symbol']}
  方向: {position_info['side']}
  エントリー: ¥{position_info['entry_price']:,.0f}
  サイズ: {position_info['size']:.6f}
  含み損益: ¥{position_info.get('unrealized_pnl', 0):,.0f}"""

        pnl_sign = "+" if daily_pnl >= 0 else ""

        # Direction breakdown
        direction_text = ""
        if direction_stats:
            long = direction_stats.get("long", {})
            short = direction_stats.get("short", {})

            long_trades = long.get("trades", 0)
            long_pnl = long.get("pnl", 0)
            long_pnl_sign = "+" if long_pnl >= 0 else ""

            short_trades = short.get("trades", 0)
            short_pnl = short.get("pnl", 0)
            short_pnl_sign = "+" if short_pnl >= 0 else ""

            if long_trades > 0 or short_trades > 0:
                direction_text = f"""
📈 LONG: {long_trades}回 ({long_pnl_sign}¥{long_pnl:,.0f})
📉 SHORT: {short_trades}回 ({short_pnl_sign}¥{short_pnl:,.0f})
"""

        text = f"""
📋 <b>ステータスレポート</b>
━━━━━━━━━━━━━━

💼 資金: ¥{capital:,.0f}

📊 本日の実績:
  • 取引数: {daily_trades}回
  • 損益: {pnl_sign}¥{daily_pnl:,.0f}
{direction_text}
📍 現在ポジション:
{position_text}

━━━━━━━━━━━━━━
"""
        return await self.send_message(text.strip())
