"""Report generation module."""

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from src.database.models import DailyPnL, DailyPnLRepository, Trade, TradeRepository
from src.telegram.bot import TelegramBot


class ReportGenerator:
    """Generate and send trading reports."""

    def __init__(
        self,
        trade_repo: TradeRepository,
        daily_pnl_repo: DailyPnLRepository,
        telegram_bot: TelegramBot,
    ) -> None:
        """
        Initialize report generator.

        Args:
            trade_repo: Trade repository
            daily_pnl_repo: Daily PnL repository
            telegram_bot: Telegram bot for sending reports
        """
        self.trade_repo = trade_repo
        self.daily_pnl_repo = daily_pnl_repo
        self.telegram_bot = telegram_bot

    async def generate_status_report(
        self,
        position_info: dict[str, Any] | None,
        capital: float,
        report_type: str = "朝",
        direction_stats: dict[str, Any] | None = None,
    ) -> bool:
        """
        Generate and send status report (morning/noon/evening).

        Args:
            position_info: Current position information
            capital: Current capital
            report_type: Report type (朝/昼/夕方)
            direction_stats: Optional LONG/SHORT breakdown from RiskManager
        """
        today = date.today().isoformat()
        daily_record = self.daily_pnl_repo.get_or_create(today)

        return await self.telegram_bot.send_status_report(
            position_info=position_info,
            capital=capital,
            daily_pnl=daily_record.net_pnl,
            daily_trades=daily_record.trades,
            direction_stats=direction_stats,
        )

    async def generate_daily_report(
        self,
        capital: float,
        direction_stats: dict[str, Any] | None = None,
    ) -> bool:
        """
        Generate and send daily report.

        Args:
            capital: Current capital
            direction_stats: Optional LONG/SHORT breakdown from RiskManager
        """
        today = date.today().isoformat()
        daily_record = self.daily_pnl_repo.get_or_create(today)

        return await self.telegram_bot.send_daily_report(
            date=today,
            trades=daily_record.trades,
            wins=daily_record.wins,
            net_pnl=daily_record.net_pnl,
            capital=capital,
            report_type="日次",
            direction_stats=direction_stats,
        )

    async def generate_weekly_report(self, capital: float) -> bool:
        """
        Generate and send weekly report.

        Args:
            capital: Current capital
        """
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = today

        # Get all daily records for the week
        daily_records = self.daily_pnl_repo.get_by_period(
            week_start.isoformat(),
            week_end.isoformat(),
        )

        # Aggregate statistics
        total_trades = sum(r.trades for r in daily_records)
        total_wins = sum(r.wins for r in daily_records)
        total_pnl = sum(r.net_pnl for r in daily_records)

        # Get capital at start and end of week
        capital_start = daily_records[0].capital_start if daily_records else capital
        capital_end = capital

        # Get trades for best/worst calculation
        trades = self.trade_repo.get_trades_by_period(
            week_start.isoformat(),
            week_end.isoformat(),
        )

        pnls = [t.pnl for t in trades if t.pnl is not None]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        return await self.telegram_bot.send_weekly_report(
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            trades=total_trades,
            wins=total_wins,
            net_pnl=total_pnl,
            capital_start=capital_start or capital,
            capital_end=capital_end,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    async def generate_monthly_report(self, capital: float) -> bool:
        """
        Generate and send monthly report.

        Args:
            capital: Current capital
        """
        today = date.today()
        month_start = today.replace(day=1)

        # Handle previous month report on first day
        if today.day == 1:
            # Report for previous month
            prev_month = (month_start - timedelta(days=1))
            month_start = prev_month.replace(day=1)
            month_end = prev_month
        else:
            month_end = today

        month_str = month_start.strftime("%Y年%m月")

        # Get all daily records for the month
        daily_records = self.daily_pnl_repo.get_by_period(
            month_start.isoformat(),
            month_end.isoformat(),
        )

        # Aggregate statistics
        total_trades = sum(r.trades for r in daily_records)
        total_wins = sum(r.wins for r in daily_records)
        total_pnl = sum(r.net_pnl for r in daily_records)

        # Get capital at start and end of month
        capital_start = daily_records[0].capital_start if daily_records else capital
        capital_end = capital

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(daily_records)

        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_records)

        return await self.telegram_bot.send_monthly_report(
            month=month_str,
            trades=total_trades,
            wins=total_wins,
            net_pnl=total_pnl,
            capital_start=capital_start or capital,
            capital_end=capital_end,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )

    def _calculate_max_drawdown(self, daily_records: list[DailyPnL]) -> float:
        """Calculate maximum drawdown from daily records."""
        if not daily_records:
            return 0.0

        capitals = []
        for r in daily_records:
            if r.capital_end:
                capitals.append(r.capital_end)

        if len(capitals) < 2:
            return 0.0

        peak = capitals[0]
        max_dd = 0.0

        for cap in capitals:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_sharpe_ratio(
        self,
        daily_records: list[DailyPnL],
        risk_free_rate: float = 0.0,
    ) -> float | None:
        """Calculate Sharpe ratio from daily returns."""
        if len(daily_records) < 2:
            return None

        returns = []
        for i in range(1, len(daily_records)):
            if daily_records[i - 1].capital_end and daily_records[i].capital_end:
                ret = (
                    daily_records[i].capital_end - daily_records[i - 1].capital_end
                ) / daily_records[i - 1].capital_end
                returns.append(ret)

        if len(returns) < 5:
            return None

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr) - risk_free_rate / 365
        std_return = np.std(returns_arr)

        if std_return == 0:
            return None

        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(365)
        return float(sharpe)

    def update_daily_pnl(
        self,
        trade: Trade,
        capital: float,
    ) -> None:
        """
        Update daily PnL record after a trade.

        Args:
            trade: Completed trade
            capital: Current capital
        """
        if trade.entry_time is None:
            return

        trade_date = trade.entry_time.date().isoformat()
        record = self.daily_pnl_repo.get_or_create(trade_date)

        # Update statistics
        update_data = {
            "trades": record.trades + 1,
            "gross_pnl": record.gross_pnl + (trade.pnl or 0),
            "net_pnl": record.net_pnl + (trade.pnl or 0),  # Assuming fees included
            "capital_end": capital,
        }

        if trade.pnl and trade.pnl > 0:
            update_data["wins"] = record.wins + 1
        else:
            update_data["losses"] = record.losses + 1

        # Calculate win rate
        total = update_data.get("trades", record.trades)
        wins = update_data.get("wins", record.wins)
        update_data["win_rate"] = wins / total if total > 0 else 0

        # Set capital_start if first trade of the day
        if record.trades == 0:
            update_data["capital_start"] = capital - (trade.pnl or 0)

        self.daily_pnl_repo.update(trade_date, update_data)
        logger.info(f"Daily PnL updated for {trade_date}")
