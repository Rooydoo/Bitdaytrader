"""Report generation module."""

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from src.database.models import (
    DailyPnL,
    DailyPnLRepository,
    Trade,
    TradeRepository,
    WalkForwardRepository,
    WalkForwardResult,
)
from src.telegram.bot import TelegramBot
from src.tax.calculator import TaxCalculator, TaxReport


class ReportGenerator:
    """Generate and send trading reports."""

    def __init__(
        self,
        trade_repo: TradeRepository,
        daily_pnl_repo: DailyPnLRepository,
        telegram_bot: TelegramBot,
        walkforward_repo: WalkForwardRepository | None = None,
    ) -> None:
        """
        Initialize report generator.

        Args:
            trade_repo: Trade repository
            daily_pnl_repo: Daily PnL repository
            telegram_bot: Telegram bot for sending reports
            walkforward_repo: Walk-forward results repository
        """
        self.trade_repo = trade_repo
        self.daily_pnl_repo = daily_pnl_repo
        self.telegram_bot = telegram_bot
        self.walkforward_repo = walkforward_repo

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

    async def generate_model_analysis_report(self) -> bool:
        """
        Generate and send model analysis report (bi-weekly).

        Returns:
            True if sent successfully
        """
        if not self.walkforward_repo:
            logger.warning("Walk-forward repository not configured")
            return False

        latest = self.walkforward_repo.get_latest()
        if not latest:
            logger.warning("No walk-forward results available")
            return False

        # Check for degradation
        degradation = self.walkforward_repo.check_degradation()
        warning = degradation.get("reason") if degradation.get("degraded") else None

        return await self.telegram_bot.send_model_analysis_report(
            model_version=latest.model_version,
            trained_at=latest.trained_at.strftime("%Y-%m-%d %H:%M"),
            test_accuracy=latest.test_accuracy_mean or 0,
            test_auc=latest.test_auc_mean or 0,
            backtest_win_rate=latest.backtest_win_rate or 0,
            backtest_return=latest.backtest_return_pct or 0,
            backtest_sharpe=latest.backtest_sharpe or 0,
            backtest_max_dd=latest.backtest_max_drawdown or 0,
            accuracy_gap=latest.accuracy_gap,
            is_overfit=latest.is_overfit,
            live_accuracy=latest.live_accuracy,
            live_predictions=latest.live_predictions,
            degradation_warning=warning,
        )

    def get_model_performance_summary(self) -> dict[str, Any]:
        """
        Get model performance summary for inclusion in other reports.

        Returns:
            Dict with model performance metrics
        """
        if not self.walkforward_repo:
            return {}

        latest = self.walkforward_repo.get_latest()
        if not latest:
            return {}

        return {
            "model_version": latest.model_version,
            "trained_at": latest.trained_at.isoformat() if latest.trained_at else None,
            "test_accuracy": latest.test_accuracy_mean,
            "test_auc": latest.test_auc_mean,
            "backtest_win_rate": latest.backtest_win_rate,
            "is_overfit": latest.is_overfit,
            "live_accuracy": latest.live_accuracy,
            "live_predictions": latest.live_predictions,
        }

    def generate_backtest_report_with_tax(
        self,
        initial_capital: float,
        final_capital: float,
        trades: list[Trade],
        start_date: str,
        end_date: str,
        other_income: float = 0.0,
    ) -> dict[str, Any]:
        """
        バックテスト結果レポートを生成（税引後リターン含む）.

        Args:
            initial_capital: 初期資本
            final_capital: 最終資本
            trades: 取引リスト
            start_date: 開始日
            end_date: 終了日
            other_income: 他の雑所得

        Returns:
            バックテスト結果レポート
        """
        if not trades:
            return {"error": "No trades to analyze"}

        # 基本統計
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl and t.pnl > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0

        # 損益
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        net_pnl = gross_profit - gross_loss

        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 平均利益/損失
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # リターン
        gross_return_pct = (final_capital - initial_capital) / initial_capital * 100

        # 税金計算
        tax_calc = TaxCalculator(other_income=other_income)
        year = int(start_date[:4])

        # 取引を税計算に追加
        from src.tax.calculator import TradeRecord
        for trade in trades:
            if trade.pnl is not None and trade.entry_time:
                tax_calc.add_trade(TradeRecord(
                    trade_id=str(trade.id),
                    timestamp=trade.entry_time,
                    symbol=trade.symbol,
                    side=trade.side,
                    price=trade.entry_price or 0,
                    size=trade.size,
                    pnl=trade.pnl,
                ))

        tax_report = tax_calc.generate_report(year)

        # 税引後リターン
        after_tax_return_pct = (tax_report.after_tax_profit / initial_capital) * 100

        # ドローダウン計算
        capitals = [initial_capital]
        for trade in sorted(trades, key=lambda t: t.entry_time or datetime.min):
            if trade.pnl:
                capitals.append(capitals[-1] + trade.pnl)

        max_drawdown = 0.0
        peak = capitals[0]
        for cap in capitals:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            max_drawdown = max(max_drawdown, dd)

        # 損益分岐点分析
        breakeven = tax_calc.get_breakeven_win_rate(
            avg_win_loss_ratio=avg_win_loss_ratio if avg_win_loss_ratio != float('inf') else 1.5,
            risk_per_trade=0.02,
            monthly_trades=total_trades,
        )

        return {
            "period": {
                "start": start_date,
                "end": end_date,
            },
            "capital": {
                "initial": initial_capital,
                "final": final_capital,
            },
            "trades": {
                "total": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate * 100,
            },
            "pnl": {
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "net_pnl": net_pnl,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "avg_win_loss_ratio": avg_win_loss_ratio,
            },
            "returns": {
                "gross_return_pct": gross_return_pct,
                "after_tax_return_pct": after_tax_return_pct,
                "max_drawdown_pct": max_drawdown * 100,
            },
            "tax": {
                "taxable_income": tax_report.taxable_income,
                "income_tax": tax_report.income_tax,
                "resident_tax": tax_report.resident_tax,
                "total_tax": tax_report.total_tax,
                "effective_rate_pct": tax_report.effective_rate * 100,
                "after_tax_profit": tax_report.after_tax_profit,
            },
            "breakeven_analysis": breakeven,
        }

    def format_backtest_report(self, report: dict[str, Any]) -> str:
        """バックテストレポートをフォーマット."""
        if "error" in report:
            return f"Error: {report['error']}"

        period = report["period"]
        capital = report["capital"]
        trades = report["trades"]
        pnl = report["pnl"]
        returns = report["returns"]
        tax = report["tax"]

        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 バックテスト結果レポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 期間: {period['start']} ~ {period['end']}

💰 資本
├ 初期: ¥{capital['initial']:,.0f}
└ 最終: ¥{capital['final']:,.0f}

📈 取引実績
├ 総取引数: {trades['total']}回
├ 勝ち: {trades['wins']}回
├ 負け: {trades['losses']}回
└ 勝率: {trades['win_rate']:.1f}%

💵 損益
├ 総利益: ¥{pnl['gross_profit']:,.0f}
├ 総損失: ¥{pnl['gross_loss']:,.0f}
├ 純利益: ¥{pnl['net_pnl']:,.0f}
├ Profit Factor: {pnl['profit_factor']:.2f}
├ 平均勝ち: ¥{pnl['avg_win']:,.0f}
├ 平均負け: ¥{pnl['avg_loss']:,.0f}
└ 勝ち/負け比: {pnl['avg_win_loss_ratio']:.2f}:1

📊 リターン
├ 税引前: {returns['gross_return_pct']:+.1f}%
├ 税引後: {returns['after_tax_return_pct']:+.1f}%
└ 最大DD: {returns['max_drawdown_pct']:.1f}%

🏛️ 税金（年間）
├ 課税所得: ¥{tax['taxable_income']:,.0f}
├ 所得税: ¥{tax['income_tax']:,.0f}
├ 住民税: ¥{tax['resident_tax']:,.0f}
├ 合計税額: ¥{tax['total_tax']:,.0f}
├ 実効税率: {tax['effective_rate_pct']:.1f}%
└ 税引後利益: ¥{tax['after_tax_profit']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    async def generate_yearly_tax_report(self, capital: float) -> bool:
        """
        年次税金レポートを生成・送信.

        Args:
            capital: 現在の資本

        Returns:
            送信成功したかどうか
        """
        year = date.today().year
        year_start = date(year, 1, 1)
        year_end = date.today()

        # 今年の取引を取得
        trades = self.trade_repo.get_trades_by_period(
            year_start.isoformat(),
            year_end.isoformat(),
        )

        if not trades:
            logger.info(f"No trades for year {year}")
            return False

        # 年初資本を推定
        daily_records = self.daily_pnl_repo.get_by_period(
            year_start.isoformat(),
            year_end.isoformat(),
        )
        initial_capital = daily_records[0].capital_start if daily_records else capital

        # レポート生成
        report = self.generate_backtest_report_with_tax(
            initial_capital=initial_capital or capital,
            final_capital=capital,
            trades=trades,
            start_date=year_start.isoformat(),
            end_date=year_end.isoformat(),
        )

        # Telegramに送信
        formatted = self.format_backtest_report(report)
        return await self.telegram_bot.send_message(formatted)
