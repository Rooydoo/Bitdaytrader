"""Japanese cryptocurrency tax calculator.

日本の仮想通貨税制:
- 雑所得として総合課税
- 所得税: 5%〜45%（累進課税）
- 住民税: 10%
- 損失繰越不可（暗号資産は翌年繰越できない）
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from loguru import logger


@dataclass
class TaxBracket:
    """日本の所得税率ブラケット."""

    min_income: int  # 円
    max_income: int | None  # 円（Noneは上限なし）
    rate: float  # 税率（0-1）
    deduction: int  # 控除額（円）


# 2024年現在の所得税率
INCOME_TAX_BRACKETS: list[TaxBracket] = [
    TaxBracket(0, 1_950_000, 0.05, 0),
    TaxBracket(1_950_000, 3_300_000, 0.10, 97_500),
    TaxBracket(3_300_000, 6_950_000, 0.20, 427_500),
    TaxBracket(6_950_000, 9_000_000, 0.23, 636_000),
    TaxBracket(9_000_000, 18_000_000, 0.33, 1_536_000),
    TaxBracket(18_000_000, 40_000_000, 0.40, 2_796_000),
    TaxBracket(40_000_000, None, 0.45, 4_796_000),
]

# 住民税率（一律）
RESIDENT_TAX_RATE = 0.10


@dataclass
class TradeRecord:
    """取引記録."""

    trade_id: str
    timestamp: datetime
    symbol: str
    side: Literal["BUY", "SELL"]
    price: float
    size: float
    pnl: float  # 実現損益
    fees: float = 0.0


@dataclass
class TaxReport:
    """税金レポート."""

    year: int
    total_profit: float  # 総利益
    total_loss: float  # 総損失
    net_income: float  # 純所得（利益 - 損失）
    taxable_income: float  # 課税所得（損失は相殺可能）
    income_tax: float  # 所得税
    resident_tax: float  # 住民税
    total_tax: float  # 合計税額
    effective_rate: float  # 実効税率
    after_tax_profit: float  # 税引後利益
    trade_count: int  # 取引回数
    win_count: int  # 勝ち取引数
    loss_count: int  # 負け取引数

    def to_dict(self) -> dict:
        """辞書に変換."""
        return {
            "year": self.year,
            "total_profit": self.total_profit,
            "total_loss": self.total_loss,
            "net_income": self.net_income,
            "taxable_income": self.taxable_income,
            "income_tax": self.income_tax,
            "resident_tax": self.resident_tax,
            "total_tax": self.total_tax,
            "effective_rate": self.effective_rate,
            "after_tax_profit": self.after_tax_profit,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
        }


@dataclass
class TaxLossHarvestingOpportunity:
    """損出し機会."""

    symbol: str
    current_price: float
    entry_price: float
    unrealized_loss: float
    position_size: float
    estimated_tax_savings: float
    recommendation: str


class TaxCalculator:
    """日本の仮想通貨税金計算機."""

    def __init__(self, other_income: float = 0.0):
        """
        初期化.

        Args:
            other_income: 仮想通貨以外の雑所得（円）
        """
        self.other_income = other_income
        self._trades: list[TradeRecord] = []
        self._yearly_summary: dict[int, dict] = {}

    def add_trade(self, trade: TradeRecord) -> None:
        """取引を追加."""
        self._trades.append(trade)

        # 年次サマリーを更新
        year = trade.timestamp.year
        if year not in self._yearly_summary:
            self._yearly_summary[year] = {
                "profit": 0.0,
                "loss": 0.0,
                "fees": 0.0,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
            }

        summary = self._yearly_summary[year]
        summary["trade_count"] += 1
        summary["fees"] += trade.fees

        if trade.pnl > 0:
            summary["profit"] += trade.pnl
            summary["win_count"] += 1
        else:
            summary["loss"] += abs(trade.pnl)
            summary["loss_count"] += 1

    def calculate_income_tax(self, taxable_income: float) -> float:
        """
        所得税を計算.

        Args:
            taxable_income: 課税所得（円）

        Returns:
            所得税額（円）
        """
        if taxable_income <= 0:
            return 0.0

        for bracket in INCOME_TAX_BRACKETS:
            if bracket.max_income is None or taxable_income <= bracket.max_income:
                tax = taxable_income * bracket.rate - bracket.deduction
                return max(0.0, tax)

        # 最高税率を適用
        last_bracket = INCOME_TAX_BRACKETS[-1]
        return taxable_income * last_bracket.rate - last_bracket.deduction

    def calculate_resident_tax(self, taxable_income: float) -> float:
        """
        住民税を計算.

        Args:
            taxable_income: 課税所得（円）

        Returns:
            住民税額（円）
        """
        if taxable_income <= 0:
            return 0.0
        return taxable_income * RESIDENT_TAX_RATE

    def get_effective_rate(self, taxable_income: float) -> float:
        """
        実効税率を取得.

        Args:
            taxable_income: 課税所得（円）

        Returns:
            実効税率（0-1）
        """
        if taxable_income <= 0:
            return 0.0

        income_tax = self.calculate_income_tax(taxable_income)
        resident_tax = self.calculate_resident_tax(taxable_income)
        total_tax = income_tax + resident_tax

        return total_tax / taxable_income

    def generate_report(self, year: int) -> TaxReport:
        """
        年次税金レポートを生成.

        Args:
            year: 対象年

        Returns:
            税金レポート
        """
        summary = self._yearly_summary.get(year, {
            "profit": 0.0,
            "loss": 0.0,
            "fees": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
        })

        total_profit = summary["profit"]
        total_loss = summary["loss"]
        fees = summary["fees"]

        # 純所得 = 利益 - 損失 - 手数料
        net_income = total_profit - total_loss - fees

        # 課税所得（仮想通貨の損失は同年の雑所得内でのみ相殺可能）
        # 他の雑所得がある場合は合算
        taxable_income = max(0.0, net_income + self.other_income)

        # 税金計算
        income_tax = self.calculate_income_tax(taxable_income)
        resident_tax = self.calculate_resident_tax(taxable_income)
        total_tax = income_tax + resident_tax

        # 実効税率
        effective_rate = total_tax / taxable_income if taxable_income > 0 else 0.0

        # 税引後利益（仮想通貨分のみ）
        crypto_tax = total_tax * (net_income / taxable_income) if taxable_income > 0 else 0.0
        after_tax_profit = net_income - crypto_tax

        return TaxReport(
            year=year,
            total_profit=total_profit,
            total_loss=total_loss,
            net_income=net_income,
            taxable_income=taxable_income,
            income_tax=income_tax,
            resident_tax=resident_tax,
            total_tax=total_tax,
            effective_rate=effective_rate,
            after_tax_profit=after_tax_profit,
            trade_count=summary["trade_count"],
            win_count=summary["win_count"],
            loss_count=summary["loss_count"],
        )

    def simulate_annual_return(
        self,
        initial_capital: float,
        monthly_return: float,
        monthly_trades: int,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> dict:
        """
        年間リターンをシミュレーション（税引後）.

        Args:
            initial_capital: 初期資本（円）
            monthly_return: 月間リターン（0.15 = 15%）
            monthly_trades: 月間取引数
            win_rate: 勝率（0.55 = 55%）
            avg_win_loss_ratio: 平均利益/損失比（1.5 = 1.5:1）

        Returns:
            シミュレーション結果
        """
        annual_trades = monthly_trades * 12
        wins = int(annual_trades * win_rate)
        losses = annual_trades - wins

        # 年間リターン（複利）
        annual_return = (1 + monthly_return) ** 12 - 1
        gross_profit = initial_capital * annual_return

        # 利益と損失の内訳を推定
        # 勝ちトレードの平均利益 = 負けトレードの平均損失 × R比
        # total_profit = wins × avg_win
        # total_loss = losses × avg_loss
        # net = total_profit - total_loss
        if wins > 0 and losses > 0:
            # E[trade] = win_rate × avg_win - loss_rate × avg_loss = gross_profit / trades
            avg_profit_per_trade = gross_profit / annual_trades
            # avg_win × win_rate - avg_loss × (1 - win_rate) = avg_profit_per_trade
            # avg_win = avg_loss × R
            # avg_loss × R × win_rate - avg_loss × (1 - win_rate) = avg_profit_per_trade
            # avg_loss × (R × win_rate - (1 - win_rate)) = avg_profit_per_trade
            denominator = avg_win_loss_ratio * win_rate - (1 - win_rate)
            if denominator > 0:
                avg_loss = avg_profit_per_trade / denominator
                avg_win = avg_loss * avg_win_loss_ratio
            else:
                avg_loss = abs(gross_profit) / losses if losses > 0 else 0
                avg_win = 0
        else:
            avg_win = gross_profit / wins if wins > 0 else 0
            avg_loss = 0

        total_profit = wins * avg_win if avg_win > 0 else max(0, gross_profit)
        total_loss = losses * avg_loss if avg_loss > 0 else max(0, -gross_profit)

        # 税金計算
        net_income = total_profit - total_loss
        taxable_income = max(0.0, net_income + self.other_income)
        income_tax = self.calculate_income_tax(taxable_income)
        resident_tax = self.calculate_resident_tax(taxable_income)
        total_tax = income_tax + resident_tax

        # 仮想通貨分の税金
        crypto_tax_ratio = net_income / taxable_income if taxable_income > 0 else 0
        crypto_tax = total_tax * crypto_tax_ratio

        after_tax_profit = net_income - crypto_tax
        effective_rate = crypto_tax / net_income if net_income > 0 else 0

        return {
            "initial_capital": initial_capital,
            "gross_profit": gross_profit,
            "annual_return_pct": annual_return * 100,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_income": net_income,
            "taxable_income": taxable_income,
            "income_tax": income_tax,
            "resident_tax": resident_tax,
            "total_tax": total_tax,
            "crypto_tax": crypto_tax,
            "after_tax_profit": after_tax_profit,
            "after_tax_return_pct": (after_tax_profit / initial_capital) * 100,
            "effective_rate": effective_rate * 100,
            "annual_trades": annual_trades,
            "wins": wins,
            "losses": losses,
        }

    def check_tax_loss_harvesting(
        self,
        positions: list[dict],
        current_year_profit: float,
    ) -> list[TaxLossHarvestingOpportunity]:
        """
        損出し（Tax Loss Harvesting）の機会をチェック.

        年末に含み損ポジションを決済して、課税所得を減らす戦略。
        注意: 日本の仮想通貨税制では損失繰越不可のため、同年内でのみ有効。

        Args:
            positions: 保有ポジション一覧 [{symbol, entry_price, current_price, size}]
            current_year_profit: 今年の確定利益

        Returns:
            損出し機会のリスト
        """
        opportunities = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0)
            size = pos.get("size", 0)

            unrealized_pnl = (current_price - entry_price) * size

            # 含み損の場合のみ
            if unrealized_pnl < 0:
                unrealized_loss = abs(unrealized_pnl)

                # 税金削減額を計算
                # 現在の課税所得での税率
                current_rate = self.get_effective_rate(current_year_profit + self.other_income)
                # 損出し後の課税所得での税率
                new_taxable = max(0, current_year_profit - unrealized_loss + self.other_income)
                new_rate = self.get_effective_rate(new_taxable)

                # 税金削減額
                current_tax = (current_year_profit + self.other_income) * current_rate
                new_tax = new_taxable * new_rate
                tax_savings = current_tax - new_tax

                # 推奨判定
                if tax_savings > 0 and current_year_profit > 0:
                    if unrealized_loss > current_year_profit * 0.1:  # 利益の10%以上の損失
                        recommendation = "強く推奨: 大きな税金削減効果あり"
                    else:
                        recommendation = "検討推奨: 税金削減効果あり"
                else:
                    recommendation = "不要: 税金削減効果なし"

                opportunities.append(TaxLossHarvestingOpportunity(
                    symbol=symbol,
                    current_price=current_price,
                    entry_price=entry_price,
                    unrealized_loss=unrealized_loss,
                    position_size=size,
                    estimated_tax_savings=tax_savings,
                    recommendation=recommendation,
                ))

        # 税金削減額の大きい順にソート
        opportunities.sort(key=lambda x: x.estimated_tax_savings, reverse=True)

        return opportunities

    def get_breakeven_win_rate(
        self,
        avg_win_loss_ratio: float,
        risk_per_trade: float,
        monthly_trades: int,
    ) -> dict:
        """
        税引後で損益分岐となる勝率を計算.

        Args:
            avg_win_loss_ratio: 平均利益/損失比
            risk_per_trade: 1トレードあたりリスク（0.02 = 2%）
            monthly_trades: 月間取引数

        Returns:
            損益分岐点情報
        """
        # 税率別の損益分岐勝率を計算
        results = {}

        for bracket in INCOME_TAX_BRACKETS:
            rate = bracket.rate + RESIDENT_TAX_RATE  # 所得税 + 住民税

            # 税引前での損益分岐勝率
            # E = win_rate × avg_win - (1 - win_rate) × avg_loss = 0
            # win_rate × R - (1 - win_rate) = 0
            # win_rate × R - 1 + win_rate = 0
            # win_rate × (R + 1) = 1
            # win_rate = 1 / (R + 1)
            pretax_breakeven = 1 / (avg_win_loss_ratio + 1)

            # 税引後での損益分岐（利益に課税されるため、より高い勝率が必要）
            # 税引後期待値 = win_rate × avg_win × (1 - tax) - (1 - win_rate) × avg_loss = 0
            # ただし損失は税控除にならない場合を想定
            # 厳密には同年の利益と相殺可能だが、保守的に計算
            posttax_breakeven = 1 / (avg_win_loss_ratio * (1 - rate) + 1)

            bracket_name = f"~{bracket.max_income // 10000 if bracket.max_income else '∞'}万円"
            results[bracket_name] = {
                "tax_rate": rate * 100,
                "pretax_breakeven": pretax_breakeven * 100,
                "posttax_breakeven": posttax_breakeven * 100,
            }

        return {
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "risk_per_trade": risk_per_trade * 100,
            "monthly_trades": monthly_trades,
            "breakeven_by_bracket": results,
            "note": "損益分岐勝率は税率によって変わります。高所得ほど高い勝率が必要です。",
        }

    def format_report(self, report: TaxReport) -> str:
        """レポートをフォーマット."""
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 {report.year}年 税金レポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 取引実績
├ 取引回数: {report.trade_count}回
├ 勝ち: {report.win_count}回 ({report.win_count/report.trade_count*100:.1f}%)
└ 負け: {report.loss_count}回 ({report.loss_count/report.trade_count*100:.1f}%)

💰 損益
├ 総利益: ¥{report.total_profit:,.0f}
├ 総損失: ¥{report.total_loss:,.0f}
└ 純利益: ¥{report.net_income:,.0f}

🏛️ 税金
├ 課税所得: ¥{report.taxable_income:,.0f}
├ 所得税: ¥{report.income_tax:,.0f}
├ 住民税: ¥{report.resident_tax:,.0f}
├ 合計税額: ¥{report.total_tax:,.0f}
└ 実効税率: {report.effective_rate*100:.1f}%

✅ 税引後利益: ¥{report.after_tax_profit:,.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
