"""Core Meta AI Agent implementation."""

import asyncio
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.action import ActionExecutor, ExecutionSummary
from src.agent.claude_client import ClaudeClient
from src.agent.decision import AgentDecision, AutonomyLevel
from src.agent.memory import AgentMemory, SignalOutcome
from src.agent.perception import AgentContext, PerceptionModule
from src.agent.schedule import Scheduler, TaskFrequency, DEFAULT_TASKS
from src.utils.timezone import now_jst


class MetaAgent:
    """
    Autonomous Meta AI Agent for trading system oversight.

    Responsibilities:
    - Monitor trading system health and performance
    - Verify signal predictions against actual outcomes
    - Adjust parameters and features based on analysis
    - Conduct daily reviews and generate reports
    - Handle emergency situations
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8088",
        anthropic_api_key: str | None = None,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
        db_path: str = "data/agent_memory.db",
        check_interval: int = 60,  # seconds
    ) -> None:
        """
        Initialize Meta AI Agent.

        Args:
            api_base_url: Trading bot API URL
            anthropic_api_key: Anthropic API key for Claude
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            db_path: Path to agent memory database
            check_interval: Interval between state checks (seconds)
        """
        self.api_base_url = api_base_url
        self.check_interval = check_interval

        # Initialize components
        self.claude = ClaudeClient(api_key=anthropic_api_key)
        self.memory = AgentMemory(db_path=db_path)
        self.perception = PerceptionModule(api_base_url=api_base_url)
        self.executor = ActionExecutor(
            api_base_url=api_base_url,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            memory=self.memory,
        )
        self.scheduler = Scheduler()

        # State tracking
        self._running = False
        self._last_decision_time: datetime | None = None
        self._last_context: AgentContext | None = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # Setup scheduled tasks
        self._setup_scheduled_tasks()

        logger.info("Meta AI Agent initialized")

    def _setup_scheduled_tasks(self) -> None:
        """Setup default scheduled tasks."""

        # Market check (every minute)
        self.scheduler.add_task(
            name="market_check",
            task_func=self._task_market_check,
            frequency=TaskFrequency.INTERVAL,
            interval=timedelta(minutes=1),
        )

        # Signal verification (every 15 minutes)
        self.scheduler.add_task(
            name="signal_verification",
            task_func=self._task_signal_verification,
            frequency=TaskFrequency.INTERVAL,
            interval=timedelta(minutes=15),
        )

        # Performance snapshot (every hour)
        self.scheduler.add_task(
            name="performance_snapshot",
            task_func=self._task_performance_snapshot,
            frequency=TaskFrequency.INTERVAL,
            interval=timedelta(hours=1),
        )

        # Daily review at 21:00 JST
        self.scheduler.add_task(
            name="daily_review",
            task_func=self._task_daily_review,
            frequency=TaskFrequency.DAILY,
            run_time=time(21, 0),
        )

        # Morning preparation at 08:00 JST
        self.scheduler.add_task(
            name="morning_prep",
            task_func=self._task_morning_prep,
            frequency=TaskFrequency.DAILY,
            run_time=time(8, 0),
        )

        # Weekly summary on Sunday 20:00 JST
        self.scheduler.add_task(
            name="weekly_summary",
            task_func=self._task_weekly_summary,
            frequency=TaskFrequency.WEEKLY,
            run_time=time(20, 0),
            run_day=6,  # Sunday
        )

        logger.info("Scheduled tasks configured")

    async def run(self) -> None:
        """
        Main agent loop.
        Runs continuously, checking state and executing tasks.
        """
        self._running = True
        logger.info("Meta AI Agent starting main loop")

        # Send startup notification
        await self.executor._send_telegram(
            "🤖 Meta AI Agent 起動\n"
            f"監視間隔: {self.check_interval}秒\n"
            f"API: {self.api_base_url}"
        )

        try:
            while self._running:
                try:
                    # 0. Check for manual triggers (from API/Telegram)
                    await self._check_triggers()

                    # 1. Gather current context
                    context = await self.perception.get_context()
                    self._last_context = context

                    # 2. Update status file for API/Telegram
                    await self._update_status_file()

                    # 3. Check if attention is needed
                    if context.needs_attention():
                        logger.info("Context requires attention, running decision cycle")
                        await self._decision_cycle(context)

                    # 4. Run scheduled tasks
                    await self.scheduler.check_and_run()

                    # Reset error counter on success
                    self._consecutive_errors = 0

                except Exception as e:
                    self._consecutive_errors += 1
                    logger.error(f"Error in main loop: {e}")

                    if self._consecutive_errors >= self._max_consecutive_errors:
                        logger.critical(
                            f"Too many consecutive errors ({self._consecutive_errors}), "
                            "sending alert and pausing"
                        )
                        await self.executor._send_telegram(
                            f"🚨 Meta Agent エラー多発\n"
                            f"連続エラー数: {self._consecutive_errors}\n"
                            f"最新エラー: {e}\n\n"
                            "確認してください。"
                        )
                        # Wait longer before retrying
                        await asyncio.sleep(300)  # 5 minutes
                        self._consecutive_errors = 0

                # Wait before next cycle
                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Agent main loop cancelled")
        finally:
            self._running = False
            await self._cleanup()

    async def stop(self) -> None:
        """Stop the agent."""
        logger.info("Stopping Meta AI Agent")
        self._running = False

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        await self.perception.close()
        await self.executor.close()
        logger.info("Agent resources cleaned up")

    async def _decision_cycle(self, context: AgentContext) -> None:
        """
        Run a full decision cycle.

        1. Build prompt from context
        2. Ask Claude for decision
        3. Execute decided actions
        4. Record decision and results
        """
        # Get memory summary for context
        memory_summary = self.memory.get_decision_history_summary(limit=10)

        # Ask Claude for decision
        decision = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                self.claude.analyze_and_decide(
                    context_prompt=context.to_prompt(),
                    memory_summary=memory_summary,
                )
            )
        )

        # Log decision
        logger.info(
            f"Decision made: {len(decision.actions)} actions, "
            f"confidence={decision.confidence:.2f}"
        )

        if not decision.actions:
            logger.debug("No actions decided")
            return

        # Execute actions based on autonomy level
        results = await self.executor.execute_actions(decision.actions)

        # Record decision and results
        self.memory.record_decision(
            context_summary=self._summarize_context(context),
            decision=decision,
            results=[r.to_dict() for r in results.results],
            success=results.overall_success,
        )

        self._last_decision_time = now_jst()

    def _summarize_context(self, context: AgentContext) -> str:
        """Create a brief summary of context for storage."""
        parts = []

        if context.market:
            parts.append(f"BTC: ¥{context.market.current_price:,.0f}")

        if context.performance:
            parts.append(f"日次PnL: ¥{context.performance.daily_pnl:+,.0f}")

        if context.system_health:
            parts.append(f"システム: {context.system_health.status}")

        parts.append(f"シグナル: {len(context.recent_signals)}件")
        parts.append(f"取引: {len(context.recent_trades)}件")

        return " | ".join(parts)

    # ==================== Scheduled Tasks ====================

    async def _task_market_check(self) -> None:
        """Quick market state check for anomalies."""
        context = await self.perception.get_context()

        # Check for rapid price movement
        if context.market:
            if abs(context.market.price_change_1h) >= 0.03:  # 3% in 1 hour
                logger.warning(
                    f"Rapid price movement detected: {context.market.price_change_1h:+.2%}"
                )
                # This will trigger attention in the next cycle

    async def _task_signal_verification(self) -> None:
        """Verify recent signals against actual outcomes."""
        logger.info("Running signal verification")

        # Get signals from the last hour that need verification
        signals = await self.perception.get_recent_signals(hours=2)

        verified_count = 0
        for signal in signals:
            # Skip if already verified or too recent
            if signal.outcome is not None:
                continue

            # Check if enough time has passed (1 hour after signal)
            time_since_signal = now_jst() - signal.timestamp
            if time_since_signal < timedelta(hours=1):
                continue

            # Get actual price movement
            actual_move = await self.perception.calculate_price_move(
                symbol="BTC_JPY",
                start_time=signal.timestamp,
                end_time=signal.timestamp + timedelta(hours=1),
            )

            if actual_move is None:
                continue

            # Determine if prediction was correct
            # LONG prediction is correct if price went up by threshold (0.3%)
            # SHORT prediction is correct if price went down by threshold
            threshold = 0.003  # 0.3%
            was_correct = False

            if signal.direction == "LONG":
                was_correct = actual_move >= threshold
            else:  # SHORT
                was_correct = actual_move <= -threshold

            # Record outcome
            outcome = SignalOutcome(
                signal_id=signal.id,
                timestamp=signal.timestamp,
                direction=signal.direction,
                confidence=signal.confidence,
                price_at_signal=signal.price,
                price_after_1h=signal.price * (1 + actual_move),
                actual_move=actual_move,
                was_correct=was_correct,
                analysis="",
                feature_insights=[],
                suggestions=[],
            )

            self.memory.record_signal_outcome(outcome)
            verified_count += 1

        if verified_count > 0:
            logger.info(f"Verified {verified_count} signals")

            # Get stats and report if accuracy is concerning
            stats = self.memory.get_signal_accuracy_stats(days=1)
            if stats["evaluated"] >= 5 and stats["accuracy"] < 0.4:
                await self.executor._send_telegram(
                    f"⚠️ シグナル精度低下\n"
                    f"過去24時間の正解率: {stats['accuracy']:.1%}\n"
                    f"検証数: {stats['evaluated']}\n"
                    f"LONG: {stats['long_accuracy']:.1%}\n"
                    f"SHORT: {stats['short_accuracy']:.1%}"
                )

    async def _task_performance_snapshot(self) -> None:
        """Take a performance snapshot."""
        context = await self.perception.get_context()

        if context.performance:
            logger.info(
                f"Performance snapshot: "
                f"capital=¥{context.performance.capital:,.0f}, "
                f"daily_pnl=¥{context.performance.daily_pnl:+,.0f}, "
                f"win_rate={context.performance.win_rate:.1%}"
            )

    async def _task_daily_review(self) -> None:
        """Run daily review (reflection meeting)."""
        logger.info("Starting daily review")

        # Gather data for review
        signals = await self.perception.get_recent_signals(hours=24)
        trades = await self.perception.get_recent_trades(hours=24)
        performance = await self.perception.get_performance_metrics()

        # Get signal accuracy stats
        signal_stats = self.memory.get_signal_accuracy_stats(days=1)

        # Run intervention analysis (missed opportunities, stop-loss timing, etc.)
        intervention_results = await self._analyze_interventions(trades, signals)

        # Build data for Claude analysis
        signals_data = [s.to_dict() for s in signals]
        trades_data = [t.to_dict() for t in trades]
        performance_data = performance.to_dict() if performance else {}

        # Get market summary
        market = await self.perception.get_market_state()
        market_summary = f"BTC: ¥{market.current_price:,.0f}" if market else "市場データなし"

        # Generate review with Claude (including intervention analysis)
        review_report = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                self.claude.generate_daily_review(
                    signals_data=signals_data,
                    trades_data=trades_data,
                    performance_data=performance_data,
                    market_summary=market_summary,
                    intervention_summary=intervention_results.get("summary", ""),
                )
            )
        )

        # Build intervention stats text
        intervention_text = ""
        if intervention_results["analyses"]:
            intervention_text = f"\n\n📊 介入分析: {len(intervention_results['analyses'])}件検出"
            if intervention_results.get("obvious_count", 0) > 0:
                intervention_text += f"\n  ⚠️ 明白な見逃し: {intervention_results['obvious_count']}件"

        # Send report
        await self.executor._send_telegram(
            f"📋 日次レビュー ({now_jst().strftime('%Y-%m-%d')})\n\n"
            f"シグナル統計:\n"
            f"- 検証数: {signal_stats['evaluated']}\n"
            f"- 正解率: {signal_stats['accuracy']:.1%}\n"
            f"- LONG: {signal_stats['long_accuracy']:.1%}\n"
            f"- SHORT: {signal_stats['short_accuracy']:.1%}"
            f"{intervention_text}\n\n"
            f"{review_report[:2800]}"  # Telegram limit (adjusted for intervention text)
        )

        logger.info("Daily review completed")

    async def _analyze_interventions(
        self,
        trades: list,
        signals: list,
    ) -> dict:
        """
        Analyze missed or delayed interventions.

        Evaluates:
        1. Stop-loss timing - Could we have exited earlier?
        2. Missed opportunities - Large moves without positions
        3. Threshold issues - Would lower thresholds have been profitable?

        Returns a dict with analysis results.
        """
        from src.agent.memory import InterventionAnalysis

        analyses = []
        obvious_count = 0

        # 1. Analyze stop-loss timing for losing trades
        for trade in trades:
            if not hasattr(trade, "pnl") or trade.pnl is None:
                continue

            # Only analyze closed losing trades
            if trade.pnl < 0 and hasattr(trade, "exit_price"):
                # Check if price moved significantly against position before SL hit
                analysis = await self._analyze_stop_loss_timing(trade)
                if analysis:
                    analyses.append(analysis)
                    self.memory.record_intervention_analysis(analysis)
                    if analysis.hindsight_difficulty == "obvious":
                        obvious_count += 1

        # 2. Detect large price moves without positions (missed opportunities)
        market_moves = await self._detect_significant_moves(hours=24)
        for move in market_moves:
            # Check if we had no position during this move
            had_position = self._had_position_during(move, trades)
            if not had_position and abs(move["change"]) >= 0.02:  # 2% move
                analysis = self._create_missed_opportunity_analysis(move, signals)
                if analysis:
                    analyses.append(analysis)
                    self.memory.record_intervention_analysis(analysis)
                    if analysis.hindsight_difficulty == "obvious":
                        obvious_count += 1

        # Build summary for Claude
        summary_parts = []
        if analyses:
            summary_parts.append(f"検出された介入分析: {len(analyses)}件")

            by_type = {}
            for a in analyses:
                by_type[a.analysis_type] = by_type.get(a.analysis_type, 0) + 1

            for t, count in by_type.items():
                type_label = {
                    "stop_loss_timing": "損切りタイミング",
                    "missed_opportunity": "機会損失",
                    "threshold_too_strict": "閾値問題",
                }.get(t, t)
                summary_parts.append(f"  - {type_label}: {count}件")

            if obvious_count > 0:
                summary_parts.append(f"\n※明白な見逃し: {obvious_count}件 (要改善)")

        return {
            "analyses": analyses,
            "obvious_count": obvious_count,
            "summary": "\n".join(summary_parts) if summary_parts else "特に問題なし",
        }

    async def _analyze_stop_loss_timing(self, trade) -> "InterventionAnalysis | None":
        """Analyze if stop-loss could have been triggered earlier."""
        from src.agent.memory import InterventionAnalysis

        # Get price history during trade
        try:
            price_history = await self.perception.get_price_history(
                symbol="BTC_JPY",
                start_time=trade.entry_time,
                end_time=trade.exit_time if hasattr(trade, "exit_time") else None,
            )
        except Exception:
            return None

        if not price_history:
            return None

        # Find if there was an earlier opportunity to exit with less loss
        entry_price = trade.entry_price
        exit_price = trade.exit_price if hasattr(trade, "exit_price") else entry_price
        actual_loss = trade.pnl

        # For LONG: find highest price after entry
        # For SHORT: find lowest price after entry
        best_exit_price = None
        best_exit_time = None

        for point in price_history:
            price = point.get("close", point.get("price"))
            if trade.side == "BUY":  # LONG
                if best_exit_price is None or price > best_exit_price:
                    best_exit_price = price
                    best_exit_time = point.get("timestamp")
            else:  # SHORT
                if best_exit_price is None or price < best_exit_price:
                    best_exit_price = price
                    best_exit_time = point.get("timestamp")

        if best_exit_price is None:
            return None

        # Calculate potential better outcome
        if trade.side == "BUY":
            potential_pnl = (best_exit_price - entry_price) * trade.size
        else:
            potential_pnl = (entry_price - best_exit_price) * trade.size

        # Only report if significant improvement was possible
        improvement = potential_pnl - actual_loss
        if improvement < abs(actual_loss) * 0.3:  # Less than 30% improvement
            return None

        # Determine hindsight difficulty
        # If the better exit was clearly signaled (e.g., RSI extreme), it's "obvious"
        # Otherwise, it's "moderate" or "difficult"
        hindsight_difficulty = "moderate"  # Default

        # If price reversed sharply (>1% in 15 min), it was probably predictable
        if abs(best_exit_price - exit_price) / exit_price > 0.01:
            hindsight_difficulty = "obvious" if improvement > abs(actual_loss) else "moderate"

        return InterventionAnalysis(
            id=None,
            timestamp=trade.exit_time if hasattr(trade, "exit_time") else now_jst(),
            analysis_type="stop_loss_timing",
            trade_id=trade.id if hasattr(trade, "id") else None,
            price_at_event=exit_price,
            optimal_action=f"¥{best_exit_price:,.0f}で決済",
            actual_action=f"¥{exit_price:,.0f}で損切り",
            potential_impact=improvement,
            hindsight_difficulty=hindsight_difficulty,
            contributing_factors=[
                f"最良決済価格: ¥{best_exit_price:,.0f}",
                f"実際の決済: ¥{exit_price:,.0f}",
                f"改善可能額: ¥{improvement:,.0f}",
            ],
            recommendation="ATR倍率の見直しまたは時間ベースの損切りルール検討",
            evaluated_by_llm=False,
        )

    async def _detect_significant_moves(self, hours: int = 24) -> list[dict]:
        """Detect significant price movements in the past N hours."""
        try:
            # Get hourly price data
            price_history = await self.perception.get_price_history(
                symbol="BTC_JPY",
                start_time=now_jst() - timedelta(hours=hours),
                interval="1h",
            )
        except Exception:
            return []

        if not price_history or len(price_history) < 2:
            return []

        moves = []
        for i in range(1, len(price_history)):
            prev = price_history[i - 1]
            curr = price_history[i]

            prev_price = prev.get("close", prev.get("price", 0))
            curr_price = curr.get("close", curr.get("price", 0))

            if prev_price == 0:
                continue

            change = (curr_price - prev_price) / prev_price

            if abs(change) >= 0.015:  # 1.5% or more
                moves.append({
                    "start_time": prev.get("timestamp"),
                    "end_time": curr.get("timestamp"),
                    "start_price": prev_price,
                    "end_price": curr_price,
                    "change": change,
                    "direction": "up" if change > 0 else "down",
                })

        return moves

    def _had_position_during(self, move: dict, trades: list) -> bool:
        """Check if we had a position during a price move."""
        move_start = move.get("start_time")
        move_end = move.get("end_time")

        if not move_start or not move_end:
            return False

        # Convert to datetime if string
        if isinstance(move_start, str):
            move_start = datetime.fromisoformat(move_start.replace("Z", "+00:00"))
        if isinstance(move_end, str):
            move_end = datetime.fromisoformat(move_end.replace("Z", "+00:00"))

        for trade in trades:
            trade_start = trade.entry_time if hasattr(trade, "entry_time") else None
            trade_end = trade.exit_time if hasattr(trade, "exit_time") else now_jst()

            if trade_start is None:
                continue

            # Check for overlap
            if trade_start <= move_end and trade_end >= move_start:
                return True

        return False

    def _create_missed_opportunity_analysis(
        self,
        move: dict,
        signals: list,
    ) -> "InterventionAnalysis | None":
        """Create analysis for a missed trading opportunity."""
        from src.agent.memory import InterventionAnalysis

        change = move["change"]
        direction = move["direction"]

        # Check if there was a signal that could have caught this move
        matching_signals = []
        for signal in signals:
            signal_time = signal.timestamp if hasattr(signal, "timestamp") else None
            if signal_time is None:
                continue

            move_start = move.get("start_time")
            if isinstance(move_start, str):
                move_start = datetime.fromisoformat(move_start.replace("Z", "+00:00"))

            # Signal within 2 hours before the move
            if signal_time < move_start and (move_start - signal_time).total_seconds() < 7200:
                signal_direction = signal.direction if hasattr(signal, "direction") else None
                if signal_direction:
                    expected = "LONG" if direction == "up" else "SHORT"
                    if signal_direction == expected:
                        matching_signals.append(signal)

        # Determine hindsight difficulty
        if matching_signals:
            # We had a signal but didn't trade - check confidence
            max_conf = max(s.confidence for s in matching_signals if hasattr(s, "confidence"))
            if max_conf >= 0.6:
                hindsight_difficulty = "obvious"  # High confidence signal, should have traded
            else:
                hindsight_difficulty = "moderate"  # Low confidence, understandable miss
            contributing_factors = [
                f"適切な方向のシグナルあり (信頼度: {max_conf:.1%})",
                f"価格変動: {change:+.2%}",
            ]
            recommendation = "信頼度閾値を下げることを検討"
        else:
            # No signal at all
            hindsight_difficulty = "difficult"  # No signal, hard to predict
            contributing_factors = [
                "シグナルなし",
                f"価格変動: {change:+.2%}",
            ]
            recommendation = "特徴量の追加または見直しを検討"

        # Calculate potential impact (rough estimate)
        move_pct = abs(change)
        # Assume 1% position size, so potential gain is move_pct * position
        potential_impact = move["start_price"] * 0.01 * move_pct  # Rough estimate

        return InterventionAnalysis(
            id=None,
            timestamp=datetime.fromisoformat(move["end_time"].replace("Z", "+00:00"))
            if isinstance(move["end_time"], str)
            else move["end_time"],
            analysis_type="missed_opportunity",
            trade_id=None,
            price_at_event=move["end_price"],
            optimal_action=f"{'LONG' if direction == 'up' else 'SHORT'}エントリー",
            actual_action="ノーポジション",
            potential_impact=potential_impact,
            hindsight_difficulty=hindsight_difficulty,
            contributing_factors=contributing_factors,
            recommendation=recommendation,
            evaluated_by_llm=False,
        )

    async def _task_morning_prep(self) -> None:
        """Morning preparation and status check."""
        logger.info("Running morning preparation")

        context = await self.perception.get_context()

        # Build status message
        lines = [f"🌅 おはようございます ({now_jst().strftime('%Y-%m-%d %H:%M')})\n"]

        if context.system_health:
            status_emoji = {
                "healthy": "✅",
                "degraded": "⚠️",
                "unhealthy": "🚨",
            }.get(context.system_health.status, "❓")
            lines.append(f"システム状態: {status_emoji} {context.system_health.status}")

            if context.system_health.emergency_stop_active:
                lines.append("⚠️ 緊急停止中")
            if context.system_health.long_stopped:
                lines.append("🔴 LONG停止中")
            if context.system_health.short_stopped:
                lines.append("🔴 SHORT停止中")

        if context.market:
            lines.append(f"\nBTC: ¥{context.market.current_price:,.0f}")
            lines.append(f"24h変動: {context.market.price_change_24h:+.2%}")

        if context.performance:
            lines.append(f"\n資本: ¥{context.performance.capital:,.0f}")
            lines.append(f"週間PnL: ¥{context.performance.weekly_pnl:+,.0f}")
            lines.append(f"月間PnL: ¥{context.performance.monthly_pnl:+,.0f}")

        if context.open_positions:
            lines.append(f"\nオープンポジション: {len(context.open_positions)}件")

        # Get upcoming scheduled tasks
        upcoming = self.scheduler.get_upcoming_tasks(hours=24)
        if upcoming:
            lines.append("\n本日のスケジュール:")
            for task in upcoming[:5]:
                time_str = datetime.fromisoformat(task["next_run"]).strftime("%H:%M")
                lines.append(f"- {time_str}: {task['name']}")

        await self.executor._send_telegram("\n".join(lines))

    async def _task_weekly_summary(self) -> None:
        """Generate weekly summary report."""
        logger.info("Generating weekly summary")

        # Get weekly statistics
        signal_stats = self.memory.get_signal_accuracy_stats(days=7)
        decision_patterns = self.memory.get_decision_patterns()
        param_history = self.memory.get_param_history(days=7)

        # Get performance
        performance = await self.perception.get_performance_metrics()

        lines = [
            f"📊 週次サマリー ({now_jst().strftime('%Y-%m-%d')})\n",
            "=== パフォーマンス ===",
        ]

        if performance:
            lines.extend([
                f"週間PnL: ¥{performance.weekly_pnl:+,.0f}",
                f"勝率: {performance.win_rate:.1%}",
                f"取引数: {performance.trades_count}回",
            ])

        lines.extend([
            "\n=== シグナル精度 ===",
            f"検証数: {signal_stats['evaluated']}",
            f"正解率: {signal_stats['accuracy']:.1%}",
            f"LONG: {signal_stats['long_accuracy']:.1%}",
            f"SHORT: {signal_stats['short_accuracy']:.1%}",
        ])

        lines.extend([
            "\n=== エージェント判断 ===",
            f"判断数: {decision_patterns.get('total_evaluated', 0)}",
            f"成功率: {decision_patterns.get('success_rate', 0):.1%}",
        ])

        if param_history:
            lines.append(f"\nパラメータ変更: {len(param_history)}件")

        if decision_patterns.get("recommendations"):
            lines.append("\n=== 改善提案 ===")
            for rec in decision_patterns["recommendations"][:3]:
                lines.append(f"- {rec}")

        await self.executor._send_telegram("\n".join(lines))
        logger.info("Weekly summary sent")

    # ==================== Trigger Handling ====================

    async def _check_triggers(self) -> None:
        """Check for and process manual triggers from API/Telegram."""
        trigger_path = Path("data/agent_triggers.json")
        if not trigger_path.exists():
            return

        try:
            with open(trigger_path) as f:
                triggers = json.load(f)

            if not triggers:
                return

            # Process pending triggers
            updated = False
            for trigger_name, trigger_data in list(triggers.items()):
                if trigger_data.get("status") != "pending":
                    continue

                logger.info(f"Processing trigger: {trigger_name}")
                source = trigger_data.get("source", "api")

                try:
                    if trigger_name == "daily_review":
                        await self._task_daily_review()
                        triggers[trigger_name]["status"] = "completed"
                        triggers[trigger_name]["completed_at"] = now_jst().isoformat()

                    elif trigger_name == "signal_verification":
                        await self._task_signal_verification()
                        triggers[trigger_name]["status"] = "completed"
                        triggers[trigger_name]["completed_at"] = now_jst().isoformat()

                    elif trigger_name == "emergency_analysis":
                        context = trigger_data.get("context", "")
                        await self._emergency_analysis(context)
                        triggers[trigger_name]["status"] = "completed"
                        triggers[trigger_name]["completed_at"] = now_jst().isoformat()

                    updated = True
                    logger.info(f"Trigger {trigger_name} completed (source: {source})")

                except Exception as e:
                    logger.error(f"Error processing trigger {trigger_name}: {e}")
                    triggers[trigger_name]["status"] = "failed"
                    triggers[trigger_name]["error"] = str(e)
                    updated = True

            # Save updated triggers
            if updated:
                with open(trigger_path, "w") as f:
                    json.dump(triggers, f, indent=2)

        except Exception as e:
            logger.error(f"Error checking triggers: {e}")

    async def _update_status_file(self) -> None:
        """Update agent status file for API/Telegram to read."""
        try:
            status_path = Path("data/agent_status.json")
            status_path.parent.mkdir(parents=True, exist_ok=True)

            # Get recent actions from memory
            recent_decisions = self.memory.get_decision_history_summary(limit=5)

            status = {
                "status": "running" if self._running else "stopped",
                "last_check": now_jst().isoformat(),
                "decisions_today": self._count_decisions_today(),
                "consecutive_errors": self._consecutive_errors,
                "recent_actions": self._format_recent_actions(recent_decisions),
            }

            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating status file: {e}")

    def _count_decisions_today(self) -> int:
        """Count decisions made today."""
        try:
            # Simple count from memory
            return self.memory.get_decision_count_today()
        except Exception:
            return 0

    def _format_recent_actions(self, decisions: str) -> list[dict]:
        """Format recent decisions as action list."""
        # This is a simplified version - could parse the summary string
        actions = []
        if self._last_decision_time:
            actions.append({
                "type": "decision",
                "summary": "最新の判断",
                "time": self._last_decision_time.isoformat(),
            })
        return actions

    async def _emergency_analysis(self, context: str = "") -> None:
        """
        Run emergency analysis.

        This is triggered manually when immediate analysis is needed.
        """
        logger.warning(f"Running emergency analysis: {context}")

        # Notify start
        await self.executor._send_telegram(
            f"🚨 <b>緊急分析開始</b>\n"
            f"コンテキスト: {context or 'なし'}\n"
            f"時刻: {now_jst().strftime('%H:%M:%S')}"
        )

        # Gather comprehensive context
        full_context = await self.perception.get_context()

        # Build emergency prompt
        emergency_prompt = f"""
緊急分析リクエスト

ユーザーからのコンテキスト: {context or '指定なし'}

現在の状況:
{full_context.to_prompt()}

この状況を即座に分析し、必要なアクションを決定してください。
緊急度の高い問題があれば、適切な対応を提案してください。
"""

        # Get Claude's analysis
        memory_summary = self.memory.get_decision_history_summary(limit=5)

        decision = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(
                self.claude.analyze_and_decide(
                    context_prompt=emergency_prompt,
                    memory_summary=memory_summary,
                )
            )
        )

        # Build response
        response_lines = [
            f"🔍 <b>緊急分析完了</b>",
            f"",
            f"<b>分析結果:</b>",
            f"{decision.reasoning[:1000]}",
            f"",
            f"<b>推奨アクション:</b> {len(decision.actions)}件",
        ]

        for action in decision.actions[:5]:
            response_lines.append(f"• {action.type}: {action.description}")

        if decision.actions:
            # Execute high-priority actions
            results = await self.executor.execute_actions(decision.actions)
            response_lines.append(f"\n<b>実行結果:</b> {'成功' if results.overall_success else '一部失敗'}")

        await self.executor._send_telegram("\n".join(response_lines))
        logger.info("Emergency analysis completed")

    # ==================== Public Methods ====================

    async def force_daily_review(self) -> None:
        """Manually trigger daily review."""
        await self._task_daily_review()

    async def force_signal_verification(self) -> None:
        """Manually trigger signal verification."""
        await self._task_signal_verification()

    async def force_emergency_analysis(self, context: str = "") -> None:
        """Manually trigger emergency analysis."""
        await self._emergency_analysis(context)

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "running": self._running,
            "last_decision_time": self._last_decision_time.isoformat() if self._last_decision_time else None,
            "consecutive_errors": self._consecutive_errors,
            "scheduled_tasks": self.scheduler.get_all_status(),
            "memory_stats": {
                "signal_accuracy_7d": self.memory.get_signal_accuracy_stats(days=7),
                "decision_patterns": self.memory.get_decision_patterns(),
            },
        }
