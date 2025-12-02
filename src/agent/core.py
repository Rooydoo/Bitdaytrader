"""Core Meta AI Agent implementation."""

import asyncio
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.action import ActionExecutor, ExecutionSummary
from src.agent.claude_client import ClaudeClient
from src.agent.decision import AgentAction, AgentDecision, ActionType, AutonomyLevel
from src.agent.long_term_memory import LongTermMemory, ConfidenceLevel
from src.agent.memory import AgentMemory, SignalOutcome
from src.agent.perception import AgentContext, PerceptionModule
from src.agent.schedule import Scheduler, TaskFrequency, DEFAULT_TASKS
from src.features.registry import FeatureRegistry
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
        feature_registry_path: str = "data/feature_registry.json",
        long_term_memory_dir: str = "data/memory",
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
            feature_registry_path: Path to feature registry config
            long_term_memory_dir: Path to long-term memory directory
            check_interval: Interval between state checks (seconds)
        """
        self.api_base_url = api_base_url
        self.check_interval = check_interval

        # Initialize components
        self.claude = ClaudeClient(api_key=anthropic_api_key)
        self.memory = AgentMemory(db_path=db_path)
        self.long_term_memory = LongTermMemory(memory_dir=long_term_memory_dir)
        self.perception = PerceptionModule(api_base_url=api_base_url)
        self.feature_registry = FeatureRegistry(config_path=feature_registry_path)
        self.executor = ActionExecutor(
            api_base_url=api_base_url,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            memory=self.memory,
            feature_registry=self.feature_registry,
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

        # Feature optimization on Saturday 19:00 JST (weekly)
        self.scheduler.add_task(
            name="feature_optimization",
            task_func=self._task_feature_optimization,
            frequency=TaskFrequency.WEEKLY,
            run_time=time(19, 0),
            run_day=5,  # Saturday
        )

        # Database maintenance on Sunday 03:00 JST (weekly, low traffic time)
        self.scheduler.add_task(
            name="database_maintenance",
            task_func=self._task_database_maintenance,
            frequency=TaskFrequency.WEEKLY,
            run_time=time(3, 0),
            run_day=6,  # Sunday
        )

        # Memory validation on Sunday 04:00 JST (after DB maintenance)
        self.scheduler.add_task(
            name="memory_validation",
            task_func=self._task_memory_validation,
            frequency=TaskFrequency.WEEKLY,
            run_time=time(4, 0),
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
        """Stop the agent gracefully."""
        logger.info("Stopping Meta AI Agent...")
        self._running = False
        # Wait a moment for current iteration to finish
        await asyncio.sleep(1)
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Cleanup resources and save state."""
        logger.info("Cleaning up agent resources...")

        # Close HTTP sessions
        await self.perception.close()
        await self.executor.close()

        # Save feature registry state
        self.feature_registry.save_config()
        logger.info("Feature registry saved")

        # Close memory database properly
        self.memory.close()

        logger.info("Agent resources cleaned up")

    async def _decision_cycle(self, context: AgentContext) -> None:
        """
        Run a full decision cycle.

        1. Build prompt from context
        2. Ask Claude for decision (with short-term and long-term memory)
        3. Execute decided actions
        4. Record decision and results
        """
        # Get short-term memory (recent decisions)
        memory_summary = self.memory.get_decision_history_summary(limit=10)

        # Get long-term memory (learned insights and rules)
        long_term_context = self.long_term_memory.get_context_for_prompt()

        # Ask Claude for decision
        decision = await self.claude.analyze_and_decide(
            context_prompt=context.to_prompt(),
            memory_summary=memory_summary,
            long_term_memory=long_term_context,
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
        review_report = await self.claude.generate_daily_review(
            signals_data=signals_data,
            trades_data=trades_data,
            performance_data=performance_data,
            market_summary=market_summary,
            intervention_summary=intervention_results.get("summary", ""),
        )

        # Build intervention stats text
        intervention_text = ""
        if intervention_results["analyses"]:
            intervention_text = f"\n\n📊 介入分析: {len(intervention_results['analyses'])}件検出"
            if intervention_results.get("obvious_count", 0) > 0:
                intervention_text += f"\n  ⚠️ 明白な見逃し: {intervention_results['obvious_count']}件"

        # Build long-term memory stats text
        ltm_stats = self.long_term_memory.get_stats()
        ltm_text = f"\n\n🧠 長期記憶:\n"
        ltm_text += f"- 有効な洞察: {ltm_stats['insights']['active']}件"
        if ltm_stats['insights']['under_review'] > 0:
            ltm_text += f" (レビュー中: {ltm_stats['insights']['under_review']}件)"
        ltm_text += f"\n- 有効なルール: {ltm_stats['rules']['active']}件"
        if ltm_stats['rules']['under_review'] > 0:
            ltm_text += f" (レビュー中: {ltm_stats['rules']['under_review']}件)"

        # Send report
        await self.executor._send_telegram(
            f"📋 日次レビュー ({now_jst().strftime('%Y-%m-%d')})\n\n"
            f"シグナル統計:\n"
            f"- 検証数: {signal_stats['evaluated']}\n"
            f"- 正解率: {signal_stats['accuracy']:.1%}\n"
            f"- LONG: {signal_stats['long_accuracy']:.1%}\n"
            f"- SHORT: {signal_stats['short_accuracy']:.1%}"
            f"{intervention_text}"
            f"{ltm_text}\n\n"
            f"{review_report[:2600]}"  # Telegram limit (adjusted for additional text)
        )

        # Extract insights from review and save to long-term memory
        await self._extract_and_save_insights(
            review_report=review_report,
            performance_data=performance_data,
            signal_stats=signal_stats,
        )

        logger.info("Daily review completed")

    async def _extract_and_save_insights(
        self,
        review_report: str,
        performance_data: dict,
        signal_stats: dict,
    ) -> None:
        """
        Extract insights from daily review and save to long-term memory.

        Args:
            review_report: The daily review report text
            performance_data: Performance metrics
            signal_stats: Signal accuracy statistics
        """
        logger.info("Extracting insights from daily review")

        try:
            # Extract insights using Claude
            extracted = await self.claude.extract_insights_from_review(
                daily_review=review_report,
                performance_data=performance_data,
                signal_accuracy=signal_stats,
            )

            insights_added = 0
            rules_added = 0
            events_added = 0

            # Save insights
            for insight_data in extracted.get("insights", []):
                try:
                    confidence = ConfidenceLevel(insight_data.get("confidence", "low"))
                    self.long_term_memory.add_insight(
                        category=insight_data.get("category", "その他"),
                        title=insight_data.get("title", ""),
                        content=insight_data.get("content", ""),
                        evidence=insight_data.get("evidence", []),
                        conditions=insight_data.get("conditions", []),
                        confidence=confidence,
                    )
                    insights_added += 1
                except Exception as e:
                    logger.warning(f"Failed to save insight: {e}")

            # Save rules
            for rule_data in extracted.get("rules", []):
                try:
                    confidence = ConfidenceLevel(rule_data.get("confidence", "low"))
                    self.long_term_memory.add_rule(
                        name=rule_data.get("name", ""),
                        rule_type=rule_data.get("type", "conditional"),
                        content=rule_data.get("content", ""),
                        origin=rule_data.get("origin", "日次レビューから抽出"),
                        confidence=confidence,
                    )
                    rules_added += 1
                except Exception as e:
                    logger.warning(f"Failed to save rule: {e}")

            # Save events
            for event_data in extracted.get("events", []):
                try:
                    self.long_term_memory.add_event(
                        name=event_data.get("name", ""),
                        category=event_data.get("category", "other"),
                        severity=event_data.get("severity", "medium"),
                        impact=event_data.get("impact", ""),
                        situation=event_data.get("situation", ""),
                        response="日次レビューで記録",
                        result="",
                        lessons=event_data.get("lessons", []),
                    )
                    events_added += 1
                except Exception as e:
                    logger.warning(f"Failed to save event: {e}")

            if insights_added or rules_added or events_added:
                logger.info(
                    f"Long-term memory updated: "
                    f"{insights_added} insights, {rules_added} rules, {events_added} events"
                )

                # Notify about significant updates
                if insights_added + rules_added >= 2:
                    await self.executor._send_telegram(
                        f"🧠 長期記憶を更新しました\n"
                        f"- 新しい洞察: {insights_added}件\n"
                        f"- 新しいルール: {rules_added}件\n"
                        f"- 新しいイベント: {events_added}件"
                    )
            else:
                reason = extracted.get("no_new_insights_reason", "特に新しい学習事項なし")
                logger.info(f"No new insights extracted: {reason}")

        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")

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

        # Long-term memory statistics
        ltm_stats = self.long_term_memory.get_stats()
        lines.extend([
            "\n=== 長期記憶 ===",
            f"洞察: {ltm_stats['insights']['active']}件 (有効)",
        ])
        if ltm_stats['insights']['under_review'] > 0:
            lines.append(f"  └ レビュー中: {ltm_stats['insights']['under_review']}件")
        if ltm_stats['insights']['deprecated'] > 0:
            lines.append(f"  └ 淘汰済み: {ltm_stats['insights']['deprecated']}件")

        lines.append(f"ルール: {ltm_stats['rules']['active']}件 (有効)")
        if ltm_stats['rules']['under_review'] > 0:
            lines.append(f"  └ レビュー中: {ltm_stats['rules']['under_review']}件")
        if ltm_stats['rules']['deprecated'] > 0:
            lines.append(f"  └ 淘汰済み: {ltm_stats['rules']['deprecated']}件")

        lines.append(f"イベント履歴: {ltm_stats['events']['total']}件")

        # Show active high-confidence insights
        high_conf_insights = [
            i for i in self.long_term_memory.get_active_insights()
            if i.confidence.value == "high"
        ]
        if high_conf_insights:
            lines.append("\n📌 高信頼度の洞察:")
            for insight in high_conf_insights[:3]:
                lines.append(f"- [{insight.category}] {insight.title}")
                lines.append(f"  (検証{insight.verification_count}回, 成功率{insight.success_rate:.0%})")

        if decision_patterns.get("recommendations"):
            lines.append("\n=== 改善提案 ===")
            for rec in decision_patterns["recommendations"][:3]:
                lines.append(f"- {rec}")

        await self.executor._send_telegram("\n".join(lines))
        logger.info("Weekly summary sent")

    async def _task_feature_optimization(self) -> None:
        """
        Weekly feature optimization task.
        Analyzes feature performance and recommends changes.
        """
        logger.info("Running feature optimization analysis")

        try:
            # Gather data for analysis
            feature_summary = self.feature_registry.get_summary()
            signal_stats = self.memory.get_signal_accuracy_stats(days=7)
            trades = await self.perception.get_recent_trades(hours=168)  # 7 days
            trades_data = [t.to_dict() for t in trades] if trades else []

            # Get model performance if available
            performance = await self.perception.get_performance_metrics()
            model_performance = performance.to_dict() if performance else {}

            # Ask Claude for feature optimization analysis
            optimization_result = await self.claude.analyze_feature_optimization(
                feature_registry_summary=feature_summary,
                model_performance=model_performance,
                signal_accuracy=signal_stats,
                recent_trades=trades_data,
            )

            # Process recommendations
            recommendations = optimization_result.get("feature_recommendations", [])
            executed_actions = []
            proposed_actions = []

            for rec in recommendations[:3]:  # Limit to 3 changes per week
                feature_name = rec.get("feature_name")
                action_type = rec.get("action")
                autonomy = rec.get("autonomy_level", "propose")
                reason = rec.get("reason", "")

                if action_type == "enable":
                    action = AgentAction(
                        action_type=ActionType.FEATURE_TOGGLE,
                        detail=f"特徴量 '{feature_name}' を有効化",
                        autonomy_level=AutonomyLevel(autonomy),
                        reasoning=reason,
                        parameters={"feature_name": feature_name, "enabled": True},
                    )
                elif action_type == "disable":
                    action = AgentAction(
                        action_type=ActionType.FEATURE_TOGGLE,
                        detail=f"特徴量 '{feature_name}' を無効化",
                        autonomy_level=AutonomyLevel(autonomy),
                        reasoning=reason,
                        parameters={"feature_name": feature_name, "enabled": False},
                    )
                elif action_type == "update_importance":
                    action = AgentAction(
                        action_type=ActionType.FEATURE_IMPORTANCE_UPDATE,
                        detail=f"特徴量 '{feature_name}' の重要度更新",
                        autonomy_level=AutonomyLevel.AUTO_EXECUTE,
                        reasoning=reason,
                        parameters={"importance": {feature_name: rec.get("importance_score", 0.5)}},
                    )
                else:
                    continue

                # Execute or propose based on autonomy level
                if autonomy in ["auto_execute", "auto_execute_report"]:
                    result = await self.executor.execute_actions([action])
                    executed_actions.append((feature_name, action_type, result.overall_success))
                else:
                    proposed_actions.append((feature_name, action_type, reason))

            # Build and send report
            report_lines = [
                f"🔧 <b>週次特徴量最適化レポート</b>",
                f"",
                f"<b>分析結果:</b>",
                f"{optimization_result.get('analysis', 'N/A')[:500]}",
                f"",
            ]

            if executed_actions:
                report_lines.append("<b>実行済み変更:</b>")
                for name, action, success in executed_actions:
                    status = "✅" if success else "❌"
                    report_lines.append(f"{status} {name}: {action}")

            if proposed_actions:
                report_lines.append("\n<b>提案（承認待ち）:</b>")
                for name, action, reason in proposed_actions:
                    report_lines.append(f"• {name}: {action}")
                    report_lines.append(f"  理由: {reason[:100]}")

            if optimization_result.get("retrain_recommended"):
                report_lines.append(f"\n⚠️ <b>再学習推奨:</b>")
                report_lines.append(optimization_result.get("retrain_reason", "")[:200])

            extended_suggestions = optimization_result.get("extended_features_to_consider", [])
            if extended_suggestions:
                report_lines.append(f"\n💡 <b>今後検討すべき特徴量:</b>")
                for feat in extended_suggestions[:3]:
                    report_lines.append(f"• {feat}")

            await self.executor._send_telegram("\n".join(report_lines))
            logger.info(f"Feature optimization completed: {len(executed_actions)} executed, {len(proposed_actions)} proposed")

        except Exception as e:
            logger.error(f"Feature optimization failed: {e}")
            await self.executor._send_telegram(
                f"❌ 特徴量最適化でエラーが発生しました: {e}"
            )

    async def _task_database_maintenance(self) -> None:
        """
        Weekly database maintenance task.
        Cleans up old records and optimizes database.
        """
        logger.info("Running database maintenance")

        try:
            # Get stats before cleanup
            stats_before = self.memory.get_database_stats()

            # Clean up old records (3 years default, 5 years for param_history)
            deleted = self.memory.cleanup_old_records()

            # Vacuum to reclaim space
            self.memory.vacuum()

            # Get stats after cleanup
            stats_after = self.memory.get_database_stats()

            # Calculate savings
            size_saved = stats_before["file_size_mb"] - stats_after["file_size_mb"]
            total_deleted = sum(deleted.values())

            # Send report
            lines = [
                "🗄️ データベースメンテナンス完了",
                "",
                f"削除レコード数: {total_deleted}件",
            ]

            if deleted:
                for table, count in deleted.items():
                    if count > 0:
                        lines.append(f"  - {table}: {count}件")

            lines.extend([
                "",
                f"DB サイズ: {stats_before['file_size_mb']:.2f}MB → {stats_after['file_size_mb']:.2f}MB",
                f"解放容量: {size_saved:.2f}MB",
                "",
                "現在のレコード数:",
            ])

            for table, count in stats_after["tables"].items():
                lines.append(f"  - {table}: {count}件")

            await self.executor._send_telegram("\n".join(lines))
            logger.info(f"Database maintenance completed: {total_deleted} records deleted")

        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            await self.executor._send_telegram(
                f"❌ DBメンテナンスでエラー: {e}"
            )

    async def _task_memory_validation(self) -> None:
        """
        Weekly long-term memory validation task.
        Validates insights and rules against recent data.
        Deprecates items that are no longer valid.
        """
        logger.info("Running long-term memory validation")

        try:
            # 1. Run automatic validation (age-based deprecation)
            validation_results = self.long_term_memory.run_validation()

            # 2. Get items needing LLM validation
            items_to_validate = []
            for insight in self.long_term_memory.get_active_insights():
                items_to_validate.append({
                    "type": "insight",
                    "id": insight.id,
                    "category": insight.category,
                    "title": insight.title,
                    "content": insight.content,
                    "success_rate": f"{insight.success_rate:.0%}",
                    "verification_count": insight.verification_count,
                })
            for rule in self.long_term_memory.get_active_rules():
                items_to_validate.append({
                    "type": "rule",
                    "id": rule.id,
                    "name": rule.name,
                    "content": rule.content,
                    "success_rate": f"{rule.success_rate:.0%}",
                    "application_count": rule.application_count,
                })

            # 3. Get recent performance for context
            signal_stats = self.memory.get_signal_accuracy_stats(days=7)
            intervention_stats = self.memory.get_intervention_stats(days=7)

            recent_performance = {
                "signal_accuracy_7d": signal_stats,
                "intervention_stats_7d": intervention_stats,
            }

            # 4. Validate with Claude (if there are items to validate)
            llm_validations = {}
            if items_to_validate:
                llm_results = await self.claude.validate_memory_items(
                    items_to_validate=items_to_validate[:10],  # Limit to 10
                    recent_performance=recent_performance,
                )

                # Process LLM validation results
                for validation in llm_results.get("validations", []):
                    item_id = validation.get("id")
                    item_type = validation.get("type")
                    success = validation.get("success")
                    recommendation = validation.get("recommendation")

                    if item_type == "insight" and item_id:
                        if success is not None:
                            self.long_term_memory.verify_insight(
                                item_id,
                                success=success,
                                notes=validation.get("notes", ""),
                            )
                        llm_validations[f"insight:{item_id}"] = recommendation
                    elif item_type == "rule" and item_id:
                        if success is not None:
                            self.long_term_memory.apply_rule(
                                item_id,
                                success=success,
                                context=validation.get("notes", ""),
                            )
                        llm_validations[f"rule:{item_id}"] = recommendation

            # 5. Build and send report
            stats = self.long_term_memory.get_stats()
            deprecate_count = sum(
                1 for rec in llm_validations.values()
                if rec == "deprecate"
            )

            lines = [
                "🧠 長期記憶の検証完了",
                "",
                "**記憶の状態:**",
                f"- 洞察: {stats['insights']['active']}件 (レビュー中: {stats['insights']['under_review']}件)",
                f"- ルール: {stats['rules']['active']}件 (レビュー中: {stats['rules']['under_review']}件)",
                f"- イベント: {stats['events']['total']}件",
            ]

            if validation_results["items_needing_attention"]:
                lines.append("")
                lines.append("**要注意項目:**")
                for item in validation_results["items_needing_attention"][:5]:
                    item_name = item.get("title") or item.get("name", "不明")
                    lines.append(f"- {item['type']}: {item_name} ({item['reason']})")

            if deprecate_count > 0:
                lines.append("")
                lines.append(f"⚠️ 淘汰推奨: {deprecate_count}件")
                lines.append("（過学習を防ぐため、有効性の低い項目を自動的に無効化しました）")

            await self.executor._send_telegram("\n".join(lines))
            logger.info(
                f"Memory validation completed: "
                f"{stats['insights']['active']} insights, {stats['rules']['active']} rules active"
            )

            # Save weekly reflection
            now = now_jst()
            week_start = now - timedelta(days=7)

            # Build memory updates list
            memory_updates = []
            if validation_results["insights_reviewed"] > 0:
                memory_updates.append(f"洞察{validation_results['insights_reviewed']}件をレビュー中に移行")
            if validation_results["insights_deprecated"] > 0:
                memory_updates.append(f"洞察{validation_results['insights_deprecated']}件を淘汰")
            if validation_results["rules_reviewed"] > 0:
                memory_updates.append(f"ルール{validation_results['rules_reviewed']}件をレビュー中に移行")
            if validation_results["rules_deprecated"] > 0:
                memory_updates.append(f"ルール{validation_results['rules_deprecated']}件を淘汰")
            if deprecate_count > 0:
                memory_updates.append(f"LLM検証で{deprecate_count}件に淘汰を推奨")

            # Determine good things and improvements
            good_things = []
            improvements_needed = []

            if signal_stats.get("accuracy", 0) >= 0.6:
                good_things.append(f"シグナル精度が良好 ({signal_stats['accuracy']:.0%})")
            else:
                improvements_needed.append(f"シグナル精度の改善が必要 ({signal_stats['accuracy']:.0%})")

            if stats['insights']['active'] > 0:
                good_things.append(f"{stats['insights']['active']}件の有効な洞察を維持")
            if stats['rules']['active'] > 0:
                good_things.append(f"{stats['rules']['active']}件の有効なルールを維持")

            if deprecate_count > 0:
                improvements_needed.append("過学習の兆候あり、一部の記憶を淘汰")

            self.long_term_memory.add_weekly_reflection(
                start_date=week_start,
                end_date=now,
                performance_summary={
                    "signal_accuracy": signal_stats.get("accuracy", 0),
                    "intervention_success": intervention_stats.get("total", 0) > 0,
                    "major_mistakes": intervention_stats.get("obvious_misses", 0),
                },
                good_things=good_things if good_things else ["特になし"],
                improvements_needed=improvements_needed if improvements_needed else ["特になし"],
                focus_points=["継続的な記憶の検証", "過学習の防止"],
                memory_updates=memory_updates if memory_updates else ["変更なし"],
            )
            logger.info("Weekly reflection saved to long-term memory")

        except Exception as e:
            logger.error(f"Memory validation failed: {e}")
            await self.executor._send_telegram(
                f"❌ 記憶検証でエラー: {e}"
            )

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

                    elif trigger_name == "feature_optimization":
                        await self._task_feature_optimization()
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

        decision = await self.claude.analyze_and_decide(
            context_prompt=emergency_prompt,
            memory_summary=memory_summary,
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
