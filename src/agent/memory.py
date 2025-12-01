"""Memory module for Meta AI Agent - stores and retrieves decision history."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.decision import AgentDecision, AgentAction, ActionType, AutonomyLevel
from src.utils.timezone import now_jst, JST


@dataclass
class DecisionRecord:
    """A record of an agent decision and its outcome."""

    id: int | None
    timestamp: datetime
    context_summary: str
    decision_type: str
    actions: list[dict]
    results: list[dict]
    success: bool
    outcome_evaluated: bool = False
    outcome_score: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "context_summary": self.context_summary,
            "decision_type": self.decision_type,
            "actions": self.actions,
            "results": self.results,
            "success": self.success,
            "outcome_evaluated": self.outcome_evaluated,
            "outcome_score": self.outcome_score,
            "notes": self.notes,
        }


@dataclass
class SignalOutcome:
    """Record of a signal's outcome after verification."""

    signal_id: int
    timestamp: datetime
    direction: str
    confidence: float
    price_at_signal: float
    price_after_1h: float | None
    actual_move: float | None
    was_correct: bool | None
    analysis: str
    feature_insights: list[str]
    suggestions: list[str]


@dataclass
class InterventionAnalysis:
    """Record of a missed or delayed intervention analysis."""

    id: int | None
    timestamp: datetime
    analysis_type: str  # "stop_loss_timing", "missed_opportunity", "threshold_too_strict"
    trade_id: int | None  # Related trade if applicable
    price_at_event: float
    optimal_action: str  # What should have been done
    actual_action: str  # What was actually done (or "none")
    potential_impact: float  # Estimated PnL difference
    hindsight_difficulty: str  # "obvious", "moderate", "difficult" - was it predictable?
    contributing_factors: list[str]
    recommendation: str
    evaluated_by_llm: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "analysis_type": self.analysis_type,
            "trade_id": self.trade_id,
            "price_at_event": self.price_at_event,
            "optimal_action": self.optimal_action,
            "actual_action": self.actual_action,
            "potential_impact": self.potential_impact,
            "hindsight_difficulty": self.hindsight_difficulty,
            "contributing_factors": self.contributing_factors,
            "recommendation": self.recommendation,
            "evaluated_by_llm": self.evaluated_by_llm,
        }


@dataclass
class DecisionPattern:
    """A pattern identified from decision history."""

    pattern_type: str  # "successful", "failed"
    context_patterns: list[str]
    action_types: list[str]
    frequency: int
    success_rate: float
    avg_outcome_score: float


class AgentMemory:
    """
    Persistent memory for the agent.
    Stores decisions, outcomes, and learned patterns.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        context_summary TEXT NOT NULL,
        decision_type TEXT NOT NULL,
        actions TEXT NOT NULL,
        results TEXT NOT NULL,
        success INTEGER NOT NULL,
        outcome_evaluated INTEGER DEFAULT 0,
        outcome_score REAL,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS signal_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        direction TEXT NOT NULL,
        confidence REAL NOT NULL,
        price_at_signal REAL NOT NULL,
        price_after_1h REAL,
        actual_move REAL,
        was_correct INTEGER,
        analysis TEXT,
        feature_insights TEXT,
        suggestions TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS feature_importance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL,
        importance_score REAL NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        last_updated TEXT NOT NULL,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS param_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        param_name TEXT NOT NULL,
        old_value REAL NOT NULL,
        new_value REAL NOT NULL,
        change_reason TEXT,
        changed_by TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        outcome_score REAL
    );

    CREATE TABLE IF NOT EXISTS intervention_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        trade_id INTEGER,
        price_at_event REAL NOT NULL,
        optimal_action TEXT NOT NULL,
        actual_action TEXT NOT NULL,
        potential_impact REAL NOT NULL,
        hindsight_difficulty TEXT NOT NULL,
        contributing_factors TEXT,
        recommendation TEXT,
        evaluated_by_llm INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_signal_outcomes_timestamp ON signal_outcomes(timestamp);
    CREATE INDEX IF NOT EXISTS idx_signal_outcomes_signal_id ON signal_outcomes(signal_id);
    CREATE INDEX IF NOT EXISTS idx_intervention_timestamp ON intervention_analysis(timestamp);
    """

    def __init__(self, db_path: str = "data/agent_memory.db") -> None:
        """
        Initialize agent memory.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Agent memory initialized: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== Decision Recording ====================

    def record_decision(
        self,
        context_summary: str,
        decision: AgentDecision,
        results: list[dict],
        success: bool,
    ) -> int:
        """
        Record a decision and its immediate results.

        Args:
            context_summary: Summary of the context when decision was made
            decision: The decision made
            results: Results of actions taken
            success: Whether actions were successfully executed

        Returns:
            ID of the recorded decision
        """
        decision_type = self._categorize_decision(decision)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO decisions
                (timestamp, context_summary, decision_type, actions, results, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.timestamp.isoformat(),
                    context_summary,
                    decision_type,
                    json.dumps([a.to_dict() for a in decision.actions]),
                    json.dumps(results),
                    1 if success else 0,
                )
            )
            conn.commit()
            decision_id = cursor.lastrowid

        logger.debug(f"Recorded decision {decision_id}: {decision_type}")
        return decision_id

    def _categorize_decision(self, decision: AgentDecision) -> str:
        """Categorize a decision by its primary action type."""
        if not decision.actions:
            return "no_action"

        # Get the most significant action
        priority = {
            ActionType.EMERGENCY_STOP: 0,
            ActionType.DIRECTION_STOP: 1,
            ActionType.MODEL_RETRAIN_TRIGGER: 2,
            ActionType.PARAM_ADJUSTMENT: 3,
            ActionType.FEATURE_TOGGLE: 4,
            ActionType.DAILY_REVIEW: 5,
            ActionType.SIGNAL_VERIFICATION: 6,
            ActionType.ALERT_CRITICAL: 7,
            ActionType.ALERT_WARNING: 8,
            ActionType.ALERT_INFO: 9,
            ActionType.NO_ACTION: 10,
        }

        sorted_actions = sorted(
            decision.actions,
            key=lambda a: priority.get(a.action_type, 99)
        )

        return sorted_actions[0].action_type.value

    def update_decision_outcome(
        self,
        decision_id: int,
        outcome_score: float,
        notes: str = "",
    ) -> None:
        """
        Update a decision with its outcome score after evaluation.

        Args:
            decision_id: ID of the decision
            outcome_score: Score from 0 to 1 (1 = fully successful)
            notes: Optional notes about the outcome
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE decisions
                SET outcome_evaluated = 1, outcome_score = ?, notes = ?
                WHERE id = ?
                """,
                (outcome_score, notes, decision_id)
            )
            conn.commit()

        logger.debug(f"Updated decision {decision_id} outcome: {outcome_score:.2f}")

    def get_recent_decisions(self, limit: int = 50) -> list[DecisionRecord]:
        """Get recent decisions."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM decisions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [self._row_to_decision_record(row) for row in rows]

    def get_unevaluated_decisions(self, min_age_hours: int = 24) -> list[DecisionRecord]:
        """Get decisions that haven't been evaluated yet and are old enough."""
        cutoff = (now_jst() - timedelta(hours=min_age_hours)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM decisions
                WHERE outcome_evaluated = 0 AND timestamp < ?
                ORDER BY timestamp ASC
                """,
                (cutoff,)
            ).fetchall()

        return [self._row_to_decision_record(row) for row in rows]

    def _row_to_decision_record(self, row: sqlite3.Row) -> DecisionRecord:
        """Convert database row to DecisionRecord."""
        return DecisionRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            context_summary=row["context_summary"],
            decision_type=row["decision_type"],
            actions=json.loads(row["actions"]),
            results=json.loads(row["results"]),
            success=bool(row["success"]),
            outcome_evaluated=bool(row["outcome_evaluated"]),
            outcome_score=row["outcome_score"],
            notes=row["notes"] or "",
        )

    # ==================== Signal Outcome Recording ====================

    def record_signal_outcome(self, outcome: SignalOutcome) -> int:
        """
        Record a signal's outcome after verification.

        Args:
            outcome: Signal outcome data

        Returns:
            ID of the recorded outcome
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signal_outcomes
                (signal_id, timestamp, direction, confidence, price_at_signal,
                 price_after_1h, actual_move, was_correct, analysis,
                 feature_insights, suggestions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.signal_id,
                    outcome.timestamp.isoformat(),
                    outcome.direction,
                    outcome.confidence,
                    outcome.price_at_signal,
                    outcome.price_after_1h,
                    outcome.actual_move,
                    1 if outcome.was_correct else 0 if outcome.was_correct is not None else None,
                    outcome.analysis,
                    json.dumps(outcome.feature_insights),
                    json.dumps(outcome.suggestions),
                    now_jst().isoformat(),
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_signal_outcomes(
        self,
        days: int = 7,
        direction: str | None = None,
    ) -> list[SignalOutcome]:
        """Get signal outcomes for analysis."""
        cutoff = (now_jst() - timedelta(days=days)).isoformat()

        query = """
            SELECT * FROM signal_outcomes
            WHERE timestamp >= ?
        """
        params = [cutoff]

        if direction:
            query += " AND direction = ?"
            params.append(direction)

        query += " ORDER BY timestamp DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            SignalOutcome(
                signal_id=row["signal_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                direction=row["direction"],
                confidence=row["confidence"],
                price_at_signal=row["price_at_signal"],
                price_after_1h=row["price_after_1h"],
                actual_move=row["actual_move"],
                was_correct=bool(row["was_correct"]) if row["was_correct"] is not None else None,
                analysis=row["analysis"] or "",
                feature_insights=json.loads(row["feature_insights"]) if row["feature_insights"] else [],
                suggestions=json.loads(row["suggestions"]) if row["suggestions"] else [],
            )
            for row in rows
        ]

    def get_signal_accuracy_stats(self, days: int = 7) -> dict:
        """Get signal accuracy statistics."""
        outcomes = self.get_signal_outcomes(days=days)

        if not outcomes:
            return {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "long_accuracy": 0.0,
                "short_accuracy": 0.0,
                "avg_confidence_correct": 0.0,
                "avg_confidence_incorrect": 0.0,
            }

        total = len(outcomes)
        evaluated = [o for o in outcomes if o.was_correct is not None]
        correct = [o for o in evaluated if o.was_correct]
        incorrect = [o for o in evaluated if not o.was_correct]

        long_outcomes = [o for o in evaluated if o.direction == "LONG"]
        short_outcomes = [o for o in evaluated if o.direction == "SHORT"]

        long_correct = sum(1 for o in long_outcomes if o.was_correct)
        short_correct = sum(1 for o in short_outcomes if o.was_correct)

        return {
            "total": total,
            "evaluated": len(evaluated),
            "correct": len(correct),
            "incorrect": len(incorrect),
            "accuracy": len(correct) / len(evaluated) if evaluated else 0.0,
            "long_accuracy": long_correct / len(long_outcomes) if long_outcomes else 0.0,
            "short_accuracy": short_correct / len(short_outcomes) if short_outcomes else 0.0,
            "avg_confidence_correct": sum(o.confidence for o in correct) / len(correct) if correct else 0.0,
            "avg_confidence_incorrect": sum(o.confidence for o in incorrect) / len(incorrect) if incorrect else 0.0,
        }

    # ==================== Feature Importance ====================

    def update_feature_importance(
        self,
        feature_name: str,
        importance_score: float,
        enabled: bool = True,
        notes: str = "",
    ) -> None:
        """Update feature importance score."""
        with self._get_connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM feature_importance WHERE feature_name = ?",
                (feature_name,)
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE feature_importance
                    SET importance_score = ?, enabled = ?, last_updated = ?, notes = ?
                    WHERE feature_name = ?
                    """,
                    (importance_score, 1 if enabled else 0, now_jst().isoformat(), notes, feature_name)
                )
            else:
                conn.execute(
                    """
                    INSERT INTO feature_importance
                    (feature_name, importance_score, enabled, last_updated, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (feature_name, importance_score, 1 if enabled else 0, now_jst().isoformat(), notes)
                )
            conn.commit()

    def get_feature_importance(self) -> dict[str, dict]:
        """Get all feature importance scores."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM feature_importance ORDER BY importance_score DESC"
            ).fetchall()

        return {
            row["feature_name"]: {
                "importance_score": row["importance_score"],
                "enabled": bool(row["enabled"]),
                "last_updated": row["last_updated"],
                "notes": row["notes"],
            }
            for row in rows
        }

    def toggle_feature(self, feature_name: str, enabled: bool) -> bool:
        """Toggle a feature on/off."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE feature_importance
                SET enabled = ?, last_updated = ?
                WHERE feature_name = ?
                """,
                (1 if enabled else 0, now_jst().isoformat(), feature_name)
            )
            conn.commit()
            return cursor.rowcount > 0

    # ==================== Parameter History ====================

    def record_param_change(
        self,
        param_name: str,
        old_value: float,
        new_value: float,
        change_reason: str,
        changed_by: str = "agent",
    ) -> int:
        """Record a parameter change."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO param_history
                (param_name, old_value, new_value, change_reason, changed_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (param_name, old_value, new_value, change_reason, changed_by, now_jst().isoformat())
            )
            conn.commit()
            return cursor.lastrowid

    def get_param_history(self, param_name: str | None = None, days: int = 30) -> list[dict]:
        """Get parameter change history."""
        cutoff = (now_jst() - timedelta(days=days)).isoformat()

        if param_name:
            query = """
                SELECT * FROM param_history
                WHERE param_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            params = (param_name, cutoff)
        else:
            query = """
                SELECT * FROM param_history
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """
            params = (cutoff,)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    # ==================== Pattern Analysis ====================

    def get_decision_patterns(self) -> dict:
        """Analyze decision patterns from history."""
        decisions = self.get_recent_decisions(limit=200)
        evaluated = [d for d in decisions if d.outcome_evaluated and d.outcome_score is not None]

        if not evaluated:
            return {
                "successful_patterns": [],
                "failed_patterns": [],
                "recommendations": [],
            }

        # Categorize by success
        successful = [d for d in evaluated if d.outcome_score >= 0.7]
        failed = [d for d in evaluated if d.outcome_score < 0.3]

        # Analyze patterns
        successful_types = {}
        for d in successful:
            for action in d.actions:
                t = action.get("type", "unknown")
                successful_types[t] = successful_types.get(t, 0) + 1

        failed_types = {}
        for d in failed:
            for action in d.actions:
                t = action.get("type", "unknown")
                failed_types[t] = failed_types.get(t, 0) + 1

        # Generate recommendations
        recommendations = []

        # Find action types that fail more than succeed
        all_types = set(successful_types.keys()) | set(failed_types.keys())
        for t in all_types:
            s_count = successful_types.get(t, 0)
            f_count = failed_types.get(t, 0)
            total = s_count + f_count

            if total >= 5:  # Need enough samples
                success_rate = s_count / total
                if success_rate < 0.3:
                    recommendations.append(
                        f"アクション '{t}' の成功率が低い ({success_rate:.1%})。"
                        f"条件を見直すか、人間の承認を必須にすることを推奨。"
                    )

        return {
            "successful_patterns": [
                {"type": t, "count": c}
                for t, c in sorted(successful_types.items(), key=lambda x: -x[1])
            ],
            "failed_patterns": [
                {"type": t, "count": c}
                for t, c in sorted(failed_types.items(), key=lambda x: -x[1])
            ],
            "recommendations": recommendations,
            "total_evaluated": len(evaluated),
            "success_rate": len(successful) / len(evaluated) if evaluated else 0.0,
        }

    def get_decision_history_summary(self, limit: int = 10) -> str:
        """Get a summary of recent decisions for Claude prompt."""
        decisions = self.get_recent_decisions(limit=limit)

        if not decisions:
            return "過去の判断履歴はありません。"

        lines = []
        for d in decisions:
            outcome_str = ""
            if d.outcome_evaluated:
                if d.outcome_score is not None:
                    if d.outcome_score >= 0.7:
                        outcome_str = "→ 成功"
                    elif d.outcome_score >= 0.3:
                        outcome_str = "→ 部分的成功"
                    else:
                        outcome_str = "→ 失敗"

            actions_str = ", ".join(a.get("type", "unknown") for a in d.actions[:3])
            lines.append(
                f"- {d.timestamp.strftime('%m/%d %H:%M')} [{d.decision_type}] "
                f"アクション: {actions_str} {outcome_str}"
            )

        # Add pattern summary
        patterns = self.get_decision_patterns()
        if patterns["recommendations"]:
            lines.append("\n過去の分析から:")
            for rec in patterns["recommendations"][:3]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def get_decision_count_today(self) -> int:
        """Get count of decisions made today."""
        today_start = now_jst().replace(hour=0, minute=0, second=0, microsecond=0)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM decisions
                WHERE timestamp >= ?
                """,
                (today_start.isoformat(),),
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    # ==================== Intervention Analysis ====================

    def record_intervention_analysis(self, analysis: InterventionAnalysis) -> int:
        """
        Record an intervention analysis.

        Args:
            analysis: The intervention analysis to record

        Returns:
            ID of the recorded analysis
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO intervention_analysis
                (timestamp, analysis_type, trade_id, price_at_event,
                 optimal_action, actual_action, potential_impact,
                 hindsight_difficulty, contributing_factors, recommendation,
                 evaluated_by_llm, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.timestamp.isoformat(),
                    analysis.analysis_type,
                    analysis.trade_id,
                    analysis.price_at_event,
                    analysis.optimal_action,
                    analysis.actual_action,
                    analysis.potential_impact,
                    analysis.hindsight_difficulty,
                    json.dumps(analysis.contributing_factors),
                    analysis.recommendation,
                    1 if analysis.evaluated_by_llm else 0,
                    now_jst().isoformat(),
                ),
            )
            return cursor.lastrowid

    def get_intervention_analyses(self, days: int = 7) -> list[InterventionAnalysis]:
        """Get intervention analyses from the past N days."""
        cutoff = now_jst() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM intervention_analysis
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (cutoff.isoformat(),),
            )

            analyses = []
            for row in cursor.fetchall():
                analyses.append(
                    InterventionAnalysis(
                        id=row["id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        analysis_type=row["analysis_type"],
                        trade_id=row["trade_id"],
                        price_at_event=row["price_at_event"],
                        optimal_action=row["optimal_action"],
                        actual_action=row["actual_action"],
                        potential_impact=row["potential_impact"],
                        hindsight_difficulty=row["hindsight_difficulty"],
                        contributing_factors=json.loads(row["contributing_factors"] or "[]"),
                        recommendation=row["recommendation"],
                        evaluated_by_llm=bool(row["evaluated_by_llm"]),
                    )
                )
            return analyses

    def get_intervention_stats(self, days: int = 7) -> dict:
        """Get statistics about intervention analyses."""
        analyses = self.get_intervention_analyses(days=days)

        if not analyses:
            return {
                "total": 0,
                "by_type": {},
                "by_difficulty": {},
                "total_potential_impact": 0.0,
                "obvious_misses": 0,
                "recommendations": [],
            }

        by_type = {}
        by_difficulty = {}
        total_impact = 0.0
        recommendations = []

        for a in analyses:
            by_type[a.analysis_type] = by_type.get(a.analysis_type, 0) + 1
            by_difficulty[a.hindsight_difficulty] = by_difficulty.get(a.hindsight_difficulty, 0) + 1
            total_impact += a.potential_impact

            if a.recommendation and a.hindsight_difficulty in ("obvious", "moderate"):
                recommendations.append(a.recommendation)

        return {
            "total": len(analyses),
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "total_potential_impact": total_impact,
            "obvious_misses": by_difficulty.get("obvious", 0),
            "recommendations": recommendations[:5],  # Top 5 recommendations
        }

    def get_intervention_summary_for_prompt(self, days: int = 7) -> str:
        """Get a summary of intervention analyses for Claude prompt."""
        stats = self.get_intervention_stats(days=days)

        if stats["total"] == 0:
            return "介入分析の履歴はありません。"

        lines = [
            f"過去{days}日間の介入分析: {stats['total']}件",
            "",
            "タイプ別:",
        ]

        type_labels = {
            "stop_loss_timing": "損切りタイミング",
            "missed_opportunity": "機会損失",
            "threshold_too_strict": "閾値設定",
        }

        for t, count in stats["by_type"].items():
            label = type_labels.get(t, t)
            lines.append(f"  - {label}: {count}件")

        lines.append("")
        lines.append("事後判断の難易度:")

        difficulty_labels = {
            "obvious": "明白だった",
            "moderate": "ある程度予測可能だった",
            "difficult": "予測困難だった",
        }

        for d, count in stats["by_difficulty"].items():
            label = difficulty_labels.get(d, d)
            lines.append(f"  - {label}: {count}件")

        if stats["total_potential_impact"] != 0:
            sign = "+" if stats["total_potential_impact"] > 0 else ""
            lines.append(f"\n推定インパクト: {sign}¥{stats['total_potential_impact']:,.0f}")

        return "\n".join(lines)

    # ==================== Database Maintenance ====================

    def cleanup_old_records(self, retention_days: int = 90) -> dict[str, int]:
        """
        Delete records older than retention_days.

        Args:
            retention_days: Number of days to retain data (default 90)

        Returns:
            Dict with count of deleted records per table
        """
        cutoff = (now_jst() - timedelta(days=retention_days)).isoformat()
        deleted = {}

        with self._get_connection() as conn:
            # Clean decisions table
            cursor = conn.execute(
                "DELETE FROM decisions WHERE timestamp < ?",
                (cutoff,)
            )
            deleted["decisions"] = cursor.rowcount

            # Clean signal_outcomes table
            cursor = conn.execute(
                "DELETE FROM signal_outcomes WHERE timestamp < ?",
                (cutoff,)
            )
            deleted["signal_outcomes"] = cursor.rowcount

            # Clean param_history table
            cursor = conn.execute(
                "DELETE FROM param_history WHERE timestamp < ?",
                (cutoff,)
            )
            deleted["param_history"] = cursor.rowcount

            # Clean intervention_analysis table
            cursor = conn.execute(
                "DELETE FROM intervention_analysis WHERE timestamp < ?",
                (cutoff,)
            )
            deleted["intervention_analysis"] = cursor.rowcount

            conn.commit()

        total = sum(deleted.values())
        if total > 0:
            logger.info(f"Database cleanup: {total} records deleted (retention: {retention_days} days)")
            logger.debug(f"Cleanup details: {deleted}")

        return deleted

    def vacuum(self) -> None:
        """
        Reclaim disk space by running VACUUM.
        Should be called after cleanup_old_records.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_database_stats(self) -> dict:
        """Get database size and record counts."""
        stats = {
            "file_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "file_size_mb": 0.0,
            "tables": {},
        }
        stats["file_size_mb"] = stats["file_size_bytes"] / (1024 * 1024)

        with self._get_connection() as conn:
            for table in ["decisions", "signal_outcomes", "param_history", "intervention_analysis", "feature_importance"]:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats["tables"][table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    stats["tables"][table] = 0

        return stats

    def close(self) -> None:
        """
        Close the memory module gracefully.
        Ensures all pending transactions are committed.
        """
        # Run a final commit to ensure no pending changes
        with self._get_connection() as conn:
            conn.commit()
        logger.info("Agent memory closed")
