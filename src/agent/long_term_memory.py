"""Long-term memory module for Meta AI Agent.

Manages persistent knowledge in Markdown files:
- insights.md: Learned insights and lessons
- rules.md: Self-generated rules
- events.md: Important event history
- self_reflection.md: Periodic self-evaluation

Includes validation mechanism to prevent overfitting to outdated patterns.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.timezone import now_jst


class MemoryStatus(str, Enum):
    """Status of a memory item."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    DEPRECATED = "deprecated"


class ConfidenceLevel(str, Enum):
    """Confidence level of a memory item."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """A learned insight or lesson."""
    id: str
    category: str
    title: str
    content: str
    evidence: list[str]
    conditions: list[str]
    created_at: datetime
    last_verified: datetime
    confidence: ConfidenceLevel
    verification_count: int = 0
    verification_success: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE

    @property
    def success_rate(self) -> float:
        if self.verification_count == 0:
            return 0.0
        return self.verification_success / self.verification_count

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        return f"""### [{self.category}] {self.title}
- ID: {self.id}
- 記録日: {self.created_at.strftime('%Y-%m-%d')}
- 最終検証: {self.last_verified.strftime('%Y-%m-%d')}
- 信頼度: {self.confidence.value}
- 検証回数: {self.verification_count}回（成功{self.verification_success}回）
- 状態: {self.status.value}

{self.content}

**根拠:**
{chr(10).join(f'- {e}' for e in self.evidence)}

**適用条件:**
{chr(10).join(f'- {c}' for c in self.conditions)}
"""


@dataclass
class Rule:
    """A self-generated rule."""
    id: str
    name: str
    rule_type: str  # "do", "dont", "conditional"
    content: str
    origin: str  # How this rule was generated
    created_at: datetime
    last_verified: datetime
    confidence: ConfidenceLevel
    application_count: int = 0
    success_count: int = 0
    verification_history: list[dict] = field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE

    @property
    def success_rate(self) -> float:
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        history_str = ""
        for h in self.verification_history[-5:]:  # Last 5 verifications
            history_str += f"- {h.get('date', 'N/A')}: {h.get('summary', 'N/A')}\n"

        return f"""### [RULE-{self.id}] {self.name}
- 生成日: {self.created_at.strftime('%Y-%m-%d')}
- 最終検証: {self.last_verified.strftime('%Y-%m-%d')}
- 信頼度: {self.confidence.value}
- 適用回数: {self.application_count}回
- 成功率: {self.success_rate:.0%}
- 状態: {self.status.value}

**ルール内容:**
{self.content}

**生成の経緯:**
{self.origin}

**検証履歴:**
{history_str if history_str else '（まだ検証履歴がありません）'}
"""


@dataclass
class Event:
    """An important event record."""
    timestamp: datetime
    name: str
    category: str
    severity: str  # "critical", "high", "medium"
    impact: str
    situation: str
    response: str
    result: str
    lessons: list[str]

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        return f"""### [{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.name}
- カテゴリ: {self.category}
- 重要度: {self.severity}
- 影響: {self.impact}

**状況:**
{self.situation}

**エージェントの対応:**
{self.response}

**結果:**
{self.result}

**教訓:**
{chr(10).join(f'- {l}' for l in self.lessons)}
"""


class LongTermMemory:
    """
    Long-term memory manager for the Meta AI Agent.

    Stores and retrieves persistent knowledge from Markdown files.
    Includes validation mechanism to prevent overfitting.
    """

    # Validation thresholds
    MIN_VERIFICATION_FOR_HIGH_CONFIDENCE = 5
    MIN_SUCCESS_RATE_FOR_ACTIVE = 0.4
    DAYS_BEFORE_REVIEW = 30  # Days without verification before review
    DAYS_BEFORE_DEPRECATION = 90  # Days in review before deprecation

    def __init__(self, memory_dir: str = "data/memory") -> None:
        """
        Initialize long-term memory.

        Args:
            memory_dir: Directory containing memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.insights_file = self.memory_dir / "insights.md"
        self.rules_file = self.memory_dir / "rules.md"
        self.events_file = self.memory_dir / "events.md"
        self.reflection_file = self.memory_dir / "self_reflection.md"

        # In-memory cache
        self._insights: dict[str, Insight] = {}
        self._rules: dict[str, Rule] = {}
        self._events: list[Event] = []

        # Load existing data
        self._load_all()

        logger.info(f"Long-term memory initialized: {self.memory_dir}")

    def _load_all(self) -> None:
        """Load all memory from files."""
        # For now, we start fresh and build up
        # In production, we would parse existing markdown files
        pass

    # ==================== Insights ====================

    def add_insight(
        self,
        category: str,
        title: str,
        content: str,
        evidence: list[str],
        conditions: list[str] | None = None,
        confidence: ConfidenceLevel = ConfidenceLevel.LOW,
    ) -> Insight:
        """
        Add a new insight.

        Args:
            category: Category (e.g., "市場パターン", "シグナル精度")
            title: Brief title
            content: Detailed content
            evidence: List of evidence supporting this insight
            conditions: Conditions when this insight applies
            confidence: Initial confidence level

        Returns:
            The created Insight
        """
        now = now_jst()
        insight_id = f"INS-{now.strftime('%Y%m%d%H%M%S')}"

        insight = Insight(
            id=insight_id,
            category=category,
            title=title,
            content=content,
            evidence=evidence,
            conditions=conditions or [],
            created_at=now,
            last_verified=now,
            confidence=confidence,
        )

        self._insights[insight_id] = insight
        self._save_insights()

        logger.info(f"Added insight: {insight_id} - {title}")
        return insight

    def verify_insight(
        self,
        insight_id: str,
        success: bool,
        notes: str = "",
    ) -> bool:
        """
        Verify an insight against recent data.

        Args:
            insight_id: ID of the insight
            success: Whether the insight held true
            notes: Optional notes about the verification

        Returns:
            True if insight status changed
        """
        if insight_id not in self._insights:
            logger.warning(f"Insight not found: {insight_id}")
            return False

        insight = self._insights[insight_id]
        insight.verification_count += 1
        if success:
            insight.verification_success += 1
        insight.last_verified = now_jst()

        # Update confidence based on verification history
        status_changed = self._update_insight_status(insight)
        self._save_insights()

        logger.info(
            f"Verified insight {insight_id}: success={success}, "
            f"rate={insight.success_rate:.0%}, status={insight.status.value}"
        )
        return status_changed

    def _update_insight_status(self, insight: Insight) -> bool:
        """Update insight status based on verification results."""
        old_status = insight.status

        # Check if enough verifications for high confidence
        if insight.verification_count >= self.MIN_VERIFICATION_FOR_HIGH_CONFIDENCE:
            if insight.success_rate >= 0.8:
                insight.confidence = ConfidenceLevel.HIGH
            elif insight.success_rate >= 0.5:
                insight.confidence = ConfidenceLevel.MEDIUM
            else:
                insight.confidence = ConfidenceLevel.LOW

        # Check if success rate is too low
        if (insight.verification_count >= 3 and
                insight.success_rate < self.MIN_SUCCESS_RATE_FOR_ACTIVE):
            if insight.status == MemoryStatus.ACTIVE:
                insight.status = MemoryStatus.UNDER_REVIEW
                logger.warning(
                    f"Insight {insight.id} moved to review: "
                    f"success_rate={insight.success_rate:.0%}"
                )
            elif insight.status == MemoryStatus.UNDER_REVIEW:
                # Check if it's been under review for too long
                days_in_review = (now_jst() - insight.last_verified).days
                if days_in_review > self.DAYS_BEFORE_DEPRECATION:
                    insight.status = MemoryStatus.DEPRECATED
                    logger.warning(f"Insight {insight.id} deprecated")

        return old_status != insight.status

    def get_active_insights(self, category: str | None = None) -> list[Insight]:
        """Get active insights, optionally filtered by category."""
        insights = [
            i for i in self._insights.values()
            if i.status == MemoryStatus.ACTIVE
        ]
        if category:
            insights = [i for i in insights if i.category == category]
        return sorted(insights, key=lambda x: x.confidence.value)

    def get_insights_for_prompt(self, max_items: int = 5) -> str:
        """Get insights formatted for Claude prompt."""
        active = self.get_active_insights()

        if not active:
            return "（学習済みの洞察はありません）"

        # Prioritize high confidence insights
        high_conf = [i for i in active if i.confidence == ConfidenceLevel.HIGH]
        medium_conf = [i for i in active if i.confidence == ConfidenceLevel.MEDIUM]

        selected = high_conf[:max_items]
        if len(selected) < max_items:
            selected.extend(medium_conf[:max_items - len(selected)])

        lines = []
        for insight in selected:
            conf_emoji = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(
                insight.confidence.value, "⚪"
            )
            lines.append(
                f"{conf_emoji} [{insight.category}] {insight.title}\n"
                f"   {insight.content}\n"
                f"   (検証{insight.verification_count}回, 成功率{insight.success_rate:.0%})"
            )

        return "\n\n".join(lines)

    def _save_insights(self) -> None:
        """Save insights to markdown file."""
        # Group by category
        by_category: dict[str, list[Insight]] = {}
        for insight in self._insights.values():
            cat = insight.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(insight)

        content = """# 学んだ洞察・教訓

このファイルはMetaAgentが学んだ洞察を記録します。
各洞察には検証状態と信頼度が付与されます。

---

"""
        for category, insights in sorted(by_category.items()):
            content += f"## {category}\n\n"
            for insight in sorted(insights, key=lambda x: x.created_at, reverse=True):
                content += insight.to_markdown() + "\n---\n\n"

        try:
            self.insights_file.write_text(content, encoding="utf-8")
        except IOError as e:
            logger.error(f"Failed to save insights to disk: {e}")

    # ==================== Rules ====================

    def add_rule(
        self,
        name: str,
        rule_type: str,
        content: str,
        origin: str,
        confidence: ConfidenceLevel = ConfidenceLevel.LOW,
    ) -> Rule:
        """
        Add a new rule.

        Args:
            name: Rule name
            rule_type: "do", "dont", or "conditional"
            content: Rule content
            origin: How this rule was generated
            confidence: Initial confidence level

        Returns:
            The created Rule
        """
        now = now_jst()
        rule_id = now.strftime('%Y%m%d%H%M%S')

        rule = Rule(
            id=rule_id,
            name=name,
            rule_type=rule_type,
            content=content,
            origin=origin,
            created_at=now,
            last_verified=now,
            confidence=confidence,
        )

        self._rules[rule_id] = rule
        self._save_rules()

        logger.info(f"Added rule: RULE-{rule_id} - {name}")
        return rule

    def apply_rule(
        self,
        rule_id: str,
        success: bool,
        context: str = "",
    ) -> None:
        """
        Record a rule application.

        Args:
            rule_id: ID of the rule
            success: Whether the rule application was successful
            context: Context of the application
        """
        if rule_id not in self._rules:
            logger.warning(f"Rule not found: {rule_id}")
            return

        rule = self._rules[rule_id]
        rule.application_count += 1
        if success:
            rule.success_count += 1
        rule.last_verified = now_jst()

        # Add to verification history
        rule.verification_history.append({
            "date": now_jst().strftime('%Y-%m-%d'),
            "success": success,
            "summary": context[:100] if context else "N/A",
        })

        # Update status
        self._update_rule_status(rule)
        self._save_rules()

    def _update_rule_status(self, rule: Rule) -> bool:
        """Update rule status based on application results."""
        old_status = rule.status

        # Update confidence based on application history
        if rule.application_count >= self.MIN_VERIFICATION_FOR_HIGH_CONFIDENCE:
            if rule.success_rate >= 0.8:
                rule.confidence = ConfidenceLevel.HIGH
            elif rule.success_rate >= 0.5:
                rule.confidence = ConfidenceLevel.MEDIUM
            else:
                rule.confidence = ConfidenceLevel.LOW

        # Check if success rate is too low
        if (rule.application_count >= 5 and
                rule.success_rate < self.MIN_SUCCESS_RATE_FOR_ACTIVE):
            if rule.status == MemoryStatus.ACTIVE:
                rule.status = MemoryStatus.UNDER_REVIEW
                logger.warning(
                    f"Rule RULE-{rule.id} moved to review: "
                    f"success_rate={rule.success_rate:.0%}"
                )

        return old_status != rule.status

    def get_active_rules(self, rule_type: str | None = None) -> list[Rule]:
        """Get active rules, optionally filtered by type."""
        rules = [
            r for r in self._rules.values()
            if r.status == MemoryStatus.ACTIVE
        ]
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        return sorted(rules, key=lambda x: x.confidence.value)

    def get_rules_for_prompt(self, max_items: int = 10) -> str:
        """Get rules formatted for Claude prompt."""
        active = self.get_active_rules()

        if not active:
            return "（学習済みのルールはありません）"

        # Group by type
        do_rules = [r for r in active if r.rule_type == "do" and r.confidence != ConfidenceLevel.LOW]
        dont_rules = [r for r in active if r.rule_type == "dont" and r.confidence != ConfidenceLevel.LOW]
        cond_rules = [r for r in active if r.rule_type == "conditional" and r.confidence != ConfidenceLevel.LOW]

        lines = []

        if do_rules:
            lines.append("**やるべきこと:**")
            for r in do_rules[:max_items // 3]:
                lines.append(f"- {r.content} (成功率{r.success_rate:.0%})")

        if dont_rules:
            lines.append("\n**やってはいけないこと:**")
            for r in dont_rules[:max_items // 3]:
                lines.append(f"- {r.content} (成功率{r.success_rate:.0%})")

        if cond_rules:
            lines.append("\n**条件付きルール:**")
            for r in cond_rules[:max_items // 3]:
                lines.append(f"- {r.content} (成功率{r.success_rate:.0%})")

        return "\n".join(lines) if lines else "（学習済みのルールはありません）"

    def _save_rules(self) -> None:
        """Save rules to markdown file."""
        do_rules = [r for r in self._rules.values() if r.rule_type == "do"]
        dont_rules = [r for r in self._rules.values() if r.rule_type == "dont"]
        cond_rules = [r for r in self._rules.values() if r.rule_type == "conditional"]

        content = """# 自己生成ルール

このファイルはMetaAgentが経験から学んだルールを記録します。
ルールは定期的に検証され、有効性が確認されないものは淘汰されます。

---

## やるべきこと（DO）

"""
        for rule in sorted(do_rules, key=lambda x: x.created_at, reverse=True):
            content += rule.to_markdown() + "\n---\n\n"

        content += "## やってはいけないこと（DON'T）\n\n"
        for rule in sorted(dont_rules, key=lambda x: x.created_at, reverse=True):
            content += rule.to_markdown() + "\n---\n\n"

        content += "## 条件付きルール\n\n"
        for rule in sorted(cond_rules, key=lambda x: x.created_at, reverse=True):
            content += rule.to_markdown() + "\n---\n\n"

        try:
            self.rules_file.write_text(content, encoding="utf-8")
        except IOError as e:
            logger.error(f"Failed to save rules to disk: {e}")

    # ==================== Events ====================

    def add_event(
        self,
        name: str,
        category: str,
        severity: str,
        impact: str,
        situation: str,
        response: str,
        result: str,
        lessons: list[str],
    ) -> Event:
        """
        Add an important event.

        Args:
            name: Event name
            category: Category (e.g., "market_crash", "system_error")
            severity: "critical", "high", or "medium"
            impact: Impact description
            situation: What happened
            response: How the agent responded
            result: Result of the response
            lessons: Lessons learned

        Returns:
            The created Event
        """
        event = Event(
            timestamp=now_jst(),
            name=name,
            category=category,
            severity=severity,
            impact=impact,
            situation=situation,
            response=response,
            result=result,
            lessons=lessons,
        )

        self._events.append(event)
        self._save_events()

        logger.info(f"Added event: {name} ({category}, {severity})")
        return event

    def get_similar_events(
        self,
        category: str,
        limit: int = 5,
    ) -> list[Event]:
        """Get past events of similar category."""
        events = [e for e in self._events if e.category == category]
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_events_for_prompt(self, max_items: int = 3) -> str:
        """Get recent important events for Claude prompt."""
        # Get critical and high severity events from last 30 days
        cutoff = now_jst() - timedelta(days=30)
        recent = [
            e for e in self._events
            if e.timestamp >= cutoff and e.severity in ("critical", "high")
        ]

        if not recent:
            return "（最近の重要イベントはありません）"

        recent = sorted(recent, key=lambda x: x.timestamp, reverse=True)[:max_items]

        lines = []
        for event in recent:
            severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️"}.get(
                event.severity, "ℹ️"
            )
            lessons_str = "; ".join(event.lessons[:2])
            lines.append(
                f"{severity_emoji} [{event.timestamp.strftime('%m/%d')}] {event.name}\n"
                f"   教訓: {lessons_str}"
            )

        return "\n\n".join(lines)

    def _save_events(self) -> None:
        """Save events to markdown file."""
        # Group by year
        by_year: dict[int, list[Event]] = {}
        for event in self._events:
            year = event.timestamp.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(event)

        content = """# 重要イベント履歴

このファイルは市場やシステムの重要なイベントを記録します。
将来の類似状況での判断に活用されます。

---

"""
        for year in sorted(by_year.keys(), reverse=True):
            content += f"## {year}年\n\n"
            events = sorted(by_year[year], key=lambda x: x.timestamp, reverse=True)
            for event in events:
                content += event.to_markdown() + "\n---\n\n"

        try:
            self.events_file.write_text(content, encoding="utf-8")
        except IOError as e:
            logger.error(f"Failed to save events to disk: {e}")

    # ==================== Self Reflection ====================

    def add_weekly_reflection(
        self,
        start_date: datetime,
        end_date: datetime,
        performance_summary: dict,
        good_things: list[str],
        improvements_needed: list[str],
        focus_points: list[str],
        memory_updates: list[str],
    ) -> None:
        """
        Add a weekly self-reflection.

        Args:
            start_date: Start of the week
            end_date: End of the week
            performance_summary: Performance metrics
            good_things: Things that went well
            improvements_needed: Things to improve
            focus_points: Focus points for next week
            memory_updates: Rules/insights added/updated/deprecated
        """
        reflection = f"""### 週次振り返り: {start_date.strftime('%Y-%m-%d')} 〜 {end_date.strftime('%Y-%m-%d')}

**パフォーマンスサマリー:**
- シグナル精度: {performance_summary.get('signal_accuracy', 0):.0%}
- 介入成功率: {performance_summary.get('intervention_success', 0):.0%}
- 重大な判断ミス: {performance_summary.get('major_mistakes', 0)}件

**うまくいったこと:**
{chr(10).join(f'- {g}' for g in good_things)}

**改善が必要なこと:**
{chr(10).join(f'- {i}' for i in improvements_needed)}

**来週の注力ポイント:**
{chr(10).join(f'- {f}' for f in focus_points)}

**ルール・洞察の更新:**
{chr(10).join(f'- {m}' for m in memory_updates) if memory_updates else '（なし）'}

---

"""
        # Append to reflection file
        current_content = ""
        if self.reflection_file.exists():
            current_content = self.reflection_file.read_text(encoding="utf-8")

        # Find the position to insert (after "## 週次振り返り履歴")
        marker = "## 週次振り返り履歴"
        if marker in current_content:
            parts = current_content.split(marker)
            new_content = parts[0] + marker + "\n\n" + reflection + parts[1].lstrip("\n")
        else:
            new_content = current_content + "\n" + reflection

        try:
            self.reflection_file.write_text(new_content, encoding="utf-8")
            logger.info(f"Added weekly reflection: {start_date.strftime('%Y-%m-%d')}")
        except IOError as e:
            logger.error(f"Failed to save weekly reflection to disk: {e}")

    # ==================== Validation ====================

    def run_validation(self) -> dict:
        """
        Run validation on all memory items.
        Identifies items that need review or deprecation.

        Returns:
            Dict with validation results
        """
        now = now_jst()
        results = {
            "insights_reviewed": 0,
            "insights_deprecated": 0,
            "rules_reviewed": 0,
            "rules_deprecated": 0,
            "items_needing_attention": [],
        }

        # Check insights
        for insight in self._insights.values():
            if insight.status == MemoryStatus.DEPRECATED:
                continue

            days_since_verification = (now - insight.last_verified).days

            # Mark for review if not verified recently
            if days_since_verification > self.DAYS_BEFORE_REVIEW:
                if insight.status == MemoryStatus.ACTIVE:
                    insight.status = MemoryStatus.UNDER_REVIEW
                    results["insights_reviewed"] += 1
                    results["items_needing_attention"].append({
                        "type": "insight",
                        "id": insight.id,
                        "title": insight.title,
                        "reason": f"未検証期間: {days_since_verification}日",
                    })
                elif (insight.status == MemoryStatus.UNDER_REVIEW and
                      days_since_verification > self.DAYS_BEFORE_DEPRECATION):
                    insight.status = MemoryStatus.DEPRECATED
                    results["insights_deprecated"] += 1

        # Check rules
        for rule in self._rules.values():
            if rule.status == MemoryStatus.DEPRECATED:
                continue

            days_since_verification = (now - rule.last_verified).days

            if days_since_verification > self.DAYS_BEFORE_REVIEW:
                if rule.status == MemoryStatus.ACTIVE:
                    rule.status = MemoryStatus.UNDER_REVIEW
                    results["rules_reviewed"] += 1
                    results["items_needing_attention"].append({
                        "type": "rule",
                        "id": rule.id,
                        "name": rule.name,
                        "reason": f"未検証期間: {days_since_verification}日",
                    })
                elif (rule.status == MemoryStatus.UNDER_REVIEW and
                      days_since_verification > self.DAYS_BEFORE_DEPRECATION):
                    rule.status = MemoryStatus.DEPRECATED
                    results["rules_deprecated"] += 1

        # Save changes
        self._save_insights()
        self._save_rules()

        if results["items_needing_attention"]:
            logger.warning(
                f"Memory validation: {len(results['items_needing_attention'])} items need attention"
            )

        return results

    # ==================== Context Building ====================

    def get_context_for_prompt(self) -> str:
        """
        Get all relevant long-term memory for Claude prompt.

        Returns:
            Formatted string with insights, rules, and recent events
        """
        sections = []

        # Insights
        insights = self.get_insights_for_prompt()
        if insights and "学習済みの洞察はありません" not in insights:
            sections.append(f"## 学んだ洞察\n{insights}")

        # Rules
        rules = self.get_rules_for_prompt()
        if rules and "学習済みのルールはありません" not in rules:
            sections.append(f"## 遵守すべきルール\n{rules}")

        # Recent events
        events = self.get_events_for_prompt()
        if events and "最近の重要イベントはありません" not in events:
            sections.append(f"## 最近の重要イベント\n{events}")

        if not sections:
            return ""

        return "\n\n".join(sections)

    def get_stats(self) -> dict:
        """Get statistics about long-term memory."""
        return {
            "insights": {
                "total": len(self._insights),
                "active": len([i for i in self._insights.values() if i.status == MemoryStatus.ACTIVE]),
                "under_review": len([i for i in self._insights.values() if i.status == MemoryStatus.UNDER_REVIEW]),
                "deprecated": len([i for i in self._insights.values() if i.status == MemoryStatus.DEPRECATED]),
            },
            "rules": {
                "total": len(self._rules),
                "active": len([r for r in self._rules.values() if r.status == MemoryStatus.ACTIVE]),
                "under_review": len([r for r in self._rules.values() if r.status == MemoryStatus.UNDER_REVIEW]),
                "deprecated": len([r for r in self._rules.values() if r.status == MemoryStatus.DEPRECATED]),
            },
            "events": {
                "total": len(self._events),
                "critical": len([e for e in self._events if e.severity == "critical"]),
                "high": len([e for e in self._events if e.severity == "high"]),
            },
        }
