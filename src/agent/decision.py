"""Decision types and autonomous decision matrix for Meta AI Agent."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AutonomyLevel(str, Enum):
    """Level of autonomy for agent decisions."""

    # Execute automatically, log only
    AUTO_EXECUTE = "auto_execute"

    # Execute automatically + send detailed report
    AUTO_EXECUTE_REPORT = "auto_execute_report"

    # Propose to human, wait for approval
    PROPOSE_TO_HUMAN = "propose"

    # Emergency: execute immediately + urgent notification
    EMERGENCY = "emergency"


class ActionType(str, Enum):
    """Types of actions the agent can take."""

    # Parameter adjustments
    PARAM_ADJUSTMENT = "param_adjustment"
    THRESHOLD_CHANGE = "threshold_change"

    # Feature management
    FEATURE_TOGGLE = "feature_toggle"
    FEATURE_IMPORTANCE_UPDATE = "feature_importance_update"

    # Model operations
    MODEL_RETRAIN_TRIGGER = "model_retrain_trigger"
    MODEL_EVALUATION = "model_evaluation"

    # Alerts and notifications
    ALERT_INFO = "alert_info"
    ALERT_WARNING = "alert_warning"
    ALERT_CRITICAL = "alert_critical"

    # Trading controls
    EMERGENCY_STOP = "emergency_stop"
    DIRECTION_STOP = "direction_stop"
    DIRECTION_RESUME = "direction_resume"

    # Analysis and reporting
    DAILY_REVIEW = "daily_review"
    SIGNAL_VERIFICATION = "signal_verification"
    PERFORMANCE_REPORT = "performance_report"

    # No action needed
    NO_ACTION = "no_action"


@dataclass
class AgentAction:
    """A single action to be taken by the agent."""

    action_type: ActionType
    detail: str
    autonomy_level: AutonomyLevel
    reasoning: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.action_type.value,
            "detail": self.detail,
            "autonomy_level": self.autonomy_level.value,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentAction":
        return cls(
            action_type=ActionType(data["type"]),
            detail=data["detail"],
            autonomy_level=AutonomyLevel(data["autonomy_level"]),
            reasoning=data["reasoning"],
            parameters=data.get("parameters", {}),
        )


@dataclass
class AgentDecision:
    """A decision made by the agent, containing analysis and actions."""

    timestamp: datetime
    analysis: str
    issues: list[str]
    actions: list[AgentAction]
    confidence: float
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "analysis": self.analysis,
            "issues": self.issues,
            "actions": [a.to_dict() for a in self.actions],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentDecision":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            analysis=data["analysis"],
            issues=data["issues"],
            actions=[AgentAction.from_dict(a) for a in data["actions"]],
            confidence=data["confidence"],
        )

    def get_actions_by_autonomy(self, level: AutonomyLevel) -> list[AgentAction]:
        """Get actions filtered by autonomy level."""
        return [a for a in self.actions if a.autonomy_level == level]

    def has_emergency_actions(self) -> bool:
        """Check if any emergency actions are present."""
        return any(a.autonomy_level == AutonomyLevel.EMERGENCY for a in self.actions)

    def has_proposals(self) -> bool:
        """Check if any actions need human approval."""
        return any(a.autonomy_level == AutonomyLevel.PROPOSE_TO_HUMAN for a in self.actions)


class AutonomousDecisionMatrix:
    """
    Matrix defining what level of autonomy is allowed for different situations.
    This ensures the agent doesn't overstep its bounds.
    """

    # Default autonomy levels for different action types
    DEFAULT_AUTONOMY: dict[str, AutonomyLevel] = {
        # Fully autonomous (notification only)
        "feature_importance_update": AutonomyLevel.AUTO_EXECUTE,
        "minor_threshold_adjustment": AutonomyLevel.AUTO_EXECUTE,  # ±5%
        "signal_outcome_recording": AutonomyLevel.AUTO_EXECUTE,
        "performance_snapshot": AutonomyLevel.AUTO_EXECUTE,

        # Autonomous with detailed report
        "feature_enable_disable": AutonomyLevel.AUTO_EXECUTE_REPORT,
        "moderate_param_change": AutonomyLevel.AUTO_EXECUTE_REPORT,  # ±10%
        "daily_review_insights": AutonomyLevel.AUTO_EXECUTE_REPORT,
        "signal_verification_report": AutonomyLevel.AUTO_EXECUTE_REPORT,

        # Requires human approval
        "major_param_change": AutonomyLevel.PROPOSE_TO_HUMAN,  # ±20%+
        "model_retrain_trigger": AutonomyLevel.PROPOSE_TO_HUMAN,
        "new_feature_addition": AutonomyLevel.PROPOSE_TO_HUMAN,
        "strategy_modification": AutonomyLevel.PROPOSE_TO_HUMAN,
        "confidence_threshold_major_change": AutonomyLevel.PROPOSE_TO_HUMAN,

        # Emergency actions (immediate execution)
        "flash_crash_detected": AutonomyLevel.EMERGENCY,
        "critical_loss_threshold": AutonomyLevel.EMERGENCY,
        "system_anomaly": AutonomyLevel.EMERGENCY,
        "consecutive_losses_exceeded": AutonomyLevel.EMERGENCY,
    }

    # Thresholds for categorizing parameter changes
    PARAM_CHANGE_THRESHOLDS = {
        "minor": 0.05,    # ±5% - auto execute
        "moderate": 0.10,  # ±10% - auto execute with report
        "major": 0.20,     # ±20%+ - requires approval
    }

    @classmethod
    def get_autonomy_level(cls, action_key: str) -> AutonomyLevel:
        """Get the autonomy level for a given action."""
        return cls.DEFAULT_AUTONOMY.get(action_key, AutonomyLevel.PROPOSE_TO_HUMAN)

    @classmethod
    def categorize_param_change(cls, old_value: float, new_value: float) -> str:
        """Categorize a parameter change as minor/moderate/major."""
        if old_value == 0:
            return "major" if new_value != 0 else "minor"

        change_pct = abs(new_value - old_value) / abs(old_value)

        if change_pct <= cls.PARAM_CHANGE_THRESHOLDS["minor"]:
            return "minor"
        elif change_pct <= cls.PARAM_CHANGE_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "major"

    @classmethod
    def get_param_change_autonomy(cls, old_value: float, new_value: float) -> AutonomyLevel:
        """Get autonomy level for a parameter change."""
        category = cls.categorize_param_change(old_value, new_value)

        match category:
            case "minor":
                return AutonomyLevel.AUTO_EXECUTE
            case "moderate":
                return AutonomyLevel.AUTO_EXECUTE_REPORT
            case "major":
                return AutonomyLevel.PROPOSE_TO_HUMAN
            case _:
                return AutonomyLevel.PROPOSE_TO_HUMAN
