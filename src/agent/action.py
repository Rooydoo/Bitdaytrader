"""Action module for Meta AI Agent - executes decisions."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
from loguru import logger

from src.agent.decision import AgentAction, ActionType, AutonomyLevel
from src.agent.memory import AgentMemory
from src.utils.timezone import now_jst


@dataclass
class ActionResult:
    """Result of executing an action."""

    action: AgentAction
    success: bool
    message: str
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_jst)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action.action_type.value,
            "success": self.success,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionSummary:
    """Summary of all actions executed."""

    total_actions: int
    successful: int
    failed: int
    results: list[ActionResult]
    overall_success: bool

    def to_dict(self) -> dict:
        return {
            "total_actions": self.total_actions,
            "successful": self.successful,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
            "overall_success": self.overall_success,
        }


class ActionExecutor:
    """
    Executes actions decided by the agent.
    Handles different action types and autonomy levels.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8088",
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
        memory: AgentMemory | None = None,
    ) -> None:
        """
        Initialize action executor.

        Args:
            api_base_url: Base URL for trading bot API
            telegram_token: Telegram bot token for notifications
            telegram_chat_id: Telegram chat ID for notifications
            memory: Agent memory for recording changes
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.memory = memory
        self._http_session: aiohttp.ClientSession | None = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    async def execute_actions(
        self,
        actions: list[AgentAction],
        require_approval_callback: Any = None,
    ) -> ExecutionSummary:
        """
        Execute a list of actions.

        Args:
            actions: List of actions to execute
            require_approval_callback: Optional callback for actions requiring approval

        Returns:
            ExecutionSummary with results
        """
        results = []

        for action in actions:
            try:
                result = await self._execute_single_action(action, require_approval_callback)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute action {action.action_type}: {e}")
                results.append(ActionResult(
                    action=action,
                    success=False,
                    message=f"Execution error: {e}",
                ))

        successful = sum(1 for r in results if r.success)

        return ExecutionSummary(
            total_actions=len(actions),
            successful=successful,
            failed=len(actions) - successful,
            results=results,
            overall_success=successful == len(actions) if actions else True,
        )

    async def _execute_single_action(
        self,
        action: AgentAction,
        require_approval_callback: Any = None,
    ) -> ActionResult:
        """Execute a single action based on its type."""

        # Check autonomy level
        if action.autonomy_level == AutonomyLevel.PROPOSE_TO_HUMAN:
            # Send proposal to human instead of executing
            await self._send_proposal(action)
            return ActionResult(
                action=action,
                success=True,
                message="Proposal sent to human for approval",
            )

        # Execute based on action type
        match action.action_type:
            case ActionType.PARAM_ADJUSTMENT | ActionType.THRESHOLD_CHANGE:
                return await self._execute_param_adjustment(action)

            case ActionType.FEATURE_TOGGLE:
                return await self._execute_feature_toggle(action)

            case ActionType.FEATURE_IMPORTANCE_UPDATE:
                return await self._execute_feature_importance_update(action)

            case ActionType.MODEL_RETRAIN_TRIGGER:
                return await self._execute_model_retrain(action)

            case ActionType.EMERGENCY_STOP:
                return await self._execute_emergency_stop(action)

            case ActionType.DIRECTION_STOP:
                return await self._execute_direction_stop(action)

            case ActionType.DIRECTION_RESUME:
                return await self._execute_direction_resume(action)

            case ActionType.ALERT_INFO | ActionType.ALERT_WARNING | ActionType.ALERT_CRITICAL:
                return await self._execute_alert(action)

            case ActionType.DAILY_REVIEW:
                return await self._execute_daily_review_notification(action)

            case ActionType.SIGNAL_VERIFICATION:
                return await self._execute_signal_verification_notification(action)

            case ActionType.PERFORMANCE_REPORT:
                return await self._execute_performance_report(action)

            case ActionType.NO_ACTION:
                return ActionResult(
                    action=action,
                    success=True,
                    message="No action required",
                )

            case _:
                logger.warning(f"Unknown action type: {action.action_type}")
                return ActionResult(
                    action=action,
                    success=False,
                    message=f"Unknown action type: {action.action_type}",
                )

    # ==================== Action Implementations ====================

    async def _execute_param_adjustment(self, action: AgentAction) -> ActionResult:
        """Execute parameter adjustment."""
        params = action.parameters
        param_name = params.get("param_name")
        new_value = params.get("new_value")
        old_value = params.get("old_value")

        if not param_name or new_value is None:
            return ActionResult(
                action=action,
                success=False,
                message="Missing param_name or new_value",
            )

        try:
            session = await self._get_http_session()

            # Update setting via API
            async with session.post(
                f"{self.api_base_url}/api/settings",
                json={"key": param_name, "value": new_value}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return ActionResult(
                        action=action,
                        success=False,
                        message=f"API error: {error}",
                    )

            # Record in memory
            if self.memory and old_value is not None:
                self.memory.record_param_change(
                    param_name=param_name,
                    old_value=old_value,
                    new_value=new_value,
                    change_reason=action.reasoning,
                    changed_by="agent",
                )

            # Send notification if needed
            if action.autonomy_level == AutonomyLevel.AUTO_EXECUTE_REPORT:
                await self._send_telegram(
                    f"🔧 パラメータ調整\n"
                    f"パラメータ: {param_name}\n"
                    f"変更: {old_value} → {new_value}\n"
                    f"理由: {action.reasoning}"
                )

            logger.info(f"Parameter adjusted: {param_name} = {new_value}")
            return ActionResult(
                action=action,
                success=True,
                message=f"Parameter {param_name} updated to {new_value}",
                details={"param_name": param_name, "old_value": old_value, "new_value": new_value},
            )

        except Exception as e:
            logger.error(f"Failed to adjust parameter: {e}")
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_feature_toggle(self, action: AgentAction) -> ActionResult:
        """Toggle a feature on/off."""
        params = action.parameters
        feature_name = params.get("feature_name")
        enabled = params.get("enabled", True)

        if not feature_name:
            return ActionResult(
                action=action,
                success=False,
                message="Missing feature_name",
            )

        try:
            # Update in memory/feature registry
            if self.memory:
                self.memory.toggle_feature(feature_name, enabled)

            # Send notification
            status = "有効化" if enabled else "無効化"
            await self._send_telegram(
                f"🔀 特徴量{status}\n"
                f"特徴量: {feature_name}\n"
                f"理由: {action.reasoning}"
            )

            logger.info(f"Feature toggled: {feature_name} = {enabled}")
            return ActionResult(
                action=action,
                success=True,
                message=f"Feature {feature_name} {'enabled' if enabled else 'disabled'}",
                details={"feature_name": feature_name, "enabled": enabled},
            )

        except Exception as e:
            logger.error(f"Failed to toggle feature: {e}")
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_feature_importance_update(self, action: AgentAction) -> ActionResult:
        """Update feature importance scores."""
        params = action.parameters
        importance_data = params.get("importance", {})

        if not importance_data:
            return ActionResult(
                action=action,
                success=True,
                message="No importance data to update",
            )

        try:
            if self.memory:
                for feature_name, score in importance_data.items():
                    self.memory.update_feature_importance(feature_name, score)

            logger.info(f"Feature importance updated: {len(importance_data)} features")
            return ActionResult(
                action=action,
                success=True,
                message=f"Updated importance for {len(importance_data)} features",
                details={"features_updated": list(importance_data.keys())},
            )

        except Exception as e:
            logger.error(f"Failed to update feature importance: {e}")
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_model_retrain(self, action: AgentAction) -> ActionResult:
        """Trigger model retraining (proposal only, actual retrain is manual)."""
        # Model retraining should always be a proposal, not auto-execute
        await self._send_telegram(
            f"📊 モデル再学習提案\n"
            f"理由: {action.reasoning}\n"
            f"詳細: {action.detail}\n\n"
            f"再学習を実行する場合は手動で実行してください。"
        )

        return ActionResult(
            action=action,
            success=True,
            message="Model retrain proposal sent",
        )

    async def _execute_emergency_stop(self, action: AgentAction) -> ActionResult:
        """Execute emergency stop."""
        params = action.parameters
        mode = params.get("mode", "no_new_positions")
        message = params.get("message", action.reasoning)

        try:
            session = await self._get_http_session()

            async with session.post(
                f"{self.api_base_url}/api/emergency/stop",
                json={
                    "mode": mode,
                    "reason": "agent_decision",
                    "message": message,
                }
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return ActionResult(
                        action=action,
                        success=False,
                        message=f"API error: {error}",
                    )

            # Emergency notification is sent by API

            logger.warning(f"Emergency stop executed: {mode}")
            return ActionResult(
                action=action,
                success=True,
                message=f"Emergency stop activated: {mode}",
                details={"mode": mode},
            )

        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            # Try to send alert even if API failed
            await self._send_telegram(
                f"🚨 緊急停止失敗！\n"
                f"理由: {message}\n"
                f"エラー: {e}\n\n"
                f"手動で確認してください！"
            )
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_direction_stop(self, action: AgentAction) -> ActionResult:
        """Stop trading for a specific direction."""
        params = action.parameters
        direction = params.get("direction", "").upper()
        reason = params.get("reason", action.reasoning)

        if direction not in ["LONG", "SHORT"]:
            return ActionResult(
                action=action,
                success=False,
                message="Invalid direction",
            )

        try:
            session = await self._get_http_session()

            async with session.post(
                f"{self.api_base_url}/api/emergency/stop/{direction.lower()}",
                params={"reason": reason}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return ActionResult(
                        action=action,
                        success=False,
                        message=f"API error: {error}",
                    )

            logger.warning(f"Direction stopped: {direction}")
            return ActionResult(
                action=action,
                success=True,
                message=f"{direction} trading stopped",
                details={"direction": direction},
            )

        except Exception as e:
            logger.error(f"Failed to stop direction: {e}")
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_direction_resume(self, action: AgentAction) -> ActionResult:
        """Resume trading for a specific direction."""
        params = action.parameters
        direction = params.get("direction", "").upper()

        if direction not in ["LONG", "SHORT"]:
            return ActionResult(
                action=action,
                success=False,
                message="Invalid direction",
            )

        try:
            session = await self._get_http_session()

            async with session.post(
                f"{self.api_base_url}/api/emergency/resume/{direction.lower()}"
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return ActionResult(
                        action=action,
                        success=False,
                        message=f"API error: {error}",
                    )

            logger.info(f"Direction resumed: {direction}")
            return ActionResult(
                action=action,
                success=True,
                message=f"{direction} trading resumed",
                details={"direction": direction},
            )

        except Exception as e:
            logger.error(f"Failed to resume direction: {e}")
            return ActionResult(
                action=action,
                success=False,
                message=f"Failed: {e}",
            )

    async def _execute_alert(self, action: AgentAction) -> ActionResult:
        """Send alert notification."""
        severity = {
            ActionType.ALERT_INFO: "ℹ️",
            ActionType.ALERT_WARNING: "⚠️",
            ActionType.ALERT_CRITICAL: "🚨",
        }.get(action.action_type, "ℹ️")

        message = f"{severity} {action.detail}\n\n理由: {action.reasoning}"

        success = await self._send_telegram(message)

        return ActionResult(
            action=action,
            success=success,
            message="Alert sent" if success else "Failed to send alert",
        )

    async def _execute_daily_review_notification(self, action: AgentAction) -> ActionResult:
        """Send daily review report."""
        report = action.parameters.get("report", action.detail)

        success = await self._send_telegram(
            f"📋 日次レビュー\n\n{report}"
        )

        return ActionResult(
            action=action,
            success=success,
            message="Daily review sent" if success else "Failed to send review",
        )

    async def _execute_signal_verification_notification(self, action: AgentAction) -> ActionResult:
        """Send signal verification results."""
        params = action.parameters
        stats = params.get("stats", {})

        message = (
            f"📊 シグナル検証結果\n\n"
            f"検証数: {stats.get('total', 0)}\n"
            f"正解率: {stats.get('accuracy', 0):.1%}\n"
            f"LONG正解率: {stats.get('long_accuracy', 0):.1%}\n"
            f"SHORT正解率: {stats.get('short_accuracy', 0):.1%}\n\n"
            f"{action.detail}"
        )

        success = await self._send_telegram(message)

        return ActionResult(
            action=action,
            success=success,
            message="Verification report sent" if success else "Failed to send report",
        )

    async def _execute_performance_report(self, action: AgentAction) -> ActionResult:
        """Send performance report."""
        report = action.parameters.get("report", action.detail)

        success = await self._send_telegram(
            f"📈 パフォーマンスレポート\n\n{report}"
        )

        return ActionResult(
            action=action,
            success=success,
            message="Performance report sent" if success else "Failed to send report",
        )

    # ==================== Notification Helpers ====================

    async def _send_proposal(self, action: AgentAction) -> bool:
        """Send a proposal to human for approval."""
        message = (
            f"📝 承認リクエスト\n\n"
            f"アクション: {action.action_type.value}\n"
            f"詳細: {action.detail}\n"
            f"理由: {action.reasoning}\n\n"
            f"このアクションを実行する場合は手動で承認してください。"
        )

        if action.parameters:
            message += f"\n\nパラメータ:\n{json.dumps(action.parameters, ensure_ascii=False, indent=2)}"

        return await self._send_telegram(message)

    async def _send_telegram(self, message: str) -> bool:
        """Send message via Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram not configured, skipping notification")
            return False

        try:
            session = await self._get_http_session()
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

            async with session.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Telegram API error: {error}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
