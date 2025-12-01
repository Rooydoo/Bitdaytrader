"""Claude API client for Meta AI Agent."""

import json
import re
from datetime import datetime
from typing import Any

import anthropic
from loguru import logger

from src.agent.decision import (
    AgentAction,
    AgentDecision,
    ActionType,
    AutonomyLevel,
)


class ClaudeClient:
    """Client for interacting with Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use. Defaults to Claude Sonnet 4.5.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        logger.info(f"Claude client initialized with model: {self.model}")

    async def analyze_and_decide(
        self,
        context_prompt: str,
        memory_summary: str = "",
        system_prompt: str | None = None,
    ) -> AgentDecision:
        """
        Send context to Claude and get a decision.

        Args:
            context_prompt: The current situation context
            memory_summary: Summary of past decisions and outcomes
            system_prompt: Optional custom system prompt

        Returns:
            AgentDecision with analysis and actions
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_user_prompt(context_prompt, memory_summary)

        try:
            # Use synchronous API (will be called in thread pool from async context)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            raw_response = response.content[0].text
            decision = self._parse_decision_response(raw_response)

            logger.info(
                f"Claude decision: {len(decision.actions)} actions, "
                f"confidence={decision.confidence:.2f}"
            )

            return decision

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return self._create_error_decision(str(e))

        except Exception as e:
            logger.error(f"Unexpected error in Claude client: {e}")
            return self._create_error_decision(str(e))

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return """あなたはビットコイン自動売買システム「Bitdaytrader」の上位監視AIエージェントです。

## あなたの役割
1. トレーディングシステムの状態を監視し、問題を検知する
2. シグナル予測の精度を検証し、改善点を見つける
3. パラメータや特徴量の調整を提案・実行する
4. 異常事態には緊急対応する

## 判断基準
あなたには以下の自律性レベルが与えられています：

### auto_execute（自律実行、ログのみ）
- 特徴量重要度の更新
- 軽微なパラメータ調整（±5%以内）
- シグナル結果の記録

### auto_execute_report（自律実行 + 詳細報告）
- 特徴量のオン/オフ切り替え
- 中程度のパラメータ変更（±10%以内）
- 日次レビューの分析結果

### propose（人間に提案、承認待ち）
- 大幅なパラメータ変更（±20%以上）
- モデル再学習のトリガー
- 新しい特徴量の追加
- 戦略の変更

### emergency（緊急対応、即時実行 + 緊急通知）
- 急落検知（15分で3%以上の下落）
- 重大な損失パターン検知
- システム異常

## 出力形式
必ず以下のJSON形式で回答してください：

```json
{
    "analysis": "状況の分析結果（日本語で簡潔に）",
    "issues": ["検出された問題1", "検出された問題2"],
    "actions": [
        {
            "type": "param_adjustment|feature_toggle|alert_warning|model_retrain_trigger|emergency_stop|daily_review|signal_verification|no_action|...",
            "detail": "具体的な内容（日本語）",
            "autonomy_level": "auto_execute|auto_execute_report|propose|emergency",
            "reasoning": "この判断の理由（日本語）",
            "parameters": {"key": "value"}
        }
    ],
    "confidence": 0.85
}
```

## 重要な注意事項
- 過去の判断と結果から学習し、同じ失敗を繰り返さない
- 不確実な場合は「propose」を選択して人間に判断を委ねる
- 複数の問題がある場合は、優先度の高いものから対処する
- 緊急事態では迷わず「emergency」を選択する"""

    def _build_user_prompt(self, context_prompt: str, memory_summary: str) -> str:
        """Build the user prompt with context and memory."""
        prompt = f"""## 現在の状況
{context_prompt}
"""

        if memory_summary:
            prompt += f"""
## 過去の判断と結果
{memory_summary}
"""

        prompt += """
上記の状況を分析し、必要なアクションを判断してください。
JSON形式で回答してください。"""

        return prompt

    def _parse_decision_response(self, raw_response: str) -> AgentDecision:
        """Parse Claude's response into an AgentDecision."""
        try:
            # Extract JSON from response (may be wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = raw_response.strip()
                # Find the JSON object
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            data = json.loads(json_str)

            # Parse actions
            actions = []
            for action_data in data.get("actions", []):
                try:
                    action = AgentAction(
                        action_type=self._parse_action_type(action_data.get("type", "no_action")),
                        detail=action_data.get("detail", ""),
                        autonomy_level=AutonomyLevel(action_data.get("autonomy_level", "propose")),
                        reasoning=action_data.get("reasoning", ""),
                        parameters=action_data.get("parameters", {}),
                    )
                    actions.append(action)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse action: {e}")
                    continue

            return AgentDecision(
                timestamp=datetime.now(),
                analysis=data.get("analysis", ""),
                issues=data.get("issues", []),
                actions=actions,
                confidence=float(data.get("confidence", 0.5)),
                raw_response=raw_response,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {raw_response}")
            return self._create_error_decision(f"JSON parse error: {e}")

    def _parse_action_type(self, type_str: str) -> ActionType:
        """Parse action type string to enum."""
        type_mapping = {
            "param_adjustment": ActionType.PARAM_ADJUSTMENT,
            "threshold_change": ActionType.THRESHOLD_CHANGE,
            "feature_toggle": ActionType.FEATURE_TOGGLE,
            "feature_importance_update": ActionType.FEATURE_IMPORTANCE_UPDATE,
            "model_retrain_trigger": ActionType.MODEL_RETRAIN_TRIGGER,
            "model_evaluation": ActionType.MODEL_EVALUATION,
            "alert_info": ActionType.ALERT_INFO,
            "alert_warning": ActionType.ALERT_WARNING,
            "alert_critical": ActionType.ALERT_CRITICAL,
            "emergency_stop": ActionType.EMERGENCY_STOP,
            "direction_stop": ActionType.DIRECTION_STOP,
            "direction_resume": ActionType.DIRECTION_RESUME,
            "daily_review": ActionType.DAILY_REVIEW,
            "signal_verification": ActionType.SIGNAL_VERIFICATION,
            "performance_report": ActionType.PERFORMANCE_REPORT,
            "no_action": ActionType.NO_ACTION,
        }
        return type_mapping.get(type_str.lower(), ActionType.NO_ACTION)

    def _create_error_decision(self, error_message: str) -> AgentDecision:
        """Create an error decision when parsing fails."""
        return AgentDecision(
            timestamp=datetime.now(),
            analysis=f"Error occurred: {error_message}",
            issues=["Claude API or response parsing error"],
            actions=[
                AgentAction(
                    action_type=ActionType.ALERT_WARNING,
                    detail=f"エージェントでエラーが発生しました: {error_message}",
                    autonomy_level=AutonomyLevel.AUTO_EXECUTE_REPORT,
                    reasoning="エラー発生時は報告する",
                    parameters={"error": error_message},
                )
            ],
            confidence=0.0,
            raw_response=error_message,
        )

    async def generate_daily_review(
        self,
        signals_data: list[dict],
        trades_data: list[dict],
        performance_data: dict,
        market_summary: str,
    ) -> str:
        """
        Generate a daily review report.

        Args:
            signals_data: Today's signals with outcomes
            trades_data: Today's trades with analysis
            performance_data: Performance metrics
            market_summary: Market conditions summary

        Returns:
            Formatted review report in Japanese
        """
        prompt = f"""本日の取引を分析し、日次レビューレポートを作成してください。

## 本日のシグナル
{json.dumps(signals_data, ensure_ascii=False, indent=2)}

## 本日の取引
{json.dumps(trades_data, ensure_ascii=False, indent=2)}

## パフォーマンス指標
{json.dumps(performance_data, ensure_ascii=False, indent=2)}

## 市場状況
{market_summary}

以下の形式でレポートを作成してください：

1. **本日のサマリー**: 取引結果の概要
2. **シグナル精度分析**: 予測の正解率と傾向
3. **エントリー/エグジット評価**: タイミングの良し悪し
4. **問題点**: 検出された問題
5. **改善提案**: 具体的な改善案
6. **明日への注意点**: 注意すべき市場状況やパターン

レポートは日本語で、Telegram送信用に整形してください（絵文字OK）。"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"Failed to generate daily review: {e}")
            return f"日次レビュー生成エラー: {e}"

    async def analyze_signal_outcome(
        self,
        signal: dict,
        actual_price_move: float,
        market_context: dict,
    ) -> dict:
        """
        Analyze a signal's outcome.

        Args:
            signal: The signal data
            actual_price_move: Actual price movement after signal
            market_context: Market conditions at signal time

        Returns:
            Analysis dict with was_correct, analysis, suggestions
        """
        prompt = f"""以下のシグナルを事後検証してください。

## シグナル情報
- 方向: {signal.get('direction')}
- 確信度: {signal.get('confidence'):.2%}
- 価格: ¥{signal.get('price'):,.0f}
- 時刻: {signal.get('timestamp')}
- 特徴量: {json.dumps(signal.get('features', {}), ensure_ascii=False)}

## 実際の結果
- 1時間後の価格変動: {actual_price_move:+.2%}

## 市場状況
{json.dumps(market_context, ensure_ascii=False, indent=2)}

以下のJSON形式で回答してください：
```json
{{
    "was_correct": true/false,
    "analysis": "分析結果",
    "feature_insights": ["特徴量に関する洞察"],
    "suggestions": ["改善提案"]
}}
```"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            raw = response.content[0].text
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                start = raw.find('{')
                end = raw.rfind('}') + 1
                return json.loads(raw[start:end])

        except Exception as e:
            logger.error(f"Failed to analyze signal outcome: {e}")
            return {
                "was_correct": None,
                "analysis": f"分析エラー: {e}",
                "feature_insights": [],
                "suggestions": [],
            }
