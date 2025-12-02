"""Claude API client for Meta AI Agent."""

import json
import re
from datetime import datetime, timedelta
from typing import Any

import anthropic
from loguru import logger

from src.agent.decision import (
    AgentAction,
    AgentDecision,
    ActionType,
    AutonomyLevel,
)
from src.utils.timezone import now_jst


class ClaudeClient:
    """Client for interacting with Claude API."""

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    MAX_TOKENS = 4096

    # Rate limiting settings
    MIN_CALL_INTERVAL = 30  # Minimum seconds between API calls
    MAX_DAILY_CALLS = 100   # Maximum API calls per day

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
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

        # Rate limiting state
        self._last_call_time: datetime | None = None
        self._daily_call_count = 0
        self._daily_reset_date = now_jst().date()

        logger.info(f"Claude client initialized with model: {self.model} (async)")

    def _check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if API call is allowed under rate limits.

        Returns:
            Tuple of (allowed, reason)
        """
        now = now_jst()

        # Reset daily counter if new day
        if now.date() > self._daily_reset_date:
            self._daily_call_count = 0
            self._daily_reset_date = now.date()
            logger.info("Daily API call counter reset")

        # Check daily limit
        if self._daily_call_count >= self.MAX_DAILY_CALLS:
            return False, f"Daily limit reached ({self.MAX_DAILY_CALLS} calls)"

        # Check minimum interval
        if self._last_call_time:
            elapsed = (now - self._last_call_time).total_seconds()
            if elapsed < self.MIN_CALL_INTERVAL:
                return False, f"Rate limit: wait {self.MIN_CALL_INTERVAL - elapsed:.0f}s"

        return True, ""

    def _record_api_call(self) -> None:
        """Record an API call for rate limiting."""
        self._last_call_time = now_jst()
        self._daily_call_count += 1
        logger.debug(f"API call #{self._daily_call_count} today")

    def get_usage_stats(self) -> dict:
        """Get current API usage statistics."""
        return {
            "daily_calls": self._daily_call_count,
            "daily_limit": self.MAX_DAILY_CALLS,
            "remaining_calls": self.MAX_DAILY_CALLS - self._daily_call_count,
            "last_call": self._last_call_time.isoformat() if self._last_call_time else None,
            "reset_date": self._daily_reset_date.isoformat(),
        }

    async def analyze_and_decide(
        self,
        context_prompt: str,
        memory_summary: str = "",
        long_term_memory: str = "",
        system_prompt: str | None = None,
    ) -> AgentDecision:
        """
        Send context to Claude and get a decision.

        Args:
            context_prompt: The current situation context
            memory_summary: Summary of past decisions and outcomes (short-term)
            long_term_memory: Summary of learned insights and rules (long-term)
            system_prompt: Optional custom system prompt

        Returns:
            AgentDecision with analysis and actions
        """
        # Check rate limit
        allowed, reason = self._check_rate_limit()
        if not allowed:
            logger.warning(f"Claude API rate limited: {reason}")
            return self._create_error_decision(f"Rate limited: {reason}")

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        user_prompt = self._build_user_prompt(context_prompt, memory_summary, long_term_memory)

        try:
            # Record API call for rate limiting
            self._record_api_call()

            # Use async API for non-blocking operation
            response = await self.client.messages.create(
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
                f"confidence={decision.confidence:.2f}, "
                f"daily_calls={self._daily_call_count}/{self.MAX_DAILY_CALLS}"
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
            "type": "アクションタイプ（下記参照）",
            "detail": "具体的な内容（日本語）",
            "autonomy_level": "auto_execute|auto_execute_report|propose|emergency",
            "reasoning": "この判断の理由（日本語）",
            "parameters": {}
        }
    ],
    "confidence": 0.85
}
```

## アクションタイプ別パラメータ形式

### param_adjustment / threshold_change
```json
{"type": "param_adjustment", "parameters": {"param_name": "long_confidence_threshold", "new_value": 0.80, "old_value": 0.75}}
```
利用可能なparam_name: long_confidence_threshold, short_confidence_threshold, long_risk_per_trade, short_risk_per_trade, long_max_position_size, short_max_position_size, max_daily_trades, daily_loss_limit

### feature_toggle
```json
{"type": "feature_toggle", "parameters": {"feature_name": "adx_14", "enabled": true}}
```

### feature_importance_update
```json
{"type": "feature_importance_update", "parameters": {"importance": {"adx_14": 0.8, "trend_strength": 0.6}}}
```

### direction_stop / direction_resume
```json
{"type": "direction_stop", "parameters": {"direction": "LONG", "reason": "連続損失"}}
```

### emergency_stop
```json
{"type": "emergency_stop", "parameters": {"mode": "no_new_positions", "message": "急落検知"}}
```

### alert_info / alert_warning / alert_critical
```json
{"type": "alert_warning", "parameters": {}}
```
（パラメータ不要、detailに内容を記載）

### no_action
```json
{"type": "no_action", "parameters": {}}
```

## 重要な注意事項
- 過去の判断と結果から学習し、同じ失敗を繰り返さない
- 不確実な場合は「propose」を選択して人間に判断を委ねる
- 複数の問題がある場合は、優先度の高いものから対処する
- 緊急事態では迷わず「emergency」を選択する"""

    def _build_user_prompt(
        self,
        context_prompt: str,
        memory_summary: str,
        long_term_memory: str = "",
    ) -> str:
        """Build the user prompt with context, short-term and long-term memory."""
        prompt = f"""## 現在の状況
{context_prompt}
"""

        # Long-term memory (learned insights and rules)
        if long_term_memory:
            prompt += f"""
## 長期記憶（学習済みの知識）
以下は過去の経験から学んだ洞察とルールです。これらを判断に活用してください。
ただし、現在の状況に適用できるかを常に検討し、盲目的に従わないでください。

{long_term_memory}
"""

        # Short-term memory (recent decisions)
        if memory_summary:
            prompt += f"""
## 短期記憶（直近の判断履歴）
{memory_summary}
"""

        prompt += """
上記の状況を分析し、必要なアクションを判断してください。

重要：
- 長期記憶の洞察・ルールは参考にしつつ、現在の状況に適切かを判断してください
- 過去の学習が現在の状況に当てはまらない場合は、それを指摘してください
- 新たな洞察があれば、analysisに含めてください

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
        intervention_summary: str = "",
    ) -> str:
        """
        Generate a daily review report.

        Args:
            signals_data: Today's signals with outcomes
            trades_data: Today's trades with analysis
            performance_data: Performance metrics
            market_summary: Market conditions summary
            intervention_summary: Summary of missed interventions analysis

        Returns:
            Formatted review report in Japanese
        """
        # Build intervention section if available
        intervention_section = ""
        if intervention_summary:
            intervention_section = f"""
## 介入分析（見逃し検出）
{intervention_summary}

※「事後判断の難易度」について：
- 明白だった: その時点で予測可能だった問題。改善が必要
- ある程度予測可能だった: シグナルはあったが閾値等で見送り
- 予測困難だった: 事前に予測することが難しかった。参考情報として記録
"""

        prompt = f"""本日の取引を分析し、日次レビュー（反省会）レポートを作成してください。

## 本日のシグナル
{json.dumps(signals_data, ensure_ascii=False, indent=2)}

## 本日の取引
{json.dumps(trades_data, ensure_ascii=False, indent=2)}

## パフォーマンス指標
{json.dumps(performance_data, ensure_ascii=False, indent=2)}

## 市場状況
{market_summary}
{intervention_section}
以下の形式でレポートを作成してください：

1. **本日のサマリー**: 取引結果の概要
2. **シグナル精度分析**: 予測の正解率と傾向
3. **エントリー/エグジット評価**: タイミングの良し悪し
4. **介入タイミング評価**: 見逃しや遅延介入の分析（該当する場合）
5. **問題点**: 検出された問題
6. **改善提案**: 具体的な改善案（優先度付き）
7. **明日への注意点**: 注意すべき市場状況やパターン

レポートは日本語で、Telegram送信用に整形してください（絵文字OK）。
「予測困難だった」ものは参考情報として扱い、「明白だった」ものを重点的に改善提案してください。"""

        try:
            response = await self.client.messages.create(
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

    async def analyze_feature_optimization(
        self,
        feature_registry_summary: dict,
        model_performance: dict,
        signal_accuracy: dict,
        recent_trades: list[dict],
    ) -> dict:
        """
        Analyze features and recommend optimizations.

        Args:
            feature_registry_summary: Summary of all features and their status
            model_performance: Model backtest/live performance metrics
            signal_accuracy: Signal accuracy statistics
            recent_trades: Recent trade data for analysis

        Returns:
            Dict with recommendations for feature changes
        """
        prompt = f"""特徴量の最適化分析を実施してください。

## 現在の特徴量設定
{json.dumps(feature_registry_summary, ensure_ascii=False, indent=2)}

## モデルパフォーマンス
{json.dumps(model_performance, ensure_ascii=False, indent=2)}

## シグナル精度統計
{json.dumps(signal_accuracy, ensure_ascii=False, indent=2)}

## 最近のトレード（直近20件）
{json.dumps(recent_trades[:20], ensure_ascii=False, indent=2)}

以下を分析してください：
1. 現在有効な特徴量の有効性評価
2. 無効化されているが有効化すべき特徴量
3. 有効だが無効化を検討すべき特徴量
4. 新しい特徴量の追加提案（FeatureRegistryにあるものから）

以下のJSON形式で回答してください：
```json
{{
    "analysis": "全体的な分析結果",
    "feature_recommendations": [
        {{
            "feature_name": "特徴量名",
            "action": "enable|disable|update_importance",
            "reason": "理由",
            "importance_score": 0.0-1.0（オプション）,
            "priority": "high|medium|low",
            "autonomy_level": "auto_execute|auto_execute_report|propose"
        }}
    ],
    "retrain_recommended": true/false,
    "retrain_reason": "再学習が推奨される場合の理由",
    "extended_features_to_consider": ["将来追加を検討すべき外部特徴量"],
    "confidence": 0.0-1.0
}}
```

注意：
- 「auto_execute」はリスクが低い変更（重要度スコア更新）に使用
- 「auto_execute_report」は中程度の変更（特徴量のオン/オフ）に使用
- 「propose」はリスクが高い変更（複数特徴量の同時変更、再学習提案）に使用
- 一度に多くの変更を行わない（最大3つまで）
- パフォーマンスが安定している場合は変更を控える"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
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
            logger.error(f"Failed to analyze feature optimization: {e}")
            return {
                "analysis": f"分析エラー: {e}",
                "feature_recommendations": [],
                "retrain_recommended": False,
                "confidence": 0.0,
            }

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
            response = await self.client.messages.create(
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

    async def extract_insights_from_review(
        self,
        daily_review: str,
        performance_data: dict,
        signal_accuracy: dict,
    ) -> dict:
        """
        Extract insights and rules from a daily review.

        Args:
            daily_review: The daily review text
            performance_data: Performance metrics
            signal_accuracy: Signal accuracy statistics

        Returns:
            Dict with extracted insights and rules
        """
        prompt = f"""以下の日次レビューから、長期的に記憶すべき洞察とルールを抽出してください。

## 日次レビュー
{daily_review}

## パフォーマンスデータ
{json.dumps(performance_data, ensure_ascii=False, indent=2)}

## シグナル精度
{json.dumps(signal_accuracy, ensure_ascii=False, indent=2)}

以下の基準で抽出してください：

### 洞察（Insights）
- 繰り返し観察されるパターン
- 特定の条件下で有効な知見
- 市場の特性に関する発見

### ルール（Rules）
- 「〜のときは〜すべき」という行動指針
- 「〜してはいけない」という禁止事項
- 条件付きの判断基準

### 抽出しない基準
- 一度きりの偶然の出来事
- 統計的に有意でない観察
- 過去に既に記録済みの内容（重複）

以下のJSON形式で回答してください：
```json
{{
    "insights": [
        {{
            "category": "市場パターン|シグナル精度|リスク管理|特徴量",
            "title": "短いタイトル",
            "content": "洞察の内容",
            "evidence": ["根拠1", "根拠2"],
            "conditions": ["適用条件1"],
            "confidence": "high|medium|low"
        }}
    ],
    "rules": [
        {{
            "name": "ルール名",
            "type": "do|dont|conditional",
            "content": "ルールの内容",
            "origin": "このルールが生まれた経緯",
            "confidence": "high|medium|low"
        }}
    ],
    "events": [
        {{
            "name": "イベント名",
            "category": "market_crash|market_surge|system_error|strategy_change",
            "severity": "critical|high|medium",
            "impact": "影響の概要",
            "situation": "何が起きたか",
            "lessons": ["教訓1", "教訓2"]
        }}
    ],
    "no_new_insights_reason": "新しい洞察がない場合の理由（オプション）"
}}
```

注意：
- 本当に価値のある洞察のみを抽出してください
- 曖昧な内容や一般論は避けてください
- 既存の知識と重複する可能性がある場合は抽出しないでください"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
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
            logger.error(f"Failed to extract insights: {e}")
            return {
                "insights": [],
                "rules": [],
                "events": [],
                "error": str(e),
            }

    async def validate_memory_items(
        self,
        items_to_validate: list[dict],
        recent_performance: dict,
    ) -> dict:
        """
        Validate existing memory items against recent data.

        Args:
            items_to_validate: List of insights/rules to validate
            recent_performance: Recent performance data

        Returns:
            Dict with validation results for each item
        """
        if not items_to_validate:
            return {"validations": []}

        items_str = json.dumps(items_to_validate, ensure_ascii=False, indent=2)

        prompt = f"""以下の学習済みの洞察・ルールを、最近のパフォーマンスデータに照らして検証してください。

## 検証対象
{items_str}

## 最近のパフォーマンス（過去7日間）
{json.dumps(recent_performance, ensure_ascii=False, indent=2)}

各項目について以下を判定してください：

1. **有効性**: この洞察/ルールは今も有効か？
2. **適用可能性**: 最近のデータで適用する機会があったか？
3. **成功**: 適用した場合、成功したか？

以下のJSON形式で回答してください：
```json
{{
    "validations": [
        {{
            "id": "項目のID",
            "type": "insight|rule",
            "still_valid": true/false,
            "was_applicable": true/false,
            "success": true/false/null,
            "notes": "検証に関するメモ",
            "recommendation": "keep|deprecate|modify",
            "modification_suggestion": "修正が必要な場合の提案"
        }}
    ],
    "overall_assessment": "全体的な評価コメント"
}}
```

注意：
- 検証できるデータがない場合は success を null にしてください
- 明らかに時代遅れになった項目は deprecate を推奨してください
- 修正で改善できる項目は modify を推奨してください"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
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
            logger.error(f"Failed to validate memory items: {e}")
            return {"validations": [], "error": str(e)}
