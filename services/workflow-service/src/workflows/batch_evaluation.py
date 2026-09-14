"""Qwen-backed evaluation for complete three-or-four-turn scoring windows."""

import hashlib
import json
import logging
import os
import re
import secrets
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

QUALITY_DELTAS = {
    "mastered": 3, "strong": 3, "satisfactory": 2, "needs_work": 1,
    "off_topic": 0, "repetitive": 0, "incorrect": 0,
}
COMPLETION_QUALITIES = {"mastered", "strong", "satisfactory"}
IDEMPOTENCY_TTL_SECONDS = 72 * 3600


class BatchEvaluationWorkflow:
    """Evaluate one immutable window and apply its score exactly once."""

    def __init__(self) -> None:
        self._model = os.getenv("BATCH_EVAL_MODEL", os.getenv("QWEN_TEXT_MODEL", "qwen3.7-flash"))
        self._base_url = os.getenv(
            "QWEN_TEXT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self._chat_completions_url = self._validated_chat_completions_url(self._base_url)
        hostname = (urlparse(self._base_url).hostname or "").lower().rstrip(".")
        self._api_key = os.getenv(
            "DASHSCOPE_API_KEY" if hostname.endswith(".maas.aliyuncs.com") else "QWEN3_OMNI_API_KEY"
        )

    async def evaluate_window(
        self, *, user_id: str, goal_id: int, turn_window: List[Dict[str, Any]],
        current_task: Dict[str, Any], native_language: str, db_connection: Any,
        task_id: Optional[int] = None, evaluation_id: Optional[str] = None,
        scoring_generation: int = 0, force_decision: bool = False,
        redis_client: Any = None,
    ) -> Dict[str, Any]:
        """Return a final evaluation or fail-closed ``evaluation_pending``."""
        window_size = len(turn_window)
        if window_size not in (3, 4):
            raise ValueError("turn_window must contain exactly 3 or 4 turns")
        if force_decision and window_size != 4:
            raise ValueError("force_decision is only valid for a 4-turn window")
        turn_ids = [str(turn.get("turn_id") or "").strip() for turn in turn_window]
        if any(not turn_id for turn_id in turn_ids) or len(set(turn_ids)) != window_size:
            raise ValueError("turn_window must contain unique, non-empty turn_id values")
        if scoring_generation < 0:
            raise ValueError("scoring_generation must be non-negative")
        resolved_task_id = task_id if task_id is not None else current_task.get("id")
        if resolved_task_id is None:
            raise ValueError("task_id is required")
        context = await self._read_task_context(
            db_connection, user_id, resolved_task_id, goal_id
        )
        if context["scoring_generation"] != scoring_generation:
            return self._stale_result(
                evaluation_id=evaluation_id or self._derive_evaluation_id(scoring_generation, turn_window),
                requested_generation=scoring_generation,
                task_id=resolved_task_id,
                context=context,
            )

        derived_evaluation_id = self._derive_evaluation_id(scoring_generation, turn_window)
        if evaluation_id is not None and evaluation_id != derived_evaluation_id:
            raise ValueError("evaluation_id does not match scoring_generation and ordered turn_ids")
        evaluation_id = derived_evaluation_id

        cached = await self._read_cached_final(
            db_connection, evaluation_id, user_id, resolved_task_id,
            scoring_generation,
        )
        if cached is not None:
            return self._reconcile_readiness(cached, redis_client)

        try:
            assessment = await self._call_llm(
                turn_window=turn_window, current_task=current_task,
                native_language=native_language,
                target_language=current_task.get("target_language") or "English",
            )
        except Exception as exc:
            logger.warning("[BATCH_EVAL] model=%s evaluation pending (%s)", self._model, type(exc).__name__)
            snapshot = await self._read_task_snapshot(db_connection, user_id, resolved_task_id)
            return self._pending_result(
                evaluation_id=evaluation_id, scoring_generation=scoring_generation,
                task_id=resolved_task_id, snapshot=snapshot,
                native_language=native_language,
            )

        evidence_sufficient = bool(assessment["evidence_sufficient"])
        if window_size == 3 and not evidence_sufficient and not force_decision:
            snapshot = await self._read_task_snapshot(db_connection, user_id, resolved_task_id)
            return {
                **self._base_result(evaluation_id, scoring_generation, resolved_task_id, snapshot),
                "evaluation_status": "insufficient_evidence", "evidence_sufficient": False,
                "quality": None, "delta": 0, "reason": assessment["reason"],
                "window_completed": False, "completed_window_count": None,
            }

        quality = assessment["quality"]
        delta = max(0, min(3, int(QUALITY_DELTAS.get(quality, 0))))
        result = await self._apply_evaluation(
            db_connection=db_connection, user_id=user_id, goal_id=goal_id,
            task_id=resolved_task_id, evaluation_id=evaluation_id,
            scoring_generation=scoring_generation, turn_count=window_size,
            quality=quality, delta=delta, reason=assessment["reason"],
            practice_tip=assessment.get("practice_tip", ""),
        )
        result.pop("_idempotent_replay", False)
        return self._reconcile_readiness(result, redis_client)

    async def _call_llm(
        self, turn_window: List[Dict[str, Any]], current_task: Dict[str, Any],
        native_language: str, target_language: str,
    ) -> Dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("DashScope API key is not configured")
        content = await self._post_chat_completion(messages=[{
            "role": "user",
            "content": self._build_prompt(turn_window, current_task, native_language, target_language),
        }])
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        content = re.sub(r"^```(?:json)?\s*", "", (content or "").strip())
        content = re.sub(r"\s*```$", "", content)
        return self._validate_llm_result(json.loads(content))

    async def _post_chat_completion(self, *, messages: List[Dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._chat_completions_url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": self._model, "messages": messages, "stream": False,
                      "enable_thinking": False, "response_format": {"type": "json_object"}},
            )
            response.raise_for_status()
            payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected Chat Completions response shape") from exc

    @staticmethod
    def _validated_chat_completions_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.query
                or parsed.fragment or not (hostname in {
                    "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
                } or hostname.endswith(".maas.aliyuncs.com"))):
            raise ValueError("QWEN_TEXT_BASE_URL must be an official DashScope HTTPS endpoint")
        if not parsed.path.rstrip("/").endswith("/compatible-mode/v1"):
            raise ValueError("QWEN_TEXT_BASE_URL must end with /compatible-mode/v1")
        return f"{base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _build_prompt(
        turn_window: List[Dict[str, Any]], current_task: Dict[str, Any],
        native_language: str, target_language: str,
    ) -> str:
        turns = "\n".join(
            f"Turn {index + 1}: Student: {turn.get('user_content', '')}\nTutor: {turn.get('ai_response', '')}"
            for index, turn in enumerate(turn_window)
        )
        return f"""Evaluate this complete language-practice window.
Task: {current_task.get('task_description') or current_task.get('text') or ''}
Scenario: {current_task.get('scenario_title') or ''}
Target language: {target_language}
Required concepts/keywords: {current_task.get('keywords') or []}

{turns}

Choose exactly one quality:
- mastered: fully correct, natural, detailed, and directly completes the task
- strong: correct, relevant, and clear with only minor limitations
- satisfactory: meaningful, relevant, understandable task progress
- needs_work: relevant attempt but incomplete or substantially flawed
- off_topic: unrelated or merely a generic greeting
- repetitive: repeats wording without meaningful new task content
- incorrect: wrong meaning or failure of the requested communicative action

For a 3-turn window, evidence_sufficient is false only when a fourth turn is genuinely
needed to make a reliable classification. For a 4-turn window, choose a quality even
when performance is poor. Generic greetings, off-topic content, keyword stuffing,
repetition, and incorrect answers must never be rewarded.

Return strict JSON only:
{{"quality":"<one value above>","evidence_sufficient":true|false,
"reason":"<one short sentence in {native_language}>",
"practice_tip":"<one specific next practice action in {native_language}, with a short example in {target_language}>"}}
The practice_tip must address this task and the student's observed difficulty.
Explain what to say or improve next; do not merely say 'keep practicing'.
Do not promise a score or completion, and do not invent errors unsupported by the window.
Do not return a score or delta; the server owns score mapping."""

    @staticmethod
    def _validate_llm_result(parsed: Any) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("model output must be a JSON object")
        quality = str(parsed.get("quality") or "").strip().lower()
        if quality not in QUALITY_DELTAS:
            raise ValueError("unsupported quality returned by model")
        if not isinstance(parsed.get("evidence_sufficient"), bool):
            raise ValueError("evidence_sufficient must be boolean")
        reason = str(parsed.get("reason") or "").strip()
        if not reason:
            raise ValueError("model returned no reason")
        tip = parsed.get("practice_tip", "")
        # Older cached/model responses remain valid during rolling deployment.
        tip = tip.strip()[:400] if isinstance(tip, str) else ""
        return {"quality": quality, "evidence_sufficient": parsed["evidence_sufficient"],
                "reason": reason[:240], "practice_tip": tip}

    async def _apply_evaluation(
        self, *, db_connection: Any, user_id: str, goal_id: int, task_id: int,
        evaluation_id: str, scoring_generation: int, turn_count: int,
        quality: str, delta: int, reason: str, practice_tip: str = "",
    ) -> Dict[str, Any]:
        async with db_connection.transaction():
            task = await db_connection.fetchrow(
                """SELECT score, status, interaction_count, scoring_generation, goal_id
                   FROM user_tasks WHERE id = $1 AND user_id = $2 FOR UPDATE""",
                task_id, user_id,
            )
            if not task:
                raise ValueError("task not found for user")
            authoritative_goal_id = int(task.get("goal_id") or 0)
            if authoritative_goal_id != int(goal_id):
                raise ValueError("task does not belong to requested goal")
            current_generation = int(task.get("scoring_generation") or 0)
            snapshot = {"score": int(task.get("score") or 0),
                        "interaction_count": int(task.get("interaction_count") or 0)}
            if current_generation != scoring_generation:
                return {
                    **self._base_result(evaluation_id, scoring_generation, task_id, snapshot),
                    "evaluation_status": "stale_generation", "evidence_sufficient": True,
                    "quality": quality, "delta": 0, "reason": reason,
                    "task_ready_to_complete": False,
                }

            await db_connection.execute(
                """DELETE FROM workflow_scoring_evaluations
                   WHERE evaluation_id = $1 AND expires_at <= NOW()""",
                evaluation_id,
            )
            existing = await db_connection.fetchrow(
                """SELECT result FROM workflow_scoring_evaluations
                   WHERE evaluation_id = $1 AND expires_at > NOW()""",
                evaluation_id,
            )
            if existing:
                value = existing.get("result")
                replay = json.loads(value) if isinstance(value, str) else dict(value)
                replay["_idempotent_replay"] = True
                return replay

            completed_window_count = int(await db_connection.fetchval(
                """SELECT COUNT(*) FROM workflow_scoring_evaluations
                   WHERE user_id = $1 AND task_id = $2 AND scoring_generation = $3
                     AND result->>'window_completed' = 'true'""",
                user_id, task_id, scoring_generation,
            ) or 0) + 1

            if task.get("status") == "completed":
                result = {
                    **self._base_result(evaluation_id, scoring_generation, task_id, snapshot),
                    "evaluation_status": "already_completed", "evidence_sufficient": True,
                    "quality": quality, "delta": 0, "reason": reason,
                    "task_completed": True, "task_ready_to_complete": False,
                    "window_completed": True,
                    "completed_window_count": completed_window_count,
                }
            else:
                safe_delta = max(0, min(3, int(delta)))
                score = min(9, snapshot["score"] + safe_delta)
                interaction_count = snapshot["interaction_count"] + turn_count
                ready = (
                    score >= 9
                    and completed_window_count >= 3
                    and interaction_count >= 9
                    and quality in COMPLETION_QUALITIES
                )
                await db_connection.execute(
                    """UPDATE user_tasks SET score = $1, interaction_count = $2, updated_at = NOW()
                       WHERE id = $3 AND user_id = $4 AND scoring_generation = $5""",
                    score, interaction_count, task_id, user_id, scoring_generation,
                )
                if safe_delta:
                    await db_connection.execute(
                        """UPDATE user_goals SET current_proficiency = current_proficiency + $1,
                           updated_at = NOW() WHERE id = $2 AND user_id = $3""",
                        safe_delta, authoritative_goal_id, user_id,
                    )
                result = {
                    **self._base_result(evaluation_id, scoring_generation, task_id,
                                        {"score": score, "interaction_count": interaction_count}),
                    "evaluation_status": "completed", "evidence_sufficient": True,
                    "quality": quality, "delta": safe_delta, "reason": reason,
                    "task_ready_to_complete": ready,
                    "window_completed": True,
                    "completed_window_count": completed_window_count,
                }

            result["practice_tip"] = practice_tip
            readiness_intent = {
                "ready": bool(result.get("task_ready_to_complete", False)),
                "scoring_generation": scoring_generation,
                "order": int(result.get("interaction_count", 0)),
                "user_id": user_id,
            }
            result["readiness_intent"] = readiness_intent
            # Capability publication occurs only after this transaction commits.
            result["ready_token"] = None
            result["task_ready_to_complete"] = False

            await db_connection.execute(
                """INSERT INTO workflow_scoring_evaluations
                   (evaluation_id, user_id, task_id, scoring_generation, result, expires_at)
                   VALUES ($1, $2, $3, $4, $5::jsonb, NOW() + INTERVAL '72 hours')""",
                evaluation_id, user_id, task_id, scoring_generation,
                json.dumps(result, ensure_ascii=False),
            )
            return result

    @staticmethod
    async def _read_task_context(
        db_connection: Any, user_id: str, task_id: int, goal_id: int,
    ) -> Dict[str, int]:
        task = await db_connection.fetchrow(
            """SELECT goal_id, scoring_generation, score, interaction_count
               FROM user_tasks WHERE id = $1 AND user_id = $2""",
            task_id, user_id,
        )
        if not task:
            raise ValueError("task not found for user")
        if int(task.get("goal_id") or 0) != int(goal_id):
            raise ValueError("task does not belong to requested goal")
        return {
            "goal_id": int(task.get("goal_id") or 0),
            "scoring_generation": int(task.get("scoring_generation") or 0),
            "score": int(task.get("score") or 0),
            "interaction_count": int(task.get("interaction_count") or 0),
        }

    @staticmethod
    async def _read_cached_final(
        db_connection: Any, evaluation_id: str, user_id: str, task_id: int,
        scoring_generation: int,
    ) -> Optional[Dict[str, Any]]:
        row = await db_connection.fetchrow(
            """SELECT evaluation.result
               FROM workflow_scoring_evaluations AS evaluation
               JOIN user_tasks AS task
                 ON task.id = evaluation.task_id AND task.user_id = evaluation.user_id
               WHERE evaluation.evaluation_id = $1
                 AND evaluation.user_id = $2 AND evaluation.task_id = $3
                 AND evaluation.scoring_generation = $4
                 AND task.scoring_generation = $4
                 AND evaluation.expires_at > NOW()""",
            evaluation_id, user_id, task_id, scoring_generation,
        )
        if not row:
            return None
        value = row.get("result")
        return json.loads(value) if isinstance(value, str) else dict(value)

    def _stale_result(
        self, *, evaluation_id: str, requested_generation: int, task_id: int,
        context: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            **self._base_result(
                evaluation_id, requested_generation, task_id,
                {"score": context["score"], "interaction_count": context["interaction_count"]},
            ),
            "evaluation_status": "stale_generation",
            "current_scoring_generation": context["scoring_generation"],
            "evidence_sufficient": False,
            "quality": None,
            "delta": 0,
            "reason": "This evaluation belongs to an older scoring generation.",
        }

    @staticmethod
    async def _read_task_snapshot(db_connection: Any, user_id: str, task_id: int) -> Dict[str, int]:
        task = await db_connection.fetchrow(
            "SELECT score, interaction_count FROM user_tasks WHERE id = $1 AND user_id = $2",
            task_id, user_id,
        )
        return {"score": int(task.get("score") or 0) if task else 0,
                "interaction_count": int(task.get("interaction_count") or 0) if task else 0}

    def _pending_result(
        self, *, evaluation_id: str, scoring_generation: int, task_id: int,
        snapshot: Dict[str, int], native_language: str,
    ) -> Dict[str, Any]:
        language = (native_language or "").lower()
        if language in {"chinese", "zh", "zh-cn", "中文"}:
            reason = "评分服务暂时不可用，请稍后重试。"
        elif language in {"japanese", "ja", "日本語"}:
            reason = "評価サービスは一時的に利用できません。後でもう一度お試しください。"
        else:
            reason = "Evaluation is temporarily unavailable; please retry later."
        return {
            **self._base_result(evaluation_id, scoring_generation, task_id, snapshot),
            "evaluation_status": "evaluation_pending", "evidence_sufficient": False,
            "quality": None, "delta": 0, "reason": reason,
            "window_completed": False, "completed_window_count": None,
        }

    def _base_result(
        self, evaluation_id: str, scoring_generation: int, task_id: int,
        snapshot: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "evaluation_id": evaluation_id, "scoring_generation": scoring_generation,
            "task_id": task_id, "score": snapshot["score"],
            "interaction_count": snapshot["interaction_count"], "task_completed": False,
            "task_ready_to_complete": False, "ready_token": None, "model_id": self._model,
            "window_completed": False, "completed_window_count": None,
        }

    @staticmethod
    def _derive_evaluation_id(scoring_generation: int, turn_window: List[Dict[str, Any]]) -> str:
        turn_ids = [str(turn.get("turn_id") or turn.get("turn_order")) for turn in turn_window]
        return hashlib.sha256(f"{scoring_generation}\0".encode() + "\0".join(turn_ids).encode()).hexdigest()

    @staticmethod
    def _ready_key(user_id: str, task_id: int) -> str:
        digest = hashlib.sha256(f"{user_id}\0{task_id}".encode()).hexdigest()
        return f"task_ready:v1:{digest}"

    def _reconcile_readiness(
        self, result: Dict[str, Any], redis_client: Any,
    ) -> Dict[str, Any]:
        reconciled = dict(result)
        intent = reconciled.get("readiness_intent") or {}
        if not intent or reconciled.get("evaluation_status") not in {
            "completed", "already_completed"
        }:
            return reconciled
        token = self._update_ready_gate(
            redis_client=redis_client,
            user_id=str(reconciled.get("user_id") or intent.get("user_id") or ""),
            task_id=int(reconciled["task_id"]),
            scoring_generation=int(intent["scoring_generation"]),
            ready=bool(intent["ready"]),
            order=int(intent["order"]),
        )
        reconciled["ready_token"] = token
        reconciled["task_ready_to_complete"] = bool(token) and bool(intent["ready"])
        if reconciled.get("task_completed") or reconciled["task_ready_to_complete"]:
            blocker = None
        elif intent["ready"]:
            blocker = "readiness_unavailable"
        elif int(reconciled.get("score") or 0) < 9:
            blocker = "score"
        elif (int(reconciled.get("completed_window_count") or 0) < 3
              or int(reconciled.get("interaction_count") or 0) < 9):
            blocker = "windows"
        else:
            blocker = "quality"
        reconciled["completion_blocker"] = blocker
        return reconciled

    @classmethod
    def _update_ready_gate(
        cls, *, redis_client: Any, user_id: str, task_id: int,
        scoring_generation: int, ready: bool, order: int,
    ) -> Optional[str]:
        if redis_client is None:
            return None
        token = secrets.token_urlsafe(24) if ready else ""
        generation = max(0, int(scoring_generation))
        order = max(0, int(order))
        value = f"{generation}:{order}:{token}"
        try:
            applied = redis_client.eval(
                """
                local current = redis.call('GET', KEYS[1])
                if current then
                  local first = string.find(current, ':')
                  local second = first and string.find(current, ':', first + 1)
                  -- The retired per-turn scorer stored "turn_order:token" under
                  -- the same v1 key.  Its timestamp-like order is not comparable
                  -- with the batch scorer's "generation:interaction_count:token"
                  -- format and must never be interpreted as a generation.  Only
                  -- apply monotonic checks to a value that has both separators;
                  -- any legacy/malformed value is atomically upgraded below.
                  if first and second then
                    local current_generation = tonumber(string.sub(current, 1, first - 1)) or 0
                    local current_order = tonumber(string.sub(current, first + 1, second - 1)) or 0
                    if current_generation > tonumber(ARGV[1]) then return '__STALE__' end
                    if current_generation == tonumber(ARGV[1]) and current_order > tonumber(ARGV[2]) then return '__STALE__' end
                    if current_generation == tonumber(ARGV[1]) and current_order == tonumber(ARGV[2]) then return current end
                  end
                end
                redis.call('SETEX', KEYS[1], ARGV[3], ARGV[4])
                return ARGV[4]
                """,
                1,
                cls._ready_key(user_id, task_id),
                generation,
                order,
                IDEMPOTENCY_TTL_SECONDS,
                value,
            )
            if not applied or applied == "__STALE__":
                return None
            current = applied.decode() if isinstance(applied, bytes) else str(applied)
            parts = current.split(":", 2)
            return parts[2] if len(parts) == 3 and parts[2] else None
        except Exception as exc:
            logger.warning("[BATCH_EVAL] Redis readiness write failed: %s", type(exc).__name__)
            return None


batch_evaluation_workflow = BatchEvaluationWorkflow()
