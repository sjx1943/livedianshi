import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from workflows.batch_evaluation import BatchEvaluationWorkflow, QUALITY_DELTAS


TASK = {
    "id": 42,
    "task_description": "Order coffee politely",
    "scenario_title": "Coffee shop",
    "target_language": "English",
    "keywords": ["coffee", "please"],
}


def window(size=3, prefix="window"):
    return [
        {"turn_id": f"{prefix}-turn-{i}", "turn_order": i,
         "user_content": f"A coffee please, attempt {i}.", "ai_response": "Certainly."}
        for i in range(1, size + 1)
    ]


class FakeRedis:
    def __init__(self):
        self.values = {}

    def setex(self, key, ttl, value):
        self.values[key] = value

    def eval(self, script, numkeys, key, generation, order, ttl, value):
        current = str(self.values.get(key) or "")
        if current:
            parts = current.split(":", 2)
            if len(parts) == 3:
                current_generation, current_order, _ = parts
                if int(current_generation) > int(generation):
                    return "__STALE__"
                if int(current_generation) == int(generation) and int(current_order) > int(order):
                    return "__STALE__"
                if int(current_generation) == int(generation) and int(current_order) == int(order):
                    return current
        self.values[key] = value
        return value


class FakeDB:
    def __init__(self, *, score=0, interaction_count=0, generation=0, status="pending"):
        self.task = {"score": score, "interaction_count": interaction_count,
                     "goal_id": 7,
                     "scoring_generation": generation, "status": status}
        self.evaluations = {}
        self.task_updates = 0
        self.goal_delta = 0

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetchrow(self, query, *args):
        if "workflow_scoring_evaluations" in query:
            value = self.evaluations.get(args[0])
            return {"result": value} if value is not None else None
        return dict(self.task)

    async def fetchval(self, query, *args):
        return sum(
            1 for result in self.evaluations.values()
            if result.get("scoring_generation") == args[2]
            and result.get("window_completed") is True
        )

    async def execute(self, query, *args):
        if "UPDATE user_tasks" in query:
            self.task["score"], self.task["interaction_count"] = args[:2]
            self.task_updates += 1
        elif "UPDATE user_goals" in query:
            self.goal_delta += args[0]
        elif "INSERT INTO workflow_scoring_evaluations" in query:
            self.evaluations[args[0]] = json.loads(args[4])


async def evaluate(workflow, db, quality="strong", evidence=True, *, size=3,
                   evaluation_id="eval-1", redis=None, force_decision=False):
    turns = window(size, evaluation_id)
    derived_id = workflow._derive_evaluation_id(0, turns)
    assessment = {"quality": quality, "evidence_sufficient": evidence,
                  "reason": f"{quality} reason"}
    with patch.object(workflow, "_call_llm", new=AsyncMock(return_value=assessment)):
        return await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42,
            evaluation_id=derived_id,
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db, redis_client=redis,
            force_decision=force_decision,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("quality,delta", QUALITY_DELTAS.items())
async def test_server_maps_every_quality_and_clamps_delta(quality, delta):
    workflow = BatchEvaluationWorkflow()
    result = await evaluate(workflow, FakeDB(), quality)
    assert result["evaluation_status"] == "completed"
    assert result["quality"] == quality
    assert result["delta"] == delta
    assert 0 <= result["delta"] <= 3


def test_model_supplied_delta_is_ignored():
    parsed = BatchEvaluationWorkflow._validate_llm_result({
        "quality": "strong", "evidence_sufficient": True,
        "reason": "Good work.", "delta": 10,
    })
    assert "delta" not in parsed
    assert QUALITY_DELTAS[parsed["quality"]] == 3


@pytest.mark.asyncio
async def test_three_turns_with_insufficient_evidence_stays_open_without_write():
    db = FakeDB()
    result = await evaluate(BatchEvaluationWorkflow(), db, evidence=False)
    assert result["evaluation_status"] == "insufficient_evidence"
    assert result["evidence_sufficient"] is False
    assert result["window_completed"] is False
    assert result["delta"] == 0
    assert db.task_updates == 0
    assert not db.evaluations


@pytest.mark.asyncio
async def test_force_decision_requires_a_four_turn_window():
    workflow = BatchEvaluationWorkflow()
    turns = window(size=3, prefix="forced-too-early")
    with pytest.raises(ValueError, match="only valid for a 4-turn"):
        await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42,
            evaluation_id=workflow._derive_evaluation_id(0, turns),
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=FakeDB(),
            force_decision=True,
        )


@pytest.mark.asyncio
async def test_fourth_turn_forces_quality_decision():
    db = FakeDB()
    result = await evaluate(
        BatchEvaluationWorkflow(), db, "needs_work", evidence=False, size=4
    )
    assert result["evaluation_status"] == "completed"
    assert result["evidence_sufficient"] is True
    assert result["window_completed"] is True
    assert result["delta"] == 1
    assert result["interaction_count"] == 4


@pytest.mark.asyncio
async def test_qwen_failure_is_pending_and_never_writes_or_falls_back():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    with patch.object(workflow, "_call_llm", new=AsyncMock(side_effect=TimeoutError("slow"))):
        turns = window(prefix="pending")
        result = await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42,
            evaluation_id=workflow._derive_evaluation_id(0, turns),
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    assert result["evaluation_status"] == "evaluation_pending"
    assert result["delta"] == 0
    assert result["window_completed"] is False
    assert db.task_updates == 0
    assert db.goal_delta == 0
    assert not db.evaluations


@pytest.mark.asyncio
async def test_duplicate_evaluation_id_is_idempotent():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    first = await evaluate(workflow, db, "strong")
    second = await evaluate(workflow, db, "needs_work")
    assert second == first
    assert db.task_updates == 1
    assert db.task["score"] == 3
    assert db.task["interaction_count"] == 3


@pytest.mark.asyncio
async def test_duplicate_is_returned_before_second_qwen_call_and_is_unchanged():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    first_model = AsyncMock(return_value={
        "quality": "strong", "evidence_sufficient": True, "reason": "first",
    })
    turns = window(prefix="stable")
    stable_id = workflow._derive_evaluation_id(0, turns)
    with patch.object(workflow, "_call_llm", new=first_model):
        first = await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42, evaluation_id=stable_id,
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    second_model = AsyncMock(side_effect=AssertionError("Qwen must not run"))
    with patch.object(workflow, "_call_llm", new=second_model):
        replay = await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42, evaluation_id=stable_id,
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    second_model.assert_not_awaited()
    assert replay == first
    assert replay["reason"] == "first"


@pytest.mark.asyncio
async def test_task_goal_mismatch_is_rejected_without_write():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    with patch.object(workflow, "_call_llm", new=AsyncMock(return_value={
        "quality": "strong", "evidence_sufficient": True, "reason": "good",
    })):
        turns = window(prefix="wrong-goal")
        with pytest.raises(ValueError, match="requested goal"):
            await workflow.evaluate_window(
                user_id="u1", goal_id=999, task_id=42,
                evaluation_id=workflow._derive_evaluation_id(0, turns),
                scoring_generation=0, turn_window=turns, current_task=TASK,
                native_language="Chinese", db_connection=db,
            )
    assert db.task_updates == 0


@pytest.mark.asyncio
async def test_old_scoring_generation_is_rejected_without_write():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB(generation=1)
    result = await evaluate(workflow, db, "strong")
    assert result["evaluation_status"] == "stale_generation"
    assert result["delta"] == 0
    assert result["window_completed"] is False
    assert db.task_updates == 0
    assert not db.evaluations


@pytest.mark.asyncio
async def test_cached_final_is_rejected_after_generation_reset_before_qwen():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    turns = window(prefix="reset-replay")
    evaluation_id = workflow._derive_evaluation_id(0, turns)
    with patch.object(workflow, "_call_llm", new=AsyncMock(return_value={
        "quality": "strong", "evidence_sufficient": True, "reason": "first",
    })):
        await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42, evaluation_id=evaluation_id,
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    db.task.update(score=0, interaction_count=0, scoring_generation=1)
    model = AsyncMock(side_effect=AssertionError("stale replay must not call Qwen"))
    with patch.object(workflow, "_call_llm", new=model):
        replay = await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42, evaluation_id=evaluation_id,
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    model.assert_not_awaited()
    assert replay["evaluation_status"] == "stale_generation"
    assert replay["score"] == 0


@pytest.mark.asyncio
async def test_caller_cannot_alias_a_different_window_evaluation_id():
    workflow = BatchEvaluationWorkflow()
    turns = window(prefix="canonical")
    with pytest.raises(ValueError, match="evaluation_id does not match"):
        await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42, evaluation_id="forged-id",
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=FakeDB(),
        )


@pytest.mark.asyncio
async def test_readiness_is_not_published_when_transaction_insert_fails():
    workflow = BatchEvaluationWorkflow()

    class InsertFailDB(FakeDB):
        async def execute(self, query, *args):
            if "INSERT INTO workflow_scoring_evaluations" in query:
                raise RuntimeError("insert failed")
            return await super().execute(query, *args)

    gate = AsyncMock()
    turns = window(prefix="commit-first")
    with patch.object(workflow, "_call_llm", new=AsyncMock(return_value={
        "quality": "strong", "evidence_sufficient": True, "reason": "good",
    })), patch.object(workflow, "_update_ready_gate", gate):
        with pytest.raises(RuntimeError, match="insert failed"):
            await workflow.evaluate_window(
                user_id="u1", goal_id=7, task_id=42,
                evaluation_id=workflow._derive_evaluation_id(0, turns),
                scoring_generation=0, turn_window=turns, current_task=TASK,
                native_language="Chinese", db_connection=InsertFailDB(),
            )
    gate.assert_not_called()


@pytest.mark.asyncio
async def test_three_strong_windows_reach_nine_and_latest_is_ready():
    workflow = BatchEvaluationWorkflow()
    db = FakeDB()
    redis = FakeRedis()
    results = [
        await evaluate(workflow, db, "strong", evaluation_id=f"eval-{i}", redis=redis)
        for i in range(1, 4)
    ]
    assert [result["score"] for result in results] == [3, 6, 9]
    assert [result["completed_window_count"] for result in results] == [1, 2, 3]
    assert results[-1]["interaction_count"] == 9
    assert results[-1]["task_ready_to_complete"] is True
    assert results[-1]["ready_token"]
    assert results[-1]["task_completed"] is False


@pytest.mark.asyncio
async def test_task_372_sequence_explains_99_then_next_satisfactory_window_is_ready():
    workflow, db, redis = BatchEvaluationWorkflow(), FakeDB(), FakeRedis()
    qualities = ["satisfactory", "needs_work", "needs_work", "satisfactory"] + ["needs_work"] * 5
    for index, quality in enumerate(qualities):
        result = await evaluate(workflow, db, quality, evaluation_id=f"incident-{index}", redis=redis)
    assert (result["score"], result["interaction_count"], result["completed_window_count"]) == (9, 27, 9)
    assert result["completion_blocker"] == "quality"
    assert not result["task_ready_to_complete"]
    assert result["ready_token"] is None
    for quality in ("off_topic", "repetitive", "incorrect"):
        result = await evaluate(workflow, db, quality, evaluation_id=quality, redis=redis)
        assert result["delta"] == 0
        assert result["completion_blocker"] == "quality"
        assert not result["task_ready_to_complete"]
    result = await evaluate(workflow, db, "satisfactory", evaluation_id="recovery", redis=redis)
    assert result["task_ready_to_complete"] is True
    assert result["task_completed"] is False  # Explicit user confirmation still required.
    assert result["completion_blocker"] is None
    assert redis.values[workflow._ready_key("u1", 42)].endswith(":" + result["ready_token"])


@pytest.mark.asyncio
async def test_readiness_outage_is_distinct_and_replay_recovers_without_scoring_again():
    workflow, db, redis = BatchEvaluationWorkflow(), FakeDB(), FakeRedis()
    for index in range(2):
        await evaluate(workflow, db, evaluation_id=f"ready-{index}", redis=redis)
    result = await evaluate(workflow, db, evaluation_id="ready-2", redis=None)
    assert result["completion_blocker"] == "readiness_unavailable"
    updates = db.task_updates
    recovered = await evaluate(workflow, db, evaluation_id="ready-2", redis=redis)
    assert recovered["task_ready_to_complete"] is True
    assert recovered["completion_blocker"] is None
    assert db.task_updates == updates


def test_practice_tip_is_bounded_optional_and_prompt_requests_specific_example():
    result = BatchEvaluationWorkflow._validate_llm_result({
        "quality": "needs_work", "evidence_sufficient": True,
        "reason": "Missing a detail", "practice_tip": "x" * 500,
    })
    assert len(result["practice_tip"]) == 400
    prompt = BatchEvaluationWorkflow._build_prompt(window(), TASK, "Chinese", "English")
    assert "one specific next practice action in Chinese" in prompt
    assert "short example in English" in prompt


@pytest.mark.asyncio
async def test_zero_delta_closed_window_still_counts_actual_turns():
    db = FakeDB()
    result = await evaluate(BatchEvaluationWorkflow(), db, "off_topic", size=4)
    assert result["delta"] == 0
    assert result["window_completed"] is True
    assert result["interaction_count"] == 4
    assert db.task_updates == 1
    assert db.goal_delta == 0


def test_readiness_gate_is_generation_aware_and_monotonic():
    redis = FakeRedis()
    workflow = BatchEvaluationWorkflow()
    newest_token = workflow._update_ready_gate(
        redis_client=redis, user_id="u1", task_id=42,
        scoring_generation=2, ready=True, order=12,
    )
    key = workflow._ready_key("u1", 42)
    newest_value = redis.values[key]
    assert newest_token

    assert workflow._update_ready_gate(
        redis_client=redis, user_id="u1", task_id=42,
        scoring_generation=2, ready=False, order=9,
    ) is None
    assert redis.values[key] == newest_value

    assert workflow._update_ready_gate(
        redis_client=redis, user_id="u1", task_id=42,
        scoring_generation=1, ready=False, order=99,
    ) is None
    assert redis.values[key] == newest_value

    workflow._update_ready_gate(
        redis_client=redis, user_id="u1", task_id=42,
        scoring_generation=3, ready=False, order=3,
    )
    assert redis.values[key] == "3:3:"


def test_readiness_gate_atomically_upgrades_legacy_turn_order_value():
    redis = FakeRedis()
    workflow = BatchEvaluationWorkflow()
    key = workflow._ready_key("u1", 42)
    # The retired per-turn scorer used "turn_order:token". Production had an
    # empty-token timestamp in this format, which the batch scorer previously
    # misread as a future scoring_generation and rejected forever as stale.
    redis.values[key] = "1788094759335706:"

    token = workflow._update_ready_gate(
        redis_client=redis, user_id="u1", task_id=42,
        scoring_generation=0, ready=True, order=19,
    )

    assert token
    assert redis.values[key] == f"0:19:{token}"


@pytest.mark.asyncio
async def test_invalid_json_is_pending():
    workflow = BatchEvaluationWorkflow()
    workflow._api_key = "test-key"
    db = FakeDB()
    with patch.object(workflow, "_post_chat_completion", new=AsyncMock(return_value="not json")):
        turns = window(prefix="bad-json")
        result = await workflow.evaluate_window(
            user_id="u1", goal_id=7, task_id=42,
            evaluation_id=workflow._derive_evaluation_id(0, turns),
            scoring_generation=0, turn_window=turns, current_task=TASK,
            native_language="Chinese", db_connection=db,
        )
    assert result["evaluation_status"] == "evaluation_pending"
    assert db.task_updates == 0
