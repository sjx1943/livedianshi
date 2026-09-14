import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ._omni_stubs import load_main


omni = load_main()


@pytest.mark.asyncio
async def test_restored_feedback_is_task_generation_and_score_bound_without_capability():
    redis = FakeRedis()
    current = {**task(), "score": 9, "interaction_count": 27}
    result = {"task_id": 9, "scoring_generation": 2, "score": 9, "interaction_count": 27,
              "completion_blocker": "quality", "reason": "Add the size.",
              "practice_tip": "A small latte, please.", "ready_token": "must-not-restore"}
    redis.data[omni._scoring_window_key("u1", 4, 9, 2)] = json.dumps({"completed_results": [{"result": result}]})
    restored = await omni._restore_progress_feedback(redis, "u1", 4, current)
    assert restored["practice_tip"] == result["practice_tip"]
    assert "ready_token" not in restored
    assert await omni._restore_progress_feedback(redis, "u2", 4, current) is None
    assert await omni._restore_progress_feedback(redis, "u1", 4, {**current, "scoring_generation": 3}) is None
    assert await omni._restore_progress_feedback(redis, "u1", 4, {**current, "score": 0}) is None


@pytest.mark.asyncio
async def test_feedback_is_forwarded_and_qualified_window_clears_blocker():
    callback = callback_for()
    current = task()
    for quality, ready in [("needs_work", False), ("satisfactory", True)]:
        result = {
            "score": 9, "interaction_count": 30 if ready else 27,
            "scoring_generation": 2, "task_ready_to_complete": ready,
            "ready_token": "token-for-confirmation" if ready else None,
            "completion_blocker": None if ready else "quality",
            "practice_tip": "Specify a size: A small latte, please.",
            "reason": "Add the drink size.", "quality": quality,
        }
        emitted = await omni._emit_scoring_result(callback, current, 9, [], result, "token")
        payload = callback._safe_send.await_args.args[0]["payload"]
        assert payload["practice_tip"] == result["practice_tip"]
        assert payload["completion_blocker"] == result["completion_blocker"]
        assert emitted["task_ready_to_complete"] == ready
        assert emitted["scoring_generation"] == 2


@pytest.mark.asyncio
async def test_readiness_pending_keeps_window_retryable_without_positive_progress():
    redis, callback, current = FakeRedis(), callback_for(), task()
    result = {"evaluation_status": "readiness_pending", "completion_blocker": "readiness_unavailable",
              "score": 9, "interaction_count": 27}
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", AsyncMock(return_value=result)
    ):
        for turn_id in ("t1", "t2", "t3"):
            await add_turn(callback, turn_id, current)
    state = json.loads(redis.data[omni._scoring_window_key("u1", 4, 9, 2)])
    assert state["frozen"] is True
    assert len(state["turns"]) == 3
    callback._safe_send.assert_awaited_once()
    assert callback._safe_send.await_args.args[0]["type"] == "scoring_feedback"
    assert callback._safe_send.await_args.args[0]["payload"]["completion_blocker"] == "readiness_unavailable"


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}
        self.lists = {}

    async def get(self, key):
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttls[key] = ttl

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return False
        self.data[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def eval(self, script, numkeys, *args):
        if numkeys == 2:
            state_key, lock_key, owner, ttl, value = args
            if self.data.get(lock_key) != owner:
                return 0
            self.data[state_key] = value
            self.ttls[state_key] = int(ttl)
            return 1
        if "RPUSH" in script:
            key, ttl, value = args
            self.lists.setdefault(key, []).append(value)
            self.ttls[key] = int(ttl)
            return 1
        if "LRANGE" in script:
            (key,) = args
            values = self.lists.pop(key, [])
            self.ttls.pop(key, None)
            return values
        lock_key, owner = args
        if self.data.get(lock_key) != owner:
            return 0
        del self.data[lock_key]
        self.ttls.pop(lock_key, None)
        return 1


def callback_for(turn_id="t1"):
    callback = SimpleNamespace(
        messages=[{"role": "user", "turn_id": turn_id, "content": "answer"}],
        current_turn_id=turn_id,
        user_context={"active_goal": {"current_proficiency": 0}},
        token="token",
        scenario="Cafe",
        processed_turn_ids=set(),
        turn_evaluations_inflight=set(),
        _safe_send=AsyncMock(),
    )
    return callback


def task():
    return {
        "id": 9,
        "task_description": "Order coffee",
        "scenario_title": "Cafe",
        "score": 0,
        "interaction_count": 0,
        "scoring_generation": 2,
    }


async def add_turn(callback, turn_id, current_task):
    callback.current_turn_id = turn_id
    callback.messages.append({
        "role": "user", "turn_id": turn_id, "content": f"answer {turn_id}"
    })
    return await omni._handle_turn_with_accumulator(
        callback, None, None, "u1", 4, 9, f"answer {turn_id}",
        f"reply {turn_id}", current_task, "Chinese", "token",
    )


@pytest.mark.asyncio
async def test_first_two_turns_only_accumulate_and_third_scores():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    evaluated = {
        "evaluation_status": "evaluated", "evidence_sufficient": True,
        "quality": "strong", "delta": 3, "score": 3,
        "interaction_count": 3, "task_completed": False,
        "scoring_generation": 2,
    }
    post = AsyncMock(return_value=evaluated)
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", post
    ):
        assert await add_turn(callback, "t1", current_task) is None
        assert await add_turn(callback, "t2", current_task) is None
        assert post.await_count == 0
        result = await add_turn(callback, "t3", current_task)

    assert result["proficiency_delta"] == 3
    payload = post.await_args.args[0]
    assert [turn["turn_id"] for turn in payload["turn_window"]] == ["t1", "t2", "t3"]
    assert [turn["turn_index"] for turn in payload["turn_window"]] == [1, 2, 3]
    assert all(isinstance(turn["turn_order"], int) for turn in payload["turn_window"])
    assert payload["scoring_generation"] == 2
    assert payload["force_decision"] is False
    callback._safe_send.assert_awaited_once()
    event = callback._safe_send.await_args.args[0]["payload"]
    assert event["evaluation_status"] == "completed"
    assert event["window_completed"] is True
    assert event["scoring_generation"] == 2


@pytest.mark.asyncio
async def test_insufficient_third_turn_is_retained_for_forced_fourth():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    post = AsyncMock(side_effect=[
        {"evaluation_status": "evaluated", "evidence_sufficient": False},
        {
            "evaluation_status": "evaluated", "evidence_sufficient": True,
            "quality": "satisfactory", "delta": 2, "score": 2,
            "interaction_count": 4, "task_completed": False,
        },
    ])
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", post
    ):
        for turn_id in ("t1", "t2", "t3"):
            await add_turn(callback, turn_id, current_task)
        assert callback._safe_send.await_count == 0
        result = await add_turn(callback, "t4", current_task)

    assert result["proficiency_delta"] == 2
    assert len(post.await_args_list[1].args[0]["turn_window"]) == 4
    assert post.await_args_list[1].args[0]["force_decision"] is True


@pytest.mark.asyncio
async def test_duplicate_turn_id_is_not_added_or_scored_twice():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    post = AsyncMock(return_value={
        "evaluation_status": "evaluated", "evidence_sufficient": True,
        "delta": 1, "score": 1, "interaction_count": 3,
    })
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", post
    ):
        await add_turn(callback, "t1", current_task)
        await add_turn(callback, "t1", current_task)
        await add_turn(callback, "t2", current_task)
        assert post.await_count == 0
        await add_turn(callback, "t3", current_task)

    assert post.await_count == 1


@pytest.mark.asyncio
async def test_failed_window_freezes_and_queues_new_turn_without_progress():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    post = AsyncMock(return_value=None)
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", post
    ):
        for turn_id in ("t1", "t2", "t3", "t4"):
            await add_turn(callback, turn_id, current_task)

    key = omni._scoring_window_key("u1", 4, 9, 2)
    state = json.loads(redis.data[key])
    assert [turn["turn_id"] for turn in state["turns"]] == ["t1", "t2", "t3"]
    assert [turn["turn_id"] for turn in state["queue"]] == ["t4"]
    assert state["frozen"] is True
    assert callback._safe_send.await_count > 0
    for call in callback._safe_send.await_args_list:
        assert call.args[0]["type"] == "scoring_feedback"
        assert call.args[0]["payload"]["completion_blocker"] == "evaluation_unavailable"
        assert "delta" not in call.args[0]["payload"]
    assert redis.ttls[key] == 72 * 3600


@pytest.mark.asyncio
async def test_workflow_failure_retries_at_one_two_four_seconds():
    responses = [
        SimpleNamespace(status_code=503),
        SimpleNamespace(status_code=401),
        SimpleNamespace(status_code=200, json=lambda: {
            "data": {"evaluation_status": "evaluation_pending"}
        }),
        SimpleNamespace(status_code=200, json=lambda: {
            "data": {"evaluation_status": "evaluated", "delta": 0}
        }),
    ]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return responses.pop(0)

    sleep = AsyncMock()
    payload = {"evaluation_id": "abc", "turn_window": [{}, {}, {}]}
    with patch.object(omni.httpx, "AsyncClient", return_value=Client()), patch.object(
        omni.asyncio, "sleep", sleep
    ):
        result = await omni._post_scoring_window(payload, "token")

    assert result["evaluation_status"] == "evaluated"
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2, 4]


@pytest.mark.asyncio
async def test_missing_required_ready_token_retries_same_evaluation():
    responses = [
        SimpleNamespace(status_code=200, json=lambda: {"data": {
            "evaluation_status": "completed",
            "readiness_intent": {"ready": True},
            "ready_token": None,
        }}),
        SimpleNamespace(status_code=200, json=lambda: {"data": {
            "evaluation_status": "completed",
            "readiness_intent": {"ready": True},
            "ready_token": "ready-token",
        }}),
    ]

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return responses.pop(0)

    sleep = AsyncMock()
    payload = {"evaluation_id": "same-eval", "turn_window": [{}, {}, {}]}
    with patch.object(omni.httpx, "AsyncClient", return_value=Client()), patch.object(
        omni.asyncio, "sleep", sleep
    ):
        result = await omni._post_scoring_window(payload, "token")

    assert result["ready_token"] == "ready-token"
    sleep.assert_awaited_once_with(1)


def test_reset_generation_uses_a_new_window_and_evaluation_identity():
    turns = [{"turn_id": "t1"}, {"turn_id": "t2"}, {"turn_id": "t3"}]
    assert omni._scoring_window_key("u", 1, 2, 0) != omni._scoring_window_key(
        "u", 1, 2, 1
    )
    assert omni._scoring_evaluation_id(0, turns) != omni._scoring_evaluation_id(1, turns)


@pytest.mark.asyncio
async def test_stale_generation_is_discarded_without_websocket_update():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    post = AsyncMock(return_value={
        "evaluation_status": "stale_generation", "evidence_sufficient": True,
        "delta": 0,
    })
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", post
    ):
        for turn_id in ("t1", "t2", "t3"):
            await add_turn(callback, turn_id, current_task)

    key = omni._scoring_window_key("u1", 4, 9, 2)
    state = json.loads(redis.data[key])
    assert state["turns"] == []
    assert state["queue"] == []
    callback._safe_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_lock_release_is_ownership_safe():
    redis = FakeRedis()
    key = omni._scoring_window_key("u1", 4, 9, 2)
    lock_key, owner = await omni._acquire_scoring_lock(redis, key)
    redis.data[lock_key] = "new-replica-owner"

    await omni._release_scoring_lock(redis, lock_key, owner)

    assert redis.data[lock_key] == "new-replica-owner"


@pytest.mark.asyncio
async def test_window_save_rejects_expired_or_replaced_lease():
    redis = FakeRedis()
    key = omni._scoring_window_key("u1", 4, 9, 2)
    lock_key, owner = await omni._acquire_scoring_lock(redis, key)
    redis.data[lock_key] = "replacement"

    with pytest.raises(omni._ScoringLockLost):
        await omni._save_scoring_window(
            redis, key, {"turns": []}, lock_key, owner
        )


@pytest.mark.asyncio
async def test_new_replica_queues_turn_while_evaluator_is_in_network_call():
    redis = FakeRedis()
    callback_a = callback_for()
    callback_b = callback_for()
    current_task = task()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def post(payload, _token):
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            await release.wait()
            return {"evaluation_status": "insufficient_evidence", "evidence_sufficient": False}
        return {
            "evaluation_status": "completed", "evidence_sufficient": True,
            "quality": "satisfactory", "delta": 2, "score": 2,
            "interaction_count": 4, "task_completed": False,
        }

    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", side_effect=post
    ):
        await add_turn(callback_a, "t1", current_task)
        await add_turn(callback_a, "t2", current_task)
        evaluator = asyncio.create_task(add_turn(callback_a, "t3", current_task))
        await entered.wait()
        # This simulates another replica: it can acquire the short Redis state
        # lease because the evaluator does not hold it across Workflow I/O.
        replacement = asyncio.create_task(add_turn(callback_b, "t4", current_task))
        await asyncio.sleep(0)
        key = omni._scoring_window_key("u1", 4, 9, 2)
        in_flight = json.loads(redis.data[key])
        assert [turn["turn_id"] for turn in in_flight["turns"]] == ["t1", "t2", "t3"]
        assert [turn["turn_id"] for turn in in_flight["queue"]] == ["t4"]
        release.set()
        result = await evaluator
        replacement_result = await replacement

    assert result["proficiency_delta"] == 2
    assert replacement_result["proficiency_delta"] == 2
    replacement_payload = callback_b._safe_send.await_args.args[0]["payload"]
    assert replacement_payload["turn_ids"] == ["t1", "t2", "t3", "t4"]
    assert [len(call["turn_window"]) for call in calls] == [3, 4]
    final = json.loads(redis.data[key])
    assert final["turns"] == []
    assert final["queue"] == []


@pytest.mark.asyncio
async def test_replacement_callback_replays_completed_inflight_window():
    redis = FakeRedis()
    callback_a = callback_for()
    callback_b = callback_for()
    current_task = task()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def post(_payload, _token):
        entered.set()
        await release.wait()
        return {
            "evaluation_status": "completed", "evidence_sufficient": True,
            "quality": "strong", "delta": 3, "score": 3,
            "interaction_count": 3, "completed_window_count": 1,
            "scoring_generation": 2, "task_completed": False,
        }

    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", side_effect=post
    ):
        await add_turn(callback_a, "t1", current_task)
        await add_turn(callback_a, "t2", current_task)
        evaluator = asyncio.create_task(add_turn(callback_a, "t3", current_task))
        await entered.wait()
        replacement = asyncio.create_task(add_turn(callback_b, "t4", current_task))
        await asyncio.sleep(0)
        release.set()
        await evaluator
        replayed = await replacement

    assert replayed["proficiency_delta"] == 3
    replay_payload = callback_b._safe_send.await_args.args[0]["payload"]
    assert replay_payload["evaluation_id"]
    assert replay_payload["turn_ids"] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_concurrent_duplicate_turns_are_persisted_once():
    redis = FakeRedis()
    callback_a = callback_for()
    callback_b = callback_for()
    current_task = task()
    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_post_scoring_window", AsyncMock()
    ):
        await asyncio.gather(
            add_turn(callback_a, "same-turn", current_task),
            add_turn(callback_b, "same-turn", current_task),
        )

    key = omni._scoring_window_key("u1", 4, 9, 2)
    state = json.loads(redis.data[key])
    assert [turn["turn_id"] for turn in state["turns"]] == ["same-turn"]
    assert state["seen_turn_ids"] == ["same-turn"]


@pytest.mark.asyncio
async def test_cas_loss_requeues_every_drained_inbox_turn():
    redis = FakeRedis()
    callback = callback_for()
    current_task = task()
    key = omni._scoring_window_key("u1", 4, 9, 2)
    for turn_id in ("overflow-1", "overflow-2"):
        await omni._persist_scoring_inbox(redis, key, {
            "turn_id": turn_id, "turn_order": 1,
            "user_content": "answer", "ai_response": "reply",
        })

    with patch.object(omni, "_get_redis_client", return_value=redis), patch.object(
        omni, "_save_scoring_window", side_effect=omni._ScoringLockLost(key)
    ):
        assert await add_turn(callback, "current", current_task) is None

    recovered = [json.loads(raw)["turn_id"] for raw in redis.lists[omni._scoring_inbox_key(key)]]
    assert recovered == ["overflow-1", "overflow-2", "current"]


@pytest.mark.asyncio
async def test_scene_progress_forwards_generation_and_ready_token():
    callback = callback_for()
    callback.phase_key = "u1:Cafe"
    callback.mode = None
    callback.is_daily_qa_mode = False
    callback.user_id = "u1"
    callback.conversation = None
    callback.websocket = None
    callback.user_context = {
        "native_language": "Chinese",
        "active_goal": {
            "id": 4,
            "target_language": "English",
            "current_task": task(),
        },
    }
    omni.session_phases[callback.phase_key] = {"phase": "scene_theater"}
    evaluated = {
        "task_ready_to_complete": True,
        "task_completed": False,
        "task_id": 9,
        "task_title": "Order coffee",
        "task_score": 9,
        "ready_token": "ready-token",
    }
    handle = AsyncMock(return_value=evaluated)
    with patch.object(omni, "_handle_turn_with_accumulator", handle):
        await omni._evaluate_scene_turn_progress(callback, 4, 9, "reply")

    current_task = handle.await_args.args[8]
    assert current_task["scoring_generation"] == 2
    ready_event = callback._safe_send.await_args.args[0]
    assert ready_event["payload"]["ready_token"] == "ready-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["recall", "daily_qa", "tour", "magic_repetition"])
async def test_non_scene_modes_never_enter_scoring(mode):
    callback = callback_for()
    callback.phase_key = "u1:Cafe"
    callback.mode = mode
    callback.is_daily_qa_mode = mode == "daily_qa"
    callback.user_id = "u1"
    callback.conversation = None
    callback.websocket = None
    callback.user_context = {
        "active_goal": {"id": 4, "current_task": {"id": 9}}
    }
    omni.session_phases[callback.phase_key] = {"phase": "scene_theater"}
    evaluate = AsyncMock()
    with patch.object(omni, "_handle_turn_with_accumulator", evaluate):
        await omni._evaluate_scene_turn_progress(callback, 4, 9, "reply")
    evaluate.assert_not_awaited()
