"""Smoke tests for server/state.py.

Run from the project root:
    python -m tests.test_state
"""

from pydantic import ValidationError

from server.state import (
    get_trip_state,
    reset_session,
    update_trip_state,
)


SESSION = "test-session"


def _fresh():
    reset_session(SESSION)


def test_initial_state_is_empty_skeleton():
    _fresh()
    state = get_trip_state(SESSION)
    assert state["preferences"]["difficulty"] is None
    assert state["preferences"]["areas"] == []
    assert state["status"] == "gathering"
    assert state["plan"]["days"] == []
    assert state["future_considerations"]["trip_purpose"] is None
    print("[ok] initial state is empty skeleton")


def test_partial_update_preserves_other_fields():
    _fresh()
    state = update_trip_state(SESSION, {
        "preferences": {
            "difficulty": "moderate",
            "areas": ["Far Eastern Fells"],
        },
    })
    assert state["preferences"]["difficulty"] == "moderate"
    assert state["preferences"]["areas"] == ["Far Eastern Fells"]
    assert state["preferences"]["distance_miles_max"] is None
    assert state["preferences"]["scenic_priorities"] == []
    assert state["status"] == "gathering"
    print("[ok] partial update preserves untouched fields")


def test_sequential_updates_compose():
    _fresh()
    update_trip_state(SESSION, {"preferences": {"difficulty": "easy"}})
    update_trip_state(SESSION, {"group": {"size": 2}})
    state = update_trip_state(SESSION, {"logistics": {"num_days": 3}})
    assert state["preferences"]["difficulty"] == "easy"
    assert state["group"]["size"] == 2
    assert state["logistics"]["num_days"] == 3
    print("[ok] sequential updates compose")


def test_list_updates_replace_not_append():
    _fresh()
    update_trip_state(SESSION, {"preferences": {"areas": ["Eastern Fells"]}})
    state = update_trip_state(SESSION, {
        "preferences": {"areas": ["Western Fells", "Northern Fells"]},
    })
    assert state["preferences"]["areas"] == ["Western Fells", "Northern Fells"]
    print("[ok] list updates replace rather than append")


def test_invalid_difficulty_rejected():
    _fresh()
    try:
        update_trip_state(SESSION, {"preferences": {"difficulty": "ultra"}})
    except ValidationError:
        print("[ok] invalid difficulty rejected")
        return
    raise AssertionError("expected ValidationError")


def test_unknown_top_level_key_rejected():
    _fresh()
    try:
        update_trip_state(SESSION, {"prefrences": {"difficulty": "easy"}})
    except ValidationError:
        print("[ok] unknown top-level key rejected")
        return
    raise AssertionError("expected ValidationError")


def test_planned_day_round_trip():
    _fresh()
    state = update_trip_state(SESSION, {
        "plan": {
            "days": [
                {
                    "day_number": 1,
                    "walk_id": 1143,
                    "accommodation_before": "node/302866833",
                    "notes": "Easy first day to ease in",
                }
            ]
        }
    })
    day = state["plan"]["days"][0]
    assert day["walk_id"] == 1143
    assert day["accommodation_before"] == "node/302866833"
    assert day["accommodation_after"] is None
    print("[ok] planned day round-trips through validation")


def test_sessions_are_isolated():
    reset_session("alice")
    reset_session("bob")
    update_trip_state("alice", {"preferences": {"difficulty": "easy"}})
    update_trip_state("bob", {"preferences": {"difficulty": "hard"}})
    assert get_trip_state("alice")["preferences"]["difficulty"] == "easy"
    assert get_trip_state("bob")["preferences"]["difficulty"] == "hard"
    print("[ok] sessions are isolated")


if __name__ == "__main__":
    test_initial_state_is_empty_skeleton()
    test_partial_update_preserves_other_fields()
    test_sequential_updates_compose()
    test_list_updates_replace_not_append()
    test_invalid_difficulty_rejected()
    test_unknown_top_level_key_rejected()
    test_planned_day_round_trip()
    test_sessions_are_isolated()
    print("\nAll state tests passed.")
