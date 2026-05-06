"""Smoke tests for server/llm.py — no API calls (those cost tokens).

Verifies:
  - Module imports cleanly (env vars present, OpenAI client constructs)
  - Tool schemas are well-formed
  - Per-session tool handlers can be built and the non-LLM ones execute correctly
  - The handler closure correctly scopes session_id

Run from project root:
    python -m tests.test_llm_imports
"""

import json

from server import state as state_module
from server.llm import MAX_AGENT_ITERATIONS, MODEL, TOOL_SCHEMAS, _build_tool_handlers


def test_module_imports_cleanly():
    assert MODEL, "MODEL env var must resolve to something"
    assert MAX_AGENT_ITERATIONS > 0
    print(f"[ok] llm module loads (MODEL={MODEL}, max_iter={MAX_AGENT_ITERATIONS})")


def test_tool_schemas_are_well_formed():
    expected_tools = {
        "search_walks", "get_walk_details", "search_accommodation",
        "update_trip_state", "get_trip_state", "show_on_map",
    }
    actual = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert actual == expected_tools, f"unexpected tools: {actual ^ expected_tools}"
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]
        # Schema must be JSON-round-trippable
        json.loads(json.dumps(schema))
    print(f"[ok] all {len(TOOL_SCHEMAS)} tool schemas are well-formed")


def test_handlers_for_data_tools_work():
    handlers = _build_tool_handlers("smoke-test")
    walks = handlers["search_walks"](difficulty=["easy"], limit=2)
    assert isinstance(walks, list) and len(walks) <= 2
    if walks:
        details = handlers["get_walk_details"](walk_id=walks[0]["walk_id"])
        assert details["walk_id"] == walks[0]["walk_id"]
    print("[ok] data-tool handlers callable directly")


def test_handlers_inject_session_id_into_state_tools():
    state_module.reset_session("session-a")
    state_module.reset_session("session-b")
    handlers_a = _build_tool_handlers("session-a")
    handlers_b = _build_tool_handlers("session-b")

    handlers_a["update_trip_state"](
        updates={"preferences": {"difficulty": "easy"}},
    )
    handlers_b["update_trip_state"](
        updates={"preferences": {"difficulty": "hard"}},
    )

    state_a = handlers_a["get_trip_state"]()
    state_b = handlers_b["get_trip_state"]()
    assert state_a["preferences"]["difficulty"] == "easy"
    assert state_b["preferences"]["difficulty"] == "hard"
    print("[ok] session_id injected correctly via handler closure")


def test_show_on_map_writes_to_state():
    state_module.reset_session("map-test-session")
    handlers = _build_tool_handlers("map-test-session")
    result = handlers["show_on_map"](walk_ids=[1143, 1109], accommodation_ids=[])
    assert result["status"] == "queued"
    state = state_module.get_trip_state("map-test-session")
    assert state["map"]["walk_ids"] == [1143, 1109]
    assert state["map"]["accommodation_ids"] == []
    assert state["map"]["fit_bounds"] is True
    print("[ok] show_on_map writes walk_ids and accommodation_ids to session state")


if __name__ == "__main__":
    test_module_imports_cleanly()
    test_tool_schemas_are_well_formed()
    test_handlers_for_data_tools_work()
    test_handlers_inject_session_id_into_state_tools()
    test_show_on_map_writes_to_state()
    print("\nAll llm import/handler tests passed.")
