"""Smoke tests for FastAPI routes — no real LLM calls.

The `/api/chat` route is exercised manually via curl/Postman because
each call hits OpenRouter and costs tokens. Here we verify route
plumbing, Pydantic validation, and the inspection/reset endpoints.

Run from the project root:
    python -m tests.test_api_routes
"""

from fastapi.testclient import TestClient

from server import llm as llm_module
from server import state as state_module
from server.main import app


client = TestClient(app)


def test_healthz_returns_ok():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("[ok] /healthz returns 200 OK")


def test_chat_rejects_empty_message():
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text}"
    print("[ok] /api/chat rejects empty message with 422")


def test_chat_rejects_missing_message():
    r = client.post("/api/chat", json={})
    assert r.status_code == 422
    print("[ok] /api/chat rejects missing message with 422")


def test_session_state_returns_default_skeleton_for_unknown_session():
    state_module.reset_session("smoke-fresh")
    r = client.get("/api/sessions/smoke-fresh/state")
    assert r.status_code == 200
    state = r.json()
    assert state["status"] == "gathering"
    assert state["preferences"]["difficulty"] is None
    assert state["plan"]["days"] == []
    print("[ok] /api/sessions/{id}/state returns default skeleton for unknown id")


def test_session_state_reflects_prior_updates():
    sid = "smoke-prior"
    state_module.reset_session(sid)
    state_module.update_trip_state(sid, {"preferences": {"difficulty": "moderate"}})

    r = client.get(f"/api/sessions/{sid}/state")
    assert r.status_code == 200
    assert r.json()["preferences"]["difficulty"] == "moderate"
    print("[ok] /api/sessions/{id}/state reflects updates made by tools")


def test_reset_clears_state_and_history():
    sid = "smoke-reset"
    state_module.update_trip_state(sid, {"preferences": {"difficulty": "easy"}})
    llm_module.set_history(sid, [{"role": "user", "content": "hi"}])

    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json() == {"status": "reset", "session_id": sid}

    assert state_module.get_trip_state(sid)["preferences"]["difficulty"] is None
    assert llm_module.get_history(sid) == []
    print("[ok] DELETE /api/sessions/{id} clears both state and history")


def test_chat_route_is_registered():
    # Confirms the route exists; doesn't actually invoke the LLM.
    routes = {(r.path, tuple(r.methods or ())) for r in app.routes if hasattr(r, "methods")}
    assert any(p == "/api/chat" and "POST" in m for p, m in routes)
    print("[ok] POST /api/chat is registered on the app")


if __name__ == "__main__":
    test_healthz_returns_ok()
    test_chat_rejects_empty_message()
    test_chat_rejects_missing_message()
    test_session_state_returns_default_skeleton_for_unknown_session()
    test_session_state_reflects_prior_updates()
    test_reset_clears_state_and_history()
    test_chat_route_is_registered()
    print("\nAll API route tests passed.")
