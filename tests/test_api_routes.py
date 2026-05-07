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


# -------- index + data routes --------

def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Lake District Trip Planner" in r.text
    print("[ok] / serves index.html with expected title")


def test_walks_route_returns_only_gpx_available():
    r = client.get("/api/walks")
    assert r.status_code == 200
    walks = r.json()
    assert walks, "expected at least one walk"
    # The dropdown only ever wanted walks with GPX, so the route filters server-side.
    assert all(w["gpx_available"] for w in walks)
    print(f"[ok] /api/walks returns {len(walks)} walks, all gpx_available")


def test_walks_route_shape():
    r = client.get("/api/walks")
    assert r.status_code == 200
    sample = r.json()[0]
    expected_keys = {
        "walk_id", "title", "area", "grade", "distance_miles",
        "ascent_metres", "start_lat", "start_lng", "gpx_available",
    }
    assert expected_keys.issubset(sample.keys()), \
        f"missing keys: {expected_keys - set(sample.keys())}"
    print("[ok] /api/walks entries expose all expected keys")


def test_accommodation_route_shape():
    r = client.get("/api/accommodation")
    assert r.status_code == 200
    accom = r.json()
    assert accom, "expected at least one accommodation entry"
    sample = accom[0]
    assert "osm_id" in sample
    assert "lat" in sample and "lon" in sample
    assert "tourism" in sample
    print(f"[ok] /api/accommodation returns {len(accom)} entries with expected shape")


# -------- GPX route --------

def test_gpx_route_returns_geojson_linestring():
    # walk 2036 (Cat Bells) is one of the canonical test walks.
    r = client.get("/api/walks/2036/gpx")
    assert r.status_code == 200
    feature = r.json()
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "LineString"
    coords = feature["geometry"]["coordinates"]
    assert len(coords) > 50, f"expected a substantial track, got {len(coords)} points"
    # GeoJSON convention: [lng, lat] — Lake District lng is negative.
    lng, lat = coords[0]
    assert -3.5 < lng < -2.5, f"longitude {lng} not in Lake District range"
    assert 54.3 < lat < 54.8, f"latitude {lat} not in Lake District range"
    print(f"[ok] /api/walks/2036/gpx returns LineString with {len(coords)} [lng,lat] points")


def test_gpx_route_404_for_unknown_walk():
    r = client.get("/api/walks/99999/gpx")
    assert r.status_code == 404
    assert "GPX not found" in r.json()["detail"]
    print("[ok] /api/walks/{id}/gpx returns 404 for unknown walk_id")


if __name__ == "__main__":
    test_healthz_returns_ok()
    test_chat_rejects_empty_message()
    test_chat_rejects_missing_message()
    test_session_state_returns_default_skeleton_for_unknown_session()
    test_session_state_reflects_prior_updates()
    test_reset_clears_state_and_history()
    test_chat_route_is_registered()
    test_index_serves_html()
    test_walks_route_returns_only_gpx_available()
    test_walks_route_shape()
    test_accommodation_route_shape()
    test_gpx_route_returns_geojson_linestring()
    test_gpx_route_404_for_unknown_walk()
    print("\nAll API route tests passed.")
