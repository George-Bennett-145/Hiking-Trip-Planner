"""FastAPI application entry point.

Run locally with:
    uvicorn server.main:app --reload --port 5001

Routes:
    GET    /                                 map UI (index.html)
    GET    /healthz                          liveness probe
    POST   /api/chat                         send a user message, get a reply
    GET    /api/sessions/{session_id}/state  inspect current trip state
    DELETE /api/sessions/{session_id}        reset state + conversation history
    GET    /api/walks                        list walks (gpx_available only)
    GET    /api/accommodation                list all accommodation
    GET    /api/walks/{walk_id}/gpx          walk route as GeoJSON LineString

Per-session state and conversation history live in memory and are keyed
by `session_id`. If the client doesn't supply one on the first chat turn,
a fresh UUID is generated and returned in the response — the client should
keep using it on subsequent turns.
"""

import json
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from server import data as data_module
from server import gpx as gpx_module
from server import llm as llm_module
from server import routing as routing_module
from server import state as state_module

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pre-load the routing graphs at startup so accommodation distance
    # calculations don't pay the ~8s walking-graph load cost on the first
    # tool call. Missing graph files are logged and skipped so the rest of
    # the app still starts.
    print("[startup] warming routing graphs...")
    t0 = time.perf_counter()
    try:
        routing_module.prewarm()
        print(f"[startup] routing graphs ready in {time.perf_counter() - t0:.1f}s")
    except FileNotFoundError as e:
        print(f"[startup] graph warm skipped: {e}")
    except Exception as e:
        print(f"[startup] graph warm failed: {e}")
    yield


app = FastAPI(title="Lake District Trip Planner", lifespan=lifespan)


# ----------------------------------------------------------- request/response


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User's message to the bot.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Stable per-conversation identifier. Omit on the first turn — "
            "the server will generate one and return it for you to reuse."
        ),
    )


class ResetResponse(BaseModel):
    status: str
    session_id: str


# --------------------------------------------------------------------- routes


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _sse_event(payload: dict) -> str:
    """Encode a single Server-Sent Event line (data-only, no event/id/retry)."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Stream the assistant's response as Server-Sent Events.

    The frontend reads:
      session   {session_id}                      — first event, before any text
      tool_call {name}                            — bot is about to run a tool
      text      {delta}                           — incremental assistant text
      done      {}                                — final marker
      error     {detail}                          — something went wrong
    """
    session_id = req.session_id or str(uuid4())
    history = llm_module.get_history(session_id)

    def event_stream():
        # Send the session id up-front so the frontend can pin it for the
        # map-poll loop even before any text arrives.
        yield _sse_event({"type": "session", "session_id": session_id})

        final_history = None
        try:
            for event in llm_module.run_agent_stream(session_id, req.message, history):
                if event["type"] == "done":
                    final_history = event["history"]
                    yield _sse_event({"type": "done"})
                else:
                    yield _sse_event(event)
        except Exception as e:
            print(f"\n[chat] agent loop failed for session {session_id}:")
            traceback.print_exc()
            yield _sse_event({
                "type": "error",
                "detail": f"{e.__class__.__name__}: {e}",
            })
            return

        # Persist the conversation only on a clean completion.
        if final_history is not None:
            llm_module.set_history(session_id, final_history)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable proxy buffering (e.g. nginx) so chunks reach the browser.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}/state")
def get_session_state(session_id: str):
    return state_module.get_trip_state(session_id)


@app.delete("/api/sessions/{session_id}", response_model=ResetResponse)
def reset_session(session_id: str):
    state_module.reset_session(session_id)
    llm_module.clear_history(session_id)
    return ResetResponse(status="reset", session_id=session_id)


# ------------------------------------------------------------------- static / data routes


@app.get("/")
def index():
    return FileResponse(_TEMPLATES_DIR / "index.html")


@app.get("/api/walks")
def get_walks():
    """Return walks that have a GPX file available (used by the map dropdown)."""
    return [w for w in data_module.load_walks() if w["gpx_available"]]


@app.get("/api/accommodation")
def get_accommodation():
    return data_module.load_accommodation()


@app.get("/api/walks/{walk_id}/gpx")
def get_gpx(walk_id: int):
    """Return a walk's GPX track as a GeoJSON LineString for Leaflet."""
    try:
        points = gpx_module.load_gpx_points(walk_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"GPX not found for walk {walk_id}")

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in points],
        },
    }
