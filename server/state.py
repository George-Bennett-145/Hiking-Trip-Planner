"""Trip state model and per-session in-memory store.

The trip state is the structured representation of a user's planning
context — preferences, group, logistics, and the developing plan.
The LLM updates it via update_trip_state and reads it via get_trip_state.

State is held in memory keyed by session ID. This will not survive a
process restart and does not work across multiple worker processes —
acceptable for the v1 capstone scope. A persistent store (Redis, SQLite,
etc.) can replace this without changing the public function signatures.

Tool wrappers in server/tools.py will inject the session_id from the
incoming request so the LLM never sees or specifies it directly.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# Allowed values mirror the values present in walks.csv and the brief's
# "Important behavioural notes". Keeping them as Literal types means a
# typo by the LLM (e.g. "very_hard" instead of "very hard") fails
# validation rather than silently corrupting the state.
Difficulty = Literal[
    "easy", "easy/mod", "moderate", "mod/hard", "hard", "severe", "very hard"
]
ExperienceLevel = Literal["beginner", "intermediate", "experienced"]
TravelStyle = Literal["single_base", "moving"]
TransportMode = Literal["driving", "walking"]
TripStatus = Literal["gathering", "proposing", "confirmed", "iterating"]


class _Strict(BaseModel):
    """Base for every state model. extra='forbid' surfaces LLM key typos."""

    model_config = ConfigDict(extra="forbid")


class Preferences(_Strict):
    difficulty: Optional[Difficulty] = None
    distance_miles_min: Optional[float] = None
    distance_miles_max: Optional[float] = None
    ascent_metres_max: Optional[int] = None
    areas: list[str] = Field(default_factory=list)
    avoid_busy: Optional[bool] = None
    scenic_priorities: list[str] = Field(default_factory=list)
    experience_level: Optional[ExperienceLevel] = None
    accessibility_needs: Optional[str] = None


class Group(_Strict):
    size: Optional[int] = None
    composition: Optional[str] = None


class Logistics(_Strict):
    num_days: Optional[int] = None
    base_location: Optional[str] = None
    max_drive_to_trailhead_minutes: Optional[int] = None
    travel_style: Optional[TravelStyle] = None
    transport_mode: Optional[TransportMode] = None
    accommodation_types: list[str] = Field(default_factory=list)
    accommodation_must_haves: list[str] = Field(default_factory=list)
    budget_per_night: Optional[float] = None


class PlannedDay(_Strict):
    day_number: int
    walk_id: Optional[int] = None
    # Accommodation IDs are OSM-style strings: "node/302866833" or "way/123".
    accommodation_before: Optional[str] = None
    accommodation_after: Optional[str] = None
    notes: Optional[str] = None


class Plan(_Strict):
    days: list[PlannedDay] = Field(default_factory=list)


class FutureConsiderations(_Strict):
    trip_purpose: Optional[str] = None
    trip_dates: Optional[str] = None


class MapState(_Strict):
    """Pending map update written by the show_on_map tool.

    The frontend polls /api/sessions/{id}/state and re-renders this layer
    whenever walk_ids or accommodation_ids change.

    NOTE: This uses in-process polling. If the app scales to multiple
    Cloud Run instances, this field must move to a shared store (Redis /
    Firestore) or be replaced with SSE/WebSocket.
    """

    walk_ids: list[int] = Field(default_factory=list)
    accommodation_ids: list[str] = Field(default_factory=list)
    fit_bounds: bool = True


class TripState(_Strict):
    preferences: Preferences = Field(default_factory=Preferences)
    group: Group = Field(default_factory=Group)
    logistics: Logistics = Field(default_factory=Logistics)
    plan: Plan = Field(default_factory=Plan)
    status: TripStatus = "gathering"
    open_questions: list[str] = Field(default_factory=list)
    future_considerations: FutureConsiderations = Field(
        default_factory=FutureConsiderations
    )
    map: MapState = Field(default_factory=MapState)


# ----------------------------------------------------------------- session store

_sessions: dict[str, TripState] = {}


def _get_or_create(session_id: str) -> TripState:
    if session_id not in _sessions:
        _sessions[session_id] = TripState()
    return _sessions[session_id]


def get_trip_state(session_id: str) -> dict:
    """Return the session's trip state as a plain dict."""
    return _get_or_create(session_id).model_dump()


def update_trip_state(session_id: str, updates: dict) -> dict:
    """Deep-merge `updates` into the state; return the new full state.

    Merge semantics:
      * Nested dicts are merged recursively, so updating one field of
        `preferences` leaves the rest untouched.
      * Lists are replaced wholesale (not appended). To extend a list,
        the LLM must read the current state, build the full new list,
        and pass that.

    The merged result is re-validated against TripState. Unknown keys,
    bad enum values, and wrong types raise pydantic.ValidationError.
    """
    state = _get_or_create(session_id)
    merged = _deep_merge(state.model_dump(), updates)
    new_state = TripState.model_validate(merged)
    _sessions[session_id] = new_state
    return new_state.model_dump()


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def _deep_merge(base: dict, updates: dict) -> dict:
    result = dict(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
