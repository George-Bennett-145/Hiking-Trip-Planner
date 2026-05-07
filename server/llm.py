"""OpenRouter client and agent loop.

Uses the `openai` SDK against OpenRouter's OpenAI-compatible endpoint.
Tool schemas are expressed in OpenAI Chat Completions format.

Environment variables (loaded from .env via python-dotenv):
    OPENROUTER_API_KEY  required
    MODEL               optional, defaults to a free model
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from server import state as state_module
from server.tools import (
    get_walk_details,
    search_accommodation,
    search_walks,
)


load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("MODEL", "meta-llama/llama-3.3-70b-instruct:free")
MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))

# Optional comma-separated fallback chain. When set, OpenRouter will try
# these in order if the primary model errors (rate-limited, unavailable, etc.).
# See https://openrouter.ai/docs/features/model-routing#models-array
MODEL_FALLBACKS = [
    m.strip() for m in os.environ.get("MODEL_FALLBACKS", "").split(",") if m.strip()
]

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

_client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")


# Per-session conversation history. Each entry is a list of OpenAI-format
# message dicts ready to pass back into run_agent. In-memory only — won't
# survive a process restart, same caveat as state._sessions.
_conversation_history: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    """Return the saved conversation history for a session (empty if none)."""
    return list(_conversation_history.get(session_id, []))


def set_history(session_id: str, history: list[dict]) -> None:
    """Replace the saved conversation history for a session."""
    _conversation_history[session_id] = list(history)


def clear_history(session_id: str) -> None:
    """Forget the conversation history for a session."""
    _conversation_history.pop(session_id, None)


# ============================================================ system prompt ===

SYSTEM_PROMPT = """\
You are a Lake District hiking trip planner. You help people choose a walk and
accommodation for a trip in England's Lake District by drawing on a dataset of
around 220 walks scraped from Walking Britain and around 870 accommodation
entries from OpenStreetMap. You cannot book anything, you do not have weather
or live trail conditions, and you only know about the Lake District. Speak
like an experienced fell-walker: concise, friendly, factual, never
marketing-flavoured.

# Hard rules

1. Never invent factual values. Distances, ascents, grades, coordinates,
   prices, mountain heights, estimated times: only state numbers that came
   back from a tool call. If a value is not in the data, say you do not know
   rather than guessing.
2. Use tools through the proper tool-call channel. Never write JSON-shaped
   text that looks like a tool call; either call the tool or reply in prose.
3. Do not use em dashes anywhere in your replies. Use commas, full stops, or
   colons instead.
4. After any turn in which you called `search_walks`, `search_accommodation`,
    or `update_trip_state` that changes the proposed walk, accommodation,
    or transport mode, you MUST call `show_on_map` before producing any
    assistant text. This is non-negotiable. It applies even when the person
    did not explicitly ask to see the map: a swap, a refinement, a
    difficulty change, a new accommodation, or a transport-mode change all
    qualify. Do not list options in text and "wait" for the person to pick
    — pick the single best fit and call `show_on_map` immediately. If you
    ran a search or updated the trip state, you are committing to an
    answer; surface it on the map first.
5. Remember rejections within this conversation. If the person dismisses a
   walk or an accommodation, do not propose it again unless they bring it
   back themselves. The same applies to category rejections: if they say "no
   camping", do not propose campsites afterwards.
5. Reproduce walk titles exactly as they appear in the data, including
   spacing. The dataset writes "Cat Bells", not "Catbells", and the trailhead
   phrase ("from Gutherscale", "via Piers Gill") is part of the title.

# Reply style

Two short paragraphs maximum. Add a third only if information genuinely
cannot fit otherwise. Default to plain conversational prose. Use **bold** for
walk and place names so they stand out. Do not use bullet lists, numbered
lists, or markdown headers in normal replies. Speak directly to the person
("you", "your"); never say "user". Avoid filler ("Great question!", "Of
course!", "I would be happy to help"); get to the substance. Ask one
question at a time, never three.

# Conversational flow

When the request is vague (e.g. "plan me a trip in the Lakes"), ask one
anchor question before proposing. Difficulty is the canonical anchor since
it must be asked rather than inferred, unless the person mentions an
accessibility need, in which case bias toward `easy` or `easy/mod`. The
exception is when the person invites you to pick ("surprise me", "just
choose something"); then propose with sensible defaults.

When the request is specific (a peak named, a difficulty stated, an area
mentioned), propose immediately. Iteration is normal. The person will
refine, change their mind, or swap things in and out, and you should
welcome that without apologising for redirecting your search.

# Tool order of operations

1. At the start of every turn, call `get_trip_state` to refresh what is
   already known.
2. Search for walks using the right parameter for what the person gave you:
   - Named peak or walk ("Scafell", "Helvellyn"): use `query`.
   - Named town or village ("near Keswick", "around Ambleside"): use
     `near_lat`, `near_lon`, `radius_km` from the coordinates table below.
     Do NOT use `areas` as a substitute for proximity. Area names group
     walks by fell grouping (e.g. "Northern Fells"), which does not
     correspond directly to towns; a town like Keswick sits on the edge of
     multiple fell areas. Results are sorted nearest-first when you use
     the proximity parameters.
   - General description ("moderate day walk", "something in the eastern
     fells"): use the structured filters — `difficulty`, `distance_miles_*`,
     `ascent_metres_max`, `areas`.
   - You can combine `query` or `areas` with proximity when both apply.
3. If a filtered search returns zero results, relax one or more filters and
   try again before reporting that nothing matches. State plainly which
   filters you relaxed.
4. Once one or two walks look promising, call `get_walk_details` for the
   full description.
5. Once a walk is settled on or proposed, call `search_accommodation`
   using the walk's `start_lat` and `start_lng` as the anchor, NOT the
   originally requested town. The person wants to sleep near the trailhead,
   not near the town they mentioned. Start with `radius_km=10`; if nothing
   suitable comes back, expand to `radius_km=20` and say so.
6. Update `trip_state` continuously as facts arrive: preferences as soon as
   stated, logistics and group as they come up, plan when a walk is decided.
   Do not batch all updates until the end of a turn.
7. Call `show_on_map` after every turn where the walk, accommodation, or
   transport mode changes (see hard rule 4). Pass the walk's integer
   `walk_id` and exactly one accommodation OSM-style id. Never pass more
   than one accommodation id — show the single best pick, not a list.

# Named-place coordinates

When the person names a Lake District town or village, use these coordinates
as the anchor for `search_walks` proximity parameters. A radius of 10 km
covers the immediately surrounding fells for most purposes; use 15-20 km to
include the wider area.

Keswick:            near_lat=54.600, near_lon=-3.134
Borrowdale:         near_lat=54.518, near_lon=-3.148
Buttermere:         near_lat=54.533, near_lon=-3.272
Ambleside:          near_lat=54.431, near_lon=-2.962
Grasmere:           near_lat=54.459, near_lon=-3.024
Langdale:           near_lat=54.444, near_lon=-3.075
Coniston:           near_lat=54.370, near_lon=-3.072
Hawkshead:          near_lat=54.371, near_lon=-2.992
Windermere/Bowness: near_lat=54.373, near_lon=-2.903
Patterdale:         near_lat=54.526, near_lon=-2.921
Glenridding:        near_lat=54.544, near_lon=-2.943
Ullswater:          near_lat=54.570, near_lon=-2.893
Thirlmere:          near_lat=54.537, near_lon=-3.063
Wasdale Head:       near_lat=54.452, near_lon=-3.298
Penrith:            near_lat=54.664, near_lon=-2.756

For places not in this list, use your knowledge of Lake District geography
to estimate approximate coordinates, or ask the person to clarify.

# Iteration patterns

Once a proposal is on the map, the person will refine it. Common refinements
and the tool sequence each one demands:

"Show me a different hotel for that walk."
  -> search_accommodation (same lat/lon as before, narrower types if implied)
  -> show_on_map (same walk_ids, new accommodation_ids)

"Show me something a bit harder / easier / shorter / longer."
  -> search_walks (same proximity, new difficulty/distance filters)
  -> get_walk_details (the chosen walk)
  -> search_accommodation (anchored on the NEW walk's start_lat/start_lng)
  -> show_on_map (new walk_ids, new accommodation_ids)

"Try Coniston instead." (area change)
  -> search_walks (proximity at the new town's coords from the table)
  -> get_walk_details
  -> search_accommodation (anchored on the new walk)
  -> show_on_map

"I'll walk from the hotel." / "I'll drive there."
  -> update_trip_state (logistics.transport_mode)
  -> show_on_map (same walk_ids and accommodation_ids)

In every case above, the turn ends with `show_on_map`. If you skip it, the
map will keep showing the old proposal while your reply talks about the
new one. That is the most common mistake — do not make it.

# Walk grades

Grades are awarded on an accumulative assessment of five criteria: terrain
(path quality, pathless sections, difficult ground), total height gain,
total distance, equipment needed, and navigation/compass skill required.
All grades assume reasonable weather; conditions can push any grade higher.

Easy: good paths, low level, minimal navigation.
Easy/Mod: mostly good paths with some rougher ground or modest ascent.
Moderate: mix of good and rougher terrain, moderate ascent and distance.
Mod/Hard: some pathless or difficult terrain, significant ascent or distance.
Hard: pathless sections, difficult terrain, strong navigation skills needed.
Very Hard: sustained difficult terrain, high ascent, full navigation required.
Severe: challenge routes, not standard walks.

Estimated time is calculated at 2 miles per hour base pace, plus 1 hour for
every 1,000 feet (305 m) of ascent. It excludes rest breaks. Use this formula
if the person asks how the time was arrived at, and note that it is a leisurely
pace so fitter walkers will typically finish faster.

# Walk proposals

Default to one well-matched walk. Offer two if there is a meaningfully
different alternative (a longer version, a quieter version, a different
fell area). Three only when the person has invited comparison ("show me a
few options"). Never propose more than three unprompted.

For each proposed walk, use the title verbatim from the data, then state
the distance in miles, the grade, and one short sentence saying why it
suits the request. Always pair the proposal with an accommodation
suggestion drawn from a real `search_accommodation` result, and always call
`show_on_map` so the person can see what you mean. End with a brief
invitation to refine, not a list of follow-up questions.

Example shape: "For a gentle introduction, **Cat Bells from Gutherscale**
is the obvious pick. 4.0 miles, easy/mod, with the classic Derwentwater
views. I have put a campsite a few minutes from the trailhead on the map
alongside it. Happy with that, or shall I try something quieter?"

# Accommodation reasoning

For tent camping, only propose entries where `tourism = "camp_site"`. The
data does not reliably tell us whether caravan sites accept tents, so do
not propose `caravan_site` for tent camping unless the person has
specifically asked about caravan sites.

For a pub stay, read the accommodation name. Names containing "Inn",
"Arms", "Tavern", or "Hotel & Pub" usually indicate a pub-with-rooms. A
bare "X Hotel" could be a pub-style hotel or a smarter traditional one;
when you cannot tell from the name, say so plainly and let the person
decide rather than pretending to know.

For a posh or country hotel, look for naming signals like "Country House",
"Hall", "Manor", or "Country Inn". For self-catering, prefer
`tourism = "apartment"`. For hostels, `tourism = "hostel"`. For
bed-and-breakfast style, `tourism = "guest_house"`.

Search within roughly 10 km of the trailhead first; if nothing acceptable
comes back, expand to about 30 km and tell the person why you expanded.

# Transport mode (very important, easy to misread)

`logistics.transport_mode` is `"walking"` or `"driving"`. It describes
**how the person gets from their accommodation to the trailhead**, not
what kind of hike they want. The whole walk itself is always on foot;
this field is only about the accommodation-to-trail link.

It also informs how far away accommodation can sensibly be: walkers want
something close to the trailhead, drivers can stay further afield.

Set it via `update_trip_state` with `{"logistics": {"transport_mode":
"walking"}}` (or `"driving"`).

CRITICAL: phrases like "walk from the hotel", "walk from my campsite",
"walk to the trail", "walk over to it", "I want to walk there" mean the
person plans to **travel on foot from their accommodation to the existing
trailhead**. They are NOT asking for a different walk. Do not search for
a new walk. Do not change the `walk_id`. Just call `update_trip_state` to
set `logistics.transport_mode = "walking"`, then re-call `show_on_map`
with the same walk and accommodation.

Phrases like "I'll drive there", "how do I drive to it", "drive to the
trailhead" mean `transport_mode = "driving"`. Same rule: keep the same
walk and accommodation, just update the mode.

Default heuristics if the person has not said:
- Accommodation within ~1.5 km of the trailhead, no driving mention:
  ask once whether they plan to walk or drive, then set the mode.
- Accommodation further than ~1.5 km from the trailhead and no walking
  mention: assume `driving` and set it without asking.
- Camping right at the trail start: assume `walking`.

When you propose a walk + accommodation pair for the first time, set
the transport mode at the same time based on the distance heuristic
above so the map link draws on the first render rather than waiting for
a follow-up question.

# State and memory

Always call `get_trip_state` at the start of a turn so you do not
contradict what was already established. Update `trip_state` as facts
arrive, not in one bulk write. Capture mentions of trip dates or trip
purpose in `future_considerations` so they are not lost, even though
those fields are not used elsewhere yet.

Manage the `status` field explicitly:
- `gathering` while you are still asking about preferences.
- `proposing` once you have made a specific walk plus accommodation
  suggestion.
- `confirmed` once the person has agreed to a plan.
- `iterating` if they accept part of a plan but want to swap something.

# Beginner safety

If the person has identified as a beginner and you are about to propose a
walk graded `hard`, `severe`, or `very hard`, add one short sentence
noting that the walk needs hill experience or proper navigation skills.
For `mod/hard`, use judgement based on ascent and distance. Do not insert
generic safety boilerplate ("remember to bring layers", "check the
weather") on every reply; mention safety only when genuinely warranted.

# Out of scope

- Walks outside the Lake District (Snowdonia, Scotland, Pennines, etc.):
  admit you only have Lake District walks and offer a Lake District
  alternative if one fits the spirit of the request.
- Weather, conditions, snow cover: you do not have this data. Suggest the
  Mountain Weather Information Service at mwis.org.uk.
- Booking accommodation or buying things: you cannot book on someone's
  behalf. Share the website or phone number from the listing instead.
- Saving or loading past trips: there is no persistence yet; the
  conversation lives only in memory.

# Things never to do

- Do not push Wainwright bagging unless the person has expressed interest.
- Do not volunteer GPX file downloads or raw coordinates; the map already
  handles those.
- Do not use bullet lists or numbered lists in conversational replies.
- Do not say "user".
- Do not pad replies with affirmations, restatements, or summary closings
  like "to summarise everything we have discussed".

The goal is a real trip the person will actually go on, not a generic
answer.
"""


# ================================================================ tool schemas

# Difficulty values mirror server/state.Difficulty exactly.
DIFFICULTY_VALUES = [
    "easy", "easy/mod", "moderate", "mod/hard", "hard", "severe", "very hard"
]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_walks",
            "description": (
                "Find Lake District walks. When the user names a specific "
                "peak or walk (e.g. 'Scafell', 'Helvellyn'), pass it as "
                "`query` — that does a substring match against title and "
                "Wainwrights. Use the structured filters (difficulty, "
                "distance, area, proximity) when the user is describing "
                "what kind of walk they want. You can combine query with "
                "filters, but if a filtered search returns nothing, try "
                "again with fewer filters before telling the user no walks "
                "match. Returns lightweight summaries — call "
                "get_walk_details for full descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Substring match against walk title and the "
                            "Wainwrights list (case-insensitive). Use when "
                            "the user names a peak or walk by name."
                        ),
                    },
                    "difficulty": {
                        "type": "array",
                        "items": {"type": "string", "enum": DIFFICULTY_VALUES},
                        "description": "Filter to walks at any of these grades.",
                    },
                    "distance_miles_min": {"type": "number"},
                    "distance_miles_max": {"type": "number"},
                    "ascent_metres_max": {"type": "integer"},
                    "areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Lake District fell areas, fuzzy-matched. "
                            "Canonical names: 'Far Eastern Fells', 'Eastern "
                            "Fells', 'Central Fells', 'Northern Fells', "
                            "'North Western Fells', 'Western Fells', "
                            "'Southern Fells', 'Outlying & Lesser Fells'."
                        ),
                    },
                    "near_lat": {"type": "number"},
                    "near_lon": {"type": "number"},
                    "radius_km": {"type": "number"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_walk_details",
            "description": (
                "Read the full row for a single walk, including its long "
                "description. Call this for the 1-2 walks you're actually "
                "considering after search_walks narrows the field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "walk_id": {"type": "integer"},
                },
                "required": ["walk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_accommodation",
            "description": (
                "Find accommodation within `radius_km` of (near_lat, near_lon). "
                "Always anchor on the chosen walk's start_lat/start_lng, not "
                "the town the user mentioned. Results are sorted by real routed "
                "distance (walking or driving network), not crow-flies, so "
                "detours around lakes and hills are accounted for. Each result "
                "includes distance_km (routed) and crow_flies_km for reference. "
                "Filter by tourism type with `types`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "near_lat": {"type": "number"},
                    "near_lon": {"type": "number"},
                    "radius_km": {"type": "number", "default": 10.0},
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "OSM tourism tags, e.g. ['hotel', 'guest_house', "
                            "'hostel', 'camp_site', 'caravan_site', 'apartment']."
                        ),
                    },
                    "limit": {"type": "integer", "default": 15},
                    "mode": {
                        "type": "string",
                        "enum": ["walking", "driving"],
                        "description": (
                            "Routing network to use for distance calculation. "
                            "Pass the transport_mode from trip state if known, "
                            "otherwise omit to use the default (walking)."
                        ),
                    },
                },
                "required": ["near_lat", "near_lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_trip_state",
            "description": (
                "Deep-merge fields into the structured trip state and return "
                "the new state. Nested dicts merge recursively; lists are "
                "REPLACED, not appended. To extend a list, read get_trip_state "
                "first, then pass the full new list. Pass only the fields you "
                "are changing; e.g. to set transport mode to walking, call "
                "with logistics={\"transport_mode\": \"walking\"}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preferences": {"type": "object"},
                    "group": {"type": "object"},
                    "logistics": {"type": "object"},
                    "plan": {"type": "object"},
                    "status": {
                        "type": "string",
                        "enum": ["gathering", "proposing", "confirmed", "iterating"],
                    },
                    "open_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "future_considerations": {"type": "object"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trip_state",
            "description": "Return the current structured trip state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_on_map",
            "description": (
                "Update the user's map with the walk route(s) and accommodation "
                "you are proposing. Pass empty lists to clear a layer. "
                "walk_ids may contain 1-3 integers when showing alternatives; "
                "accommodation_ids should contain exactly ONE id — the single "
                "best-fit place you are recommending. Calling this again "
                "replaces whatever was shown before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "walk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Walk IDs to draw as route lines.",
                    },
                    "accommodation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Exactly one OSM-style id for the recommended "
                            "accommodation, e.g. ['node/302866833']. "
                            "Do not pass multiple ids."
                        ),
                    },
                    "fit_bounds": {"type": "boolean", "default": True},
                },
            },
        },
    },
]


# =================================================== per-session tool handlers


def _build_tool_handlers(session_id: str) -> dict:
    """Build callables that the LLM-facing tool names dispatch to.

    Tools needing per-conversation context (state, map) get session_id
    injected via closure. The LLM never sees session_id in the tool schema.
    """

    def update_trip_state(**updates) -> dict:
        # Defensive: some models still wrap the payload under an 'updates' key
        # despite the flat schema. Unwrap if so.
        if list(updates.keys()) == ["updates"] and isinstance(updates["updates"], dict):
            updates = updates["updates"]
        return state_module.update_trip_state(session_id, updates)

    def get_trip_state() -> dict:
        return state_module.get_trip_state(session_id)

    def show_on_map(
        walk_ids: Optional[list[int]] = None,
        accommodation_ids: Optional[list[str]] = None,
        fit_bounds: bool = True,
    ) -> dict:
        # Writes into session state so the frontend picks it up via polling.
        # NOTE: polling works correctly only within a single process. For
        # multi-instance Cloud Run deployments, replace with SSE or WebSocket
        # backed by a shared store (Redis / Firestore).
        state_module.update_trip_state(session_id, {
            "map": {
                "walk_ids": list(walk_ids) if walk_ids is not None else [],
                "accommodation_ids": (
                    list(accommodation_ids) if accommodation_ids is not None else []
                ),
                "fit_bounds": fit_bounds,
            }
        })
        return {"status": "queued"}

    return {
        "search_walks": search_walks,
        "get_walk_details": get_walk_details,
        "search_accommodation": search_accommodation,
        "update_trip_state": update_trip_state,
        "get_trip_state": get_trip_state,
        "show_on_map": show_on_map,
    }


# ==================================================================== agent loop

MAX_AGENT_ITERATIONS = 10


def run_agent_stream(
    session_id: str,
    user_message: str,
    history: Optional[list[dict]] = None,
):
    """Stream events from one user-turn of the agent loop.

    Yields dicts of the following shapes (one per yield):
      {"type": "text",      "delta": str}       — incremental assistant text
      {"type": "tool_call", "name":  str}       — about to execute a tool
      {"type": "done",      "history": list,
                            "tool_calls_made": list[str]}
                                                — final event; caller should
                                                  persist `history` back into
                                                  the session store

    Stops yielding after `done`. Exceptions propagate to the caller.
    """
    history = list(history) if history else []
    handlers = _build_tool_handlers(session_id)

    # Has show_on_map been called in any earlier turn? If so, the user
    # already has something on the map; any subsequent search-driven
    # turn that doesn't call show_on_map is treated as a missed update.
    prior_map_setup = any(
        any(
            (tc.get("function") or {}).get("name") == "show_on_map"
            for tc in (msg.get("tool_calls") or [])
        )
        for msg in history
        if msg.get("role") == "assistant"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    tool_calls_made: list[str] = []
    nudged = False  # only nudge once per turn to avoid infinite loops
    # Capture last search results so the agent can deterministically fall
    # back to updating the map if the model fails to call `show_on_map`.
    last_search_walks: Optional[list[dict]] = None
    last_search_accommodation: Optional[list[dict]] = None

    request_kwargs: dict = {
        "model": MODEL,
        "tools": TOOL_SCHEMAS,
        "stream": True,
        "max_tokens": MAX_TOKENS,
    }
    if MODEL_FALLBACKS:
        request_kwargs["extra_body"] = {"models": [MODEL, *MODEL_FALLBACKS]}

    # Tracks last char of all text yielded so far across iterations so we can
    # insert a separating space when a new iteration's text would otherwise
    # butt directly against the previous iteration's last sentence.
    last_yielded_char = ""

    for _ in range(MAX_AGENT_ITERATIONS):
        request_kwargs["messages"] = messages
        stream = _client.chat.completions.create(**request_kwargs)

        accumulated_text = ""
        # Tool-call deltas are spread across many chunks. We accumulate them
        # by their `index` field, which OpenAI guarantees is stable per call.
        tool_calls_acc: dict[int, dict] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                chunk_text = delta.content
                # On the first text chunk of a continuation iteration, insert a
                # space if the previous iteration's text didn't end in whitespace
                # and this chunk doesn't already start with whitespace.
                if (
                    not accumulated_text
                    and last_yielded_char
                    and not last_yielded_char.isspace()
                    and not chunk_text[:1].isspace()
                ):
                    chunk_text = " " + chunk_text
                accumulated_text += chunk_text
                last_yielded_char = chunk_text[-1]
                yield {"type": "text", "delta": chunk_text}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "type": tc_delta.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

        if not tool_calls_acc:
            # No tool calls — about to produce the final reply.
            # Safety net for hard rule 4: if the model researched (called a
            # search tool this turn) but did NOT call show_on_map, and the
            # map was previously set up, the user is mid-iteration and the
            # map will go stale. Prefer a deterministic fallback: automatically
            # call `show_on_map` using the most recent search results where
            # possible. If that is not feasible, nudge the model once.
            researched_this_turn = (
                "search_walks" in tool_calls_made
                or "search_accommodation" in tool_calls_made
            )
            shown_this_turn = "show_on_map" in tool_calls_made
            if (
                prior_map_setup
                and researched_this_turn
                and not shown_this_turn
                and not nudged
            ):
                # Attempt deterministic auto-show_on_map when possible.
                auto_called = False
                try:
                    # Prefer the freshest search results; fall back to state.
                    walk_ids = []
                    accom_ids = []
                    if last_search_walks:
                        walk_ids = [w.get("walk_id") for w in last_search_walks[:3] if w.get("walk_id")]
                    else:
                        # try to read from trip state plan
                        try:
                            cur_state = state_module.get_trip_state(session_id)
                            days = cur_state.get("plan", {}).get("days", [])
                            if days:
                                wid = days[0].get("walk_id")
                                if wid:
                                    walk_ids = [wid]
                        except Exception:
                            pass

                    if last_search_accommodation:
                        first = last_search_accommodation[0]
                        if first and first.get("id"):
                            accom_ids = [first.get("id")]
                    else:
                        try:
                            cur_state = state_module.get_trip_state(session_id)
                            map_accom = cur_state.get("map", {}).get("accommodation_ids", [])
                            if map_accom:
                                accom_ids = [map_accom[0]]
                        except Exception:
                            pass

                    if walk_ids or accom_ids:
                        # Execute the show_on_map tool deterministically and
                        # inject its result into the conversation so the
                        # frontend will update even if the model omitted it.
                        tool_calls_made.append("show_on_map")
                        yield {"type": "tool_call", "name": "show_on_map"}
                        print(f"[auto-show] invoking show_on_map with walk_ids={walk_ids} accom_ids={accom_ids}", flush=True)
                        args_json = json.dumps({
                            "walk_ids": walk_ids,
                            "accommodation_ids": accom_ids,
                        })
                        _res = _execute_tool(handlers, "show_on_map", args_json)
                        # Strict providers (Anthropic via Bedrock) reject a
                        # tool_result whose tool_use_id has no matching tool_use
                        # in the prior assistant turn, so synthesize that turn
                        # before appending the result.
                        messages.append({
                            "role": "assistant",
                            "content": accumulated_text or None,
                            "tool_calls": [{
                                "id": "auto-show",
                                "type": "function",
                                "function": {
                                    "name": "show_on_map",
                                    "arguments": args_json,
                                },
                            }],
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": "auto-show",
                            "content": json.dumps(_res, default=str),
                        })
                        auto_called = True

                except Exception:
                    auto_called = False

                if auto_called:
                    # Mark that we performed the fallback and loop the agent
                    # once so it can continue from the new map state.
                    nudged = True
                    continue

                # Fall back to nudging the model to call show_on_map itself.
                messages.append({"role": "assistant", "content": accumulated_text or None})
                messages.append({
                    "role": "system",
                    "content": (
                        "REMINDER: hard rule 4. You ran a search this turn but "
                        "did not call show_on_map. The map is now out of sync "
                        "with what you are about to say. Pick the single best "
                        "walk and the single best accommodation from your "
                        "recent search results and call show_on_map now. "
                        "Do not reply with text again until you have done so."
                    ),
                })
                nudged = True
                continue

            messages.append({"role": "assistant", "content": accumulated_text})
            yield {
                "type": "done",
                "history": messages[1:],
                "tool_calls_made": tool_calls_made,
            }
            return

        # Tool calls present (text content may also be present alongside).
        ordered_tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        messages.append({
            "role": "assistant",
            "content": accumulated_text or None,
            "tool_calls": ordered_tool_calls,
        })

        for tc in ordered_tool_calls:
            name = tc["function"]["name"]
            tool_calls_made.append(name)
            yield {"type": "tool_call", "name": name}
            result = _execute_tool(handlers, name, tc["function"]["arguments"])
            # Capture recent search results for deterministic fallback.
            try:
                if name == "search_walks" and isinstance(result, list):
                    last_search_walks = result
                if name == "search_accommodation" and isinstance(result, list):
                    last_search_accommodation = result
            except Exception:
                pass
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })

    # Hit the iteration cap without producing a final reply.
    fallback = "(agent loop hit max iterations without finishing)"
    messages.append({"role": "assistant", "content": fallback})
    yield {"type": "text", "delta": fallback}
    yield {
        "type": "done",
        "history": messages[1:],
        "tool_calls_made": tool_calls_made,
    }


def run_agent(
    session_id: str,
    user_message: str,
    history: Optional[list[dict]] = None,
) -> dict:
    """Synchronous wrapper around `run_agent_stream` for the CLI and tests.

    Collects the streamed events into the legacy {reply, history,
    tool_calls_made} dict the CLI expects.
    """
    reply_parts: list[str] = []
    final_event: Optional[dict] = None
    for event in run_agent_stream(session_id, user_message, history):
        if event["type"] == "text":
            reply_parts.append(event["delta"])
        elif event["type"] == "done":
            final_event = event

    if final_event is None:
        # Should not happen: stream always ends with a `done` event.
        return {"reply": "".join(reply_parts), "history": [], "tool_calls_made": []}

    return {
        "reply": "".join(reply_parts),
        "history": final_event["history"],
        "tool_calls_made": final_event["tool_calls_made"],
    }


def _execute_tool(handlers: dict, name: str, raw_args: Optional[str]):
    """Dispatch a tool call. Always returns a JSON-serialisable dict."""
    print(f"[tool] {name}({raw_args})", flush=True)
    if name not in handlers:
        result = {"error": f"unknown tool: {name!r}"}
        print(f"[tool] -> {result}", flush=True)
        return result

    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        result = {"error": f"could not parse tool arguments as JSON: {e}"}
        print(f"[tool] -> {result}", flush=True)
        return result

    if not isinstance(args, dict):
        result = {"error": f"tool arguments must be a JSON object, got {type(args).__name__}"}
        print(f"[tool] -> {result}", flush=True)
        return result

    try:
        result = handlers[name](**args)
    except TypeError as e:
        result = {"error": f"bad arguments to {name}: {e}"}
    except Exception as e:
        result = {"error": f"{e.__class__.__name__}: {e}"}

    # Truncate large successful results so the log stays readable.
    if isinstance(result, dict) and "error" in result:
        print(f"[tool] -> ERROR: {result}", flush=True)
    else:
        summary = str(result)
        if len(summary) > 300:
            summary = summary[:300] + "...(truncated)"
        print(f"[tool] -> {summary}", flush=True)
    return result
