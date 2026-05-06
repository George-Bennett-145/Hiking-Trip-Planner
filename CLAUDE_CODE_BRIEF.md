# Lake District Trip Planner — Chatbot Scaffolding Brief

## Context

I'm building a hiking and camping trip planner for the Lake District as a capstone project. The goal is for users to chat with an LLM that helps them plan a trip — picking walks and accommodation that suit them — with everything visualised on a map.

### What already exists in this repo

- `output/walks.csv` — ~200 Lake District walks scraped from Walking Britain. Columns: `walk_id, title, area, wainwrights, county, author, grade, distance_miles, distance_km, ascent_feet, ascent_metres, estimated_time, start_lat, start_lng, start_postcode, description, gpx_available`
- `output/accommodation.json` — accommodation data from the OSM Overpass API. Each entry has: `id, type, tourism, name, lat, lon, amenity, brand, brand_wikidata, fhrs_id, addr_housename, addr_street, addr_city, addr_postcode, addr_country, phone, website`
- `output/gpx/` — GPX files named `{walk_id}.gpx`
- `app.py` — Flask backend with endpoints `/`, `/api/accommodation`, `/api/walks`, `/api/walks/<id>/gpx`
- `templates/index.html` — Leaflet map frontend that loads accommodation as colour-coded markers and lets the user pick a walk from a dropdown to plot its GPX track
- `fetch_accommodation.py`, `scraper.py` — data-collection scripts (already done, don't need changes)

### What we're building now

A chatbot that:

- Takes user input via a chat panel **added to the existing `index.html`** (not a separate page)
- Uses an LLM with tool-calling to help plan a trip
- Maintains a structured "trip state" object during the conversation
- Drives the map (causes walks/accommodation to appear, zoom, etc.) as the conversation progresses
- Asks follow-up questions when the user is vague, but is happy to propose ideas when the user wants inspiration

The chatbot will be the user's main interface — the map is supporting visualisation.

---

## Decisions already made

- **Architecture:** Single LLM agent with tools (not multi-agent). Tools are simple data fetches — no need for sub-agents.
- **State model:** Hybrid — a structured "trip state" object that the LLM updates via tools as the conversation progresses. Conversation can be free-flowing but state is always trackable.
- **No embeddings in v1.** See "Decisions to discuss" below for the reasoning and the option to add later.
- **Fuzzy name matching** with `rapidfuzz` for place/Wainwright names so misspellings don't break things.
- **LLM SDK:** Anthropic SDK for now. Easy to swap to OpenAI later if needed.
- **Frontend:** Add a chat panel to existing `index.html`. Keep the existing map functionality working.

---

## Decisions to discuss with me before scaffolding

### 1. Backend framework: Flask vs FastAPI

The existing `app.py` is Flask. I'm considering migrating to FastAPI because:

- Chatbots benefit from streaming responses (text appearing word-by-word) which FastAPI handles natively
- LLM calls are slow I/O — async means the server handles concurrent users without blocking
- Pydantic gives typed, validated request/response bodies which is useful for structured tool I/O

But Flask works too and the existing code is already written. **Please ask me which way I want to go before scaffolding.** Give me your honest recommendation based on what you see in the existing code.

### 2. Embeddings: now or later?

The walk descriptions are rich Wainwright-style prose. A user query like "a walk with a real sense of solitude" can't be answered by structured filters alone — it needs semantic search.

- **Option A (recommended):** Skip embeddings for v1. The LLM can use `search_walks` with broad filters, then read full descriptions for promising candidates via `get_walk_details`. Add a `search_walks_by_description(query)` tool later if needed (~1 hour of work).
- **Option B:** Build embeddings in from the start. ~2-3 hours extra scaffolding work upfront.

**Please ask me which I'd prefer.**

---

## Tool signatures

These are the tools the LLM will have access to. Implement these as Python functions that Claude Code's agent loop calls; expose them to the LLM via the Anthropic SDK's tool-use API.

```python
search_walks(
    difficulty: list[str] = None,
    # valid values: "easy", "easy/mod", "moderate", "mod/hard",
    #               "hard", "severe", "very hard"
    distance_miles_min: float = None,
    distance_miles_max: float = None,
    ascent_metres_max: int = None,
    areas: list[str] = None,
    wainwrights: list[str] = None,  # fuzzy matched against the wainwrights column
    near_lat: float = None,
    near_lon: float = None,
    radius_km: float = None,
    limit: int = 10,
) -> list[dict]
# Returns lightweight summaries:
# [{walk_id, title, area, grade, distance_miles, ascent_metres,
#   estimated_time, start_lat, start_lng, wainwrights, short_description}]
# short_description = first 400 chars of full description

get_walk_details(walk_id: int) -> dict
# Returns the full row including the long description.
# LLM uses this after search_walks for the 1-2 walks it's actually considering.

search_accommodation(
    near_lat: float,           # required — accommodation only makes sense relative to a location
    near_lon: float,
    radius_km: float = 10,
    types: list[str] = None,   # e.g. ["hotel", "guest_house", "camp_site"]
    limit: int = 15,
) -> list[dict]
# Returns: [{id, name, tourism, lat, lon, distance_km, addr_street,
#            addr_postcode, phone, website}]
# distance_km calculated server-side using haversine formula

update_trip_state(updates: dict) -> dict
# Merges updates into the trip state. Returns the new full state.

get_trip_state() -> dict
# Returns current state. Helps the LLM remember what's been gathered.

show_on_map(
    walk_ids: list[int] = None,           # empty list = clear walks
    accommodation_ids: list[str] = None,  # empty list = clear accommodation
    fit_bounds: bool = True,
) -> dict
# Pushes update to the frontend map. Returns confirmation.
# Implementation likely uses server-sent events or websockets — discuss with me.
```

---

## Trip state object

This is the data structure the LLM updates during conversation:

```python
{
  "preferences": {
    "difficulty": None,
    # valid values: "easy" | "easy/mod" | "moderate" | "mod/hard" |
    #               "hard" | "severe" | "very hard"
    # Bot must ASK rather than infer, unless accessibility need is mentioned.
    "distance_miles_min": None,
    "distance_miles_max": None,
    "ascent_metres_max": None,
    "areas": [],                  # e.g. ["Far Eastern Fells"]
    "wainwrights_wanted": [],     # specific peaks the user wants to bag
    "avoid_busy": None,
    "scenic_priorities": [],      # free-text tags: ["lakes", "ridges", "solitude"]
    "experience_level": None,     # "beginner" | "intermediate" | "experienced"
    "accessibility_needs": None,  # free-text, e.g. "bad knees"
  },
  "group": {
    "size": None,
    "composition": None,          # "couple", "family with kids", "solo", etc.
  },
  "logistics": {
    "num_days": None,
    "base_location": None,
    "max_drive_to_trailhead_minutes": None,
    "travel_style": None,         # "single_base" | "moving" | None
    "accommodation_types": [],
    "accommodation_must_haves": [],
    "budget_per_night": None,
  },
  "plan": {
    "days": [
      # {
      #   "day_number": 1,
      #   "walk_id": 1143,
      #   "accommodation_before": "node/302866833",  # slept here night before walk
      #   "accommodation_after": None,               # null if going home
      #   "notes": "Easy first day to ease in"
      # }
    ],
  },
  "status": "gathering",          # "gathering" | "proposing" | "confirmed" | "iterating"
  "open_questions": [],
  "future_considerations": {
    # Captured but not used in v1 — placeholders so we don't lose the data
    "trip_purpose": None,         # "anniversary", "first Lakes trip", etc.
    "trip_dates": None,
    "transport_mode": None,
  },
}
```

### Important behavioural notes (for the system prompt later)

- **Difficulty must be asked, not inferred** — unless the user mentions an accessibility need, in which case bias toward easier grades.
- **Accommodation timing:** Most users stay the night *before* a walk, not after. The bot should figure this out from context rather than assume — e.g. "I've put you in Keswick the night before Catbells. Are you also staying that night, or heading home after the walk?"
- **Accommodation proximity:** Try to find accommodation within ~15 mins of the trailhead first. If nothing acceptable, expand to ~45 mins and tell the user.
- **Vague queries are fine.** If the user just wants an idea, the bot should propose something reasonable based on broad defaults and treat it as a starting point for iteration. Don't always interrogate.

---

## Scaffolding plan (step by step)

I want to build this incrementally with checkpoints. **Don't try to build everything at once.** The plan:

**Stage 1 — Project structure & dependencies**
- Decide Flask vs FastAPI (ask me)
- Decide embeddings now vs later (ask me)
- Set up directory structure
- Update `requirements.txt` (or create one)
- Stop here, let me check the structure makes sense before proceeding

**Stage 2 — State module**
- Create the trip state object as a Python module (probably a Pydantic model if we go FastAPI, dataclass otherwise)
- Implement `update_trip_state` and `get_trip_state` (likely in-memory dict keyed by session ID for now)
- Stop here, let me check it before proceeding

**Stage 3 — Tool implementations**
- Implement `search_walks`, `get_walk_details`, `search_accommodation` as plain Python functions reading from the existing data files
- Add `rapidfuzz`-based name matching where relevant
- Write a quick test script demonstrating each tool works in isolation
- Stop here for me to verify

**Stage 4 — LLM agent loop**
- Set up the Anthropic SDK with tool definitions matching the signatures above
- Build the agent loop (send message → get response or tool calls → execute tools → loop)
- Use a placeholder system prompt for now — I'll write the real one separately
- Stop here for me to test in isolation (e.g. via a CLI before adding to the web app)

**Stage 5 — Backend endpoint for chat**
- Add `/api/chat` endpoint that takes a user message + session ID and returns the LLM response
- Wire up tool calls to actually execute
- Stop here for me to verify via curl/Postman

**Stage 6 — Map driving (`show_on_map`)**
- Decide on the mechanism: server-sent events, websockets, or polling — discuss with me
- Implement the chosen mechanism end-to-end
- Stop here for me to verify the map updates when the tool is called

**Stage 7 — Chat UI in `index.html`**
- Add a chat panel to the existing `index.html` (don't break existing dropdown/map functionality)
- Connect it to `/api/chat`
- Handle map updates from `show_on_map`
- Final stage — ready for end-to-end testing

---

## Working style

- **Ask clarifying questions whenever you're not sure.** I'd rather you ask than guess.
- **Stop at the end of each stage** and summarise what you've done so I can check before you move on.
- **Don't refactor existing code** unless we explicitly agree to (e.g. the Flask→FastAPI migration if I choose it). The existing `app.py`, scrapers, and HTML should keep working throughout.
- **Show me code as you write it** so I can spot issues early.
- **If you spot a problem with my plan, push back.** I've thought about it but I'm not infallible.

Please start by reading the existing code (`app.py`, `templates/index.html`, the data files in `output/`), then ask me the Flask vs FastAPI and embeddings questions before doing anything else.
