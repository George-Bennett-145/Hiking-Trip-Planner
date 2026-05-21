# Lake District Trip Planner

A natural-language hiking trip planner for England's Lake District. Tell it what
kind of walk you want and it picks one from a curated catalogue, finds an
accommodation nearby, and renders both on an interactive map. Built as the
capstone project for the Digital Futures programme.

**Live demo:** https://lake-district-planner-644255749905.europe-west2.run.app

The first request after the service has been idle takes around ten seconds
while it warms up; subsequent requests are immediate.

---

## What it does

You describe the trip you want in plain English. For example:

> *"I'd like a moderate day walk near Keswick with a pub to stay at, and I want
> to walk to the trailhead rather than drive."*

The assistant searches a dataset of around 220 Lake District walks scraped
from Walking Britain, picks one that matches the difficulty, distance, and
location, then finds an accommodation routed through the OpenStreetMap
walking network (not crow-flies) so the "near the trailhead" claim actually
holds up. The route polyline and accommodation marker appear on a Leaflet
map alongside the chat, and you can refine the choice as many times as you
like: ask for something harder, swap the hotel, change to driving, move to a
different area. Each refinement re-runs the relevant search and re-renders
the map.

It is deliberately narrow in scope. No bookings, no weather, no live
conditions, no walks outside the Lake District. The bet is that doing one
small thing properly is more useful than doing a generic everything-planner
badly.

---

## What makes it interesting

**Grounded retrieval through tool calling, not context stuffing.** The LLM
never sees the dataset. It has six tools (`search_walks`, `get_walk_details`,
`search_accommodation`, `update_trip_state`, `get_trip_state`, `show_on_map`)
and calls them when it needs information. A typical reply touches maybe ten
records out of more than a thousand. This keeps the context small, the cost
low, and prevents hallucinated walks because the model can only quote what
the tools return.

**Real-network routing, not Haversine.** Accommodation distance from the
trailhead uses Dijkstra over a cached OSMnx graph of the Lake District,
weighted by the OSM `length` attribute (metres on the actual path or road).
Two graphs are loaded at startup, walking and driving, so a "walk to the
trailhead" preference gives a different ranking than "drive to the
trailhead". A hotel three miles around a lake is correctly ranked as
further than a hotel one mile across a footpath, which a straight-line
distance would get backwards.

**Streamed agent loop with a deterministic safety net.** Replies stream
back to the browser over Server-Sent Events, so text appears as the model
produces it rather than as one delayed block. The agent loop runs up to
ten iterations per turn, interleaving tool calls and text. If the model
researches a new walk but forgets to call `show_on_map`, the loop
synthesises the call from the latest search results so the map can never
go stale relative to the reply text. A guardrail, not a hope.

**Structured state with strict validation.** Every conversation has a
typed `TripState` (Pydantic, `extra="forbid"`) covering preferences,
group, logistics, plan, and map. The LLM updates it through a
`update_trip_state` tool that deep-merges and re-validates, so a typo in
a field name surfaces immediately as an error instead of silently
corrupting the trip.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Anthropic Claude Haiku 4.5 via OpenRouter | Fast, strong instruction-following, supports tool calling |
| Backend | FastAPI + Uvicorn | Async, SSE-friendly, first-class Pydantic support |
| State | In-memory Pydantic models, per-session UUID | Strict typing, deep-merge updates |
| Routing | OSMnx + NetworkX over a cached OSM graph | Real path/road distances, not crow-flies |
| Frontend | Leaflet + vanilla JS + OpenStreetMap tiles | No build step, minimal dependencies |
| Scraping | BeautifulSoup over urllib with disk-cached HTML | Polite, resumable, debuggable |
| Deployment | Google Cloud Run via Cloud Build, Secret Manager | One-command deploy, free when idle |

---

## How the pieces fit together

```
+----------------------------+
|  Browser                   |
|  - Leaflet map             |
|  - Chat panel (SSE client) |
|  - Polls session state     |
+--------------+-------------+
               |
               v
+----------------------------+
|  FastAPI on Cloud Run      |
|  - SSE-streamed agent loop |
|  - Per-session trip state  |
|  - GPX + walks + accom API |
+--+----------+--------+-----+
   |          |        |
   v          v        v
+-----+   +--------+   +----------------+
| LLM |   | OSMnx  |   | walks.csv      |
| via |   | graphs |   | accom.json     |
| OR  |   |        |   | gpx/*.gpx      |
+-----+   +--------+   +----------------+
```

For the full architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project layout

```
Capstone_Project/
+-- Scraper.py                    Walking Britain scraper
+-- fetch_accommodation.py        OSM Overpass accommodation fetcher
+-- build_walking_graph.py        OSMnx walking-network builder
+-- build_driving_graph.py        OSMnx driving-network builder
+-- server/
|   +-- main.py                   FastAPI app, routes, SSE
|   +-- llm.py                    Agent loop, system prompt, tool schemas
|   +-- tools.py                  Walk and accommodation search tools
|   +-- state.py                  Pydantic trip state, session store
|   +-- routing.py                Dijkstra over the OSM graph
|   +-- gpx.py                    GPX parsing
|   +-- data.py                   CSV/JSON loaders
+-- templates/index.html          Single-page UI
+-- tests/                        State, tools, API routes, LLM behaviour
+-- output/
|   +-- walks.csv                 ~219 walks
|   +-- accommodation.json        ~870 places
|   +-- gpx/                      one .gpx per walk
|   +-- *.graphml                 OSMnx walking + driving graphs
+-- Dockerfile
+-- deploy.ps1                    Cloud Run deploy
+-- run.ps1                       Local setup + launch
+-- ARCHITECTURE.md               Full architecture writeup
```

---

## Running it locally

Requires Python 3.10 or later and an [OpenRouter](https://openrouter.ai/)
API key (Claude Haiku is paid; OpenRouter's free Llama models also work).
The convenience scripts are PowerShell for Windows, but the app itself is
plain Python and runs anywhere.

You will need a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
MODEL=anthropic/claude-haiku-4-5-20251001
```

**On Windows**, the one-shot bootstrap is `run.ps1`:

```powershell
./run.ps1
```

This creates `.venv`, installs dependencies, builds the routing graphs from
OpenStreetMap if they are not already cached (one-off, around three
minutes), and launches the app at `http://localhost:8000`.

**On macOS or Linux**, run the equivalent steps manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_walking_graph.py
python build_driving_graph.py
uvicorn server.main:app --port 8000
```

---

## Deploying to Cloud Run

`deploy.ps1` is idempotent end-to-end:

```powershell
./deploy.ps1
```

It uploads your local `OPENROUTER_API_KEY` to Secret Manager on the first
run, grants the Cloud Run service account access to the secret, then runs
`gcloud run deploy --source .` so Google Cloud Build builds the image
remotely. No local Docker required. The first build takes 8 to 12 minutes;
subsequent builds are 3 to 5.

Defaults are tuned for the demo: `min-instances=0` (free when idle, ~10 s
cold start), `max-instances=1` (single-instance is a design constraint, see
below), 2 GiB memory, 2 vCPU.

---

## Known limits

These are documented honestly because they shaped the design:

- **In-memory session state.** A Cloud Run restart wipes every conversation.
  Acceptable for a single-user demo, but a real product would back state
  with Redis or Firestore.
- **Single-instance only.** Map updates are delivered via 2-second polling
  of in-process state. A second Cloud Run instance would not see the first
  instance's sessions. `max-instances=1` is therefore deliberate.
- **Lake District only.** Bounding box, area names, named-place coordinates,
  and the system prompt are all Lake District specific. Extending to
  Snowdonia or Scotland is plumbing (scrape, build new graphs, edit the
  prompt), not configuration.
- **GPX coverage gaps.** Two walks listed by Walking Britain as having GPX
  files (1158 and 2005) return empty downloads and are excluded.

---

## Data sources and acknowledgements

- Walk metadata and GPX tracks scraped from
  [Walking Britain](https://www.walkingbritain.co.uk/), with respect (one
  request every 1.5 seconds, raw HTML cached locally so development never
  re-hits the source). Not affiliated with Walking Britain.
- Accommodation data from [OpenStreetMap](https://www.openstreetmap.org/)
  via the [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API),
  under the
  [Open Database License (ODbL)](https://www.openstreetmap.org/copyright).
- Routing graphs built with [OSMnx](https://osmnx.readthedocs.io/) from
  OpenStreetMap data.
- Map tiles from [OpenStreetMap](https://www.openstreetmap.org/), rendered
  with [Leaflet](https://leafletjs.com/).
- LLM access through [OpenRouter](https://openrouter.ai/), currently using
  Anthropic's Claude Haiku 4.5.

Built as the capstone project for the
[Digital Futures](https://digitalfutures.com/) data and AI engineering
programme.
