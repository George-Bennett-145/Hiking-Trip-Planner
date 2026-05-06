# Project Context - UK Hiking Trip Planner

## What this project is

A capstone project for the Digital Futures programme. The product helps users plan hiking trips in the UK by combining trail discovery, accommodation matching, and map visualisation in one natural language interface. The user describes what they want ("moderate two-day hike in the Lake District, pub stay overnight") and the system retrieves matching trails and nearby accommodation, then visualises everything on a map.

## Current phase

Data gathering. We are scraping Walking Britain (walkingbritain.co.uk) to build the trail dataset. Accommodation data comes later, likely from OpenStreetMap via the Overpass API.

## Data source: Walking Britain

Walking Britain has 200+ Lake District walks with everything we need per walk. The site structure is:

### Listing page
- URL: `https://www.walkingbritain.co.uk/Lake-District-walks`
- Contains HTML tables with columns: Walk ID, Description, Grade, Miles
- Walks are grouped by area (Far Eastern Fells, Eastern Fells, Central Fells, Northern Fells, North Western Fells, Western Fells, Southern Fells, Outlying & Lesser Fells)
- Walk IDs link to individual pages via pattern: `walk-{ID}-description`
- A star symbol indicates a GPS file is available
- A cross symbol indicates a route profile is available

### Individual walk pages
- URL pattern: `https://www.walkingbritain.co.uk/walk-{ID}-description`
- Contains: title, area, Wainwrights tagged, county, author, length (miles and km), ascent (feet and metres), estimated time, grade, full route description (multiple paragraphs), start point lat/lng, start postcode, nearby walks, photos
- Example page: `https://www.walkingbritain.co.uk/walk-2036-description` (Catbells from Gutherscale)

### GPX files
- URL pattern: `https://www.walkingbritain.co.uk/walk-{ID}-gps`
- Standard GPX format with lat/lng/elevation trackpoints
- Example: The Catbells GPX (walk 2036) has 400+ coordinate points with elevation data
- These provide the route polylines for map visualisation

## What the scraper needs to produce

### 1. A CSV file with one row per walk containing:
- walk_id (from the listing page)
- title (from individual walk page)
- area (e.g. "North Western Fells", from individual walk page)
- wainwrights (comma-separated list if applicable)
- grade (easy, easy/mod, moderate, mod/hard, hard, very hard, severe)
- distance_miles
- distance_km
- ascent_feet
- ascent_metres
- estimated_time
- start_lat
- start_lng
- start_postcode
- description (full route description text)
- gpx_available (boolean)

### 2. A folder of GPX files
- One GPX file per walk where available
- Named as `walk_{ID}.gpx`
- These contain the route geometry for map visualisation

## Scraping approach

1. Scrape the listing page to get all walk IDs, descriptions, grades, and distances
2. For each walk ID, visit the individual walk page and extract the full metadata
3. For each walk with a GPS file available, download the GPX file
4. Be respectful: add delays between requests, cache responses, don't hammer the server
5. Save the CSV and GPX files locally

## Technical stack (for later phases, not the scraper)

- Python (primary language)
- Vector database for embedded trail descriptions (for RAG/retrieval)
- LLM via API for natural language understanding and trip planning
- Leaflet with OpenStreetMap tiles for map visualisation
- Met Office DataPoint API for weather (stretch goal)
- OpenStreetMap Overpass API for accommodation data

## Important notes

- Do not use em dashes in any written output
- Be respectful when scraping: use delays between requests (1-2 seconds minimum)
- Cache raw HTML responses so we don't re-scrape during development
- The GPX files are the route geometry for map polylines
- The CSV descriptions are what gets embedded for retrieval/RAG later
- MVP region is Lake District only, expanding to Wales and Scotland later
