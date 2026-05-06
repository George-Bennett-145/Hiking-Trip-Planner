"""Minimal Walking Britain scraper scaffold.

This first step only fetches and caches the Lake District listing page.
Later steps can reuse the same cache and request helpers.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
WALK_CACHE_DIR = CACHE_DIR / "walks"
OUTPUT_DIR = BASE_DIR / "output"
GPX_DIR = OUTPUT_DIR / "gpx"
LISTING_URL = "https://www.walkingbritain.co.uk/Lake-District-walks"
LISTING_CACHE = CACHE_DIR / "lake_district_walks.html"
LISTING_CSV = OUTPUT_DIR / "walks_listing.csv"
WALKS_CSV = OUTPUT_DIR / "walks.csv"
WALK_PAGE_URL_TEMPLATE = "https://www.walkingbritain.co.uk/walk-{walk_id}-description"
GPX_URL_TEMPLATE = "https://www.walkingbritain.co.uk/download.php?id={walk_id}"
USER_AGENT = "Mozilla/5.0 (compatible; UKHikingTripPlanner/0.1)"
REQUEST_DELAY_SECONDS = 1.5

GPX_MARKER = "★"      # ★ star: GPS file available
PROFILE_MARKER = "☩"  # ☩ cross: route profile available
WALK_ID_RE = re.compile(r"/walk-(\d+)-description")

# Walks excluded from the dataset. The listing page advertises a GPX file but the
# server returns an empty (1-byte) download, so the route polyline is unusable.
# 1158 and 2005 are also near-duplicates ("High Rigg from Legburthwaite" /
# "High Rigg & Legburthwaite").
EXCLUDED_WALK_IDS: set[str] = {"1158", "2005"}


@dataclass
class ListingRow:
    walk_id: str
    title: str
    area: str
    grade: str
    miles: str
    gpx_available: bool
    profile_available: bool


@dataclass
class WalkDetail:
    walk_id: str
    title: str = ""
    area: str = ""
    wainwrights: str = ""
    county: str = ""
    author: str = ""
    grade: str = ""
    distance_miles: str = ""
    distance_km: str = ""
    ascent_feet: str = ""
    ascent_metres: str = ""
    estimated_time: str = ""
    start_lat: str = ""
    start_lng: str = ""
    start_postcode: str = ""
    description: str = ""
    gpx_available: bool = False


def ensure_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WALK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GPX_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        html = response.read().decode("utf-8", errors="replace")

    cache_path.write_text(html, encoding="utf-8")
    sleep(REQUEST_DELAY_SECONDS)
    return html


def fetch_listing_page() -> str:
    return fetch_url(LISTING_URL, LISTING_CACHE)


def walk_page_url(walk_id: str) -> str:
    return WALK_PAGE_URL_TEMPLATE.format(walk_id=walk_id)


def walk_cache_path(walk_id: str) -> Path:
    return WALK_CACHE_DIR / f"walk_{walk_id}.html"


def fetch_walk_page(walk_id: str) -> str:
    return fetch_url(walk_page_url(walk_id), walk_cache_path(walk_id))


def fetch_bytes(url: str, dest_path: Path) -> bytes:
    if dest_path.exists():
        return dest_path.read_bytes()

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        data = response.read()

    dest_path.write_bytes(data)
    sleep(REQUEST_DELAY_SECONDS)
    return data


def gpx_url(walk_id: str) -> str:
    return GPX_URL_TEMPLATE.format(walk_id=walk_id)


def gpx_path(walk_id: str) -> Path:
    return GPX_DIR / f"walk_{walk_id}.gpx"


def download_gpx(walk_id: str) -> tuple[bool, int]:
    """Returns (success, byte_size). On failure success=False."""
    try:
        data = fetch_bytes(gpx_url(walk_id), gpx_path(walk_id))
    except (HTTPError, URLError):
        return False, 0
    if not data.lstrip().startswith(b"<?xml"):
        # Server returned an HTML error page rather than GPX
        gpx_path(walk_id).unlink(missing_ok=True)
        return False, 0
    return True, len(data)


def download_gpx_files(
    details: list[WalkDetail], limit: int | None = None
) -> dict[str, tuple[bool, int]]:
    targets = [d for d in details if d.gpx_available]
    if limit is not None:
        targets = targets[:limit]
    results: dict[str, tuple[bool, int]] = {}
    for i, d in enumerate(targets, start=1):
        ok, size = download_gpx(d.walk_id)
        status = f"{size:>7d} bytes" if ok else "FAILED (source has no usable GPX)"
        print(f"  [{i}/{len(targets)}] walk {d.walk_id}: {status}")
        results[d.walk_id] = (ok, size)
        # Reflect ground truth back onto the detail row
        if not ok:
            d.gpx_available = False
    return results


def parse_listing(html: str) -> list[ListingRow]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[ListingRow] = []

    for table in soup.find_all("table"):
        heading_tag = table.find_previous(["h2", "h3"])
        area = heading_tag.get_text(strip=True) if heading_tag else ""
        # Strip the trailing description after the dash, e.g.
        # "Far Eastern Fells - High Street..." -> "Far Eastern Fells"
        area_short = area.split(" - ", 1)[0].strip()
        if area_short.endswith(" Walks"):
            area_short = area_short[: -len(" Walks")]

        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue  # header row or malformed

            link = cells[0].find("a")
            if link is None:
                continue
            match = WALK_ID_RE.search(link.get("href", ""))
            if not match:
                continue
            walk_id = match.group(1)
            if walk_id in EXCLUDED_WALK_IDS:
                continue

            # Markers (star/cross) live as text siblings inside the first cell
            first_cell_text = cells[0].get_text()
            gpx_available = GPX_MARKER in first_cell_text
            profile_available = PROFILE_MARKER in first_cell_text

            rows.append(
                ListingRow(
                    walk_id=walk_id,
                    title=cells[1].get_text(strip=True),
                    area=area_short,
                    grade=cells[2].get_text(strip=True),
                    miles=cells[3].get_text(strip=True),
                    gpx_available=gpx_available,
                    profile_available=profile_available,
                )
            )
    return rows


LENGTH_RE = re.compile(
    r"Length\s*-\s*([\d.]+)\s*miles?\s*/\s*([\d.]+)\s*km", re.IGNORECASE
)
ASCENT_RE = re.compile(
    r"Ascent\s*-\s*([\d.]+)\s*feet\s*/\s*([\d.]+)\s*metres?", re.IGNORECASE
)
TIME_RE = re.compile(r"Time\s*-\s*(.+?)(?:\s{2,}|\s*Grade\b|$)", re.IGNORECASE)
GRADE_RE = re.compile(r"Grade\s*-\s*([A-Za-z/ ]+)", re.IGNORECASE)
LATLNG_RE = re.compile(
    r"Latitude\s+(-?[\d.]+)\s+Longitude\s+(-?[\d.]+)", re.IGNORECASE
)
POSTCODE_RE = re.compile(
    r"Postcode\s+([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})", re.IGNORECASE
)


def _value_after_label(paragraph_text: str, label: str) -> str:
    """Return the text after `Label - ` (or `Label`) in a paragraph."""
    pattern = re.compile(rf"{re.escape(label)}\s*[-:]?\s*(.+)", re.IGNORECASE)
    m = pattern.search(paragraph_text)
    return m.group(1).strip() if m else ""


def parse_walk_page(walk_id: str, html: str) -> WalkDetail:
    soup = BeautifulSoup(html, "html.parser")
    detail = WalkDetail(walk_id=walk_id)

    h1 = soup.find("h1")
    if h1:
        detail.title = h1.get_text(strip=True)

    # Walk through every <p> looking for label/value pairs
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        if "Wainwrights" in text and not detail.wainwrights:
            detail.wainwrights = _value_after_label(text, "Wainwrights")
        if "County/Area" in text and not detail.county:
            detail.county = _value_after_label(text, "County/Area")
        if text.startswith("Author") and not detail.author:
            detail.author = _value_after_label(text, "Author")

        m = LENGTH_RE.search(text)
        if m:
            detail.distance_miles = m.group(1)
            detail.distance_km = m.group(2)
        m = ASCENT_RE.search(text)
        if m:
            detail.ascent_feet = m.group(1)
            detail.ascent_metres = m.group(2)
        m = TIME_RE.search(text)
        if m and not detail.estimated_time:
            detail.estimated_time = m.group(1).strip()
        m = GRADE_RE.search(text)
        if m and not detail.grade:
            detail.grade = m.group(1).strip()
        m = LATLNG_RE.search(text)
        if m:
            detail.start_lat = m.group(1)
            detail.start_lng = m.group(2)
        m = POSTCODE_RE.search(text)
        if m and not detail.start_postcode:
            detail.start_postcode = m.group(1).strip()

    # Description: paragraphs between "Walk Route Description" heading and the next heading
    desc_heading = soup.find(
        lambda t: t.name in ("h2", "h3") and "Walk Route Description" in t.get_text()
    )
    if desc_heading:
        paragraphs: list[str] = []
        sib = desc_heading.find_next_sibling()
        while sib is not None:
            if sib.name in ("h1", "h2", "h3"):
                break
            if sib.name == "p":
                txt = sib.get_text(" ", strip=True)
                if txt:
                    paragraphs.append(txt)
            sib = sib.find_next_sibling()
        detail.description = "\n\n".join(paragraphs)

    # GPX availability: the page links to /walk-{id}-gps when present
    gpx_link = soup.find("a", href=re.compile(rf"/walk-{walk_id}-gps"))
    detail.gpx_available = gpx_link is not None

    return detail


def fetch_and_parse_walks(
    listing: list[ListingRow], limit: int | None = None
) -> list[WalkDetail]:
    targets = listing if limit is None else listing[:limit]
    details: list[WalkDetail] = []
    for i, row in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] walk {row.walk_id}: {row.title}")
        try:
            html = fetch_walk_page(row.walk_id)
        except (HTTPError, URLError) as error:
            print(f"  ! fetch failed: {error}")
            continue
        detail = parse_walk_page(row.walk_id, html)
        # Carry over the area from the listing (per-page area is not always cleanly stated)
        if not detail.area:
            detail.area = row.area
        details.append(detail)
    return details


def write_walks_csv(details: list[WalkDetail], path: Path) -> None:
    fieldnames = list(asdict(details[0]).keys()) if details else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in details:
            writer.writerow(asdict(d))


def print_walk_detail(d: WalkDetail) -> None:
    print(f"  walk_id:        {d.walk_id}")
    print(f"  title:          {d.title}")
    print(f"  area:           {d.area}")
    print(f"  wainwrights:    {d.wainwrights}")
    print(f"  county:         {d.county}")
    print(f"  author:         {d.author}")
    print(f"  grade:          {d.grade}")
    print(f"  distance:       {d.distance_miles} mi / {d.distance_km} km")
    print(f"  ascent:         {d.ascent_feet} ft / {d.ascent_metres} m")
    print(f"  estimated_time: {d.estimated_time}")
    print(f"  start lat/lng:  {d.start_lat}, {d.start_lng}")
    print(f"  start postcode: {d.start_postcode}")
    print(f"  gpx_available:  {d.gpx_available}")
    desc_preview = d.description[:160].replace("\n", " ")
    suffix = "..." if len(d.description) > 160 else ""
    print(f"  description:    ({len(d.description)} chars) {desc_preview}{suffix}")


def write_listing_csv(rows: list[ListingRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarise_listing(rows: list[ListingRow]) -> None:
    print(f"Total walks parsed: {len(rows)}")
    print(f"With GPX: {sum(1 for r in rows if r.gpx_available)}")
    print(f"With profile: {sum(1 for r in rows if r.profile_available)}")

    by_area: dict[str, int] = {}
    for r in rows:
        by_area[r.area] = by_area.get(r.area, 0) + 1
    print("By area:")
    for area, count in by_area.items():
        print(f"  {area}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walking Britain Lake District scraper")
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        metavar="N",
        help="Fetch and parse the first N walk pages, print their data, and stop.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch and parse every walk page and write output/walks.csv.",
    )
    args = parser.parse_args()

    ensure_directories()

    try:
        html = fetch_listing_page()
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Failed to fetch listing page: {error}") from error

    print(f"Fetched listing page: {len(html)} characters")
    print(f"Cache saved to: {LISTING_CACHE}")

    rows = parse_listing(html)
    summarise_listing(rows)

    if rows:
        write_listing_csv(rows, LISTING_CSV)
        print(f"Wrote listing CSV: {LISTING_CSV}")

    if args.trial > 0 or args.all:
        if args.all:
            print(f"\n--- Full run: fetching all {len(rows)} walk pages ---")
            details = fetch_and_parse_walks(rows)
        else:
            print(f"\n--- Trial run: fetching first {args.trial} walk page(s) ---")
            details = fetch_and_parse_walks(rows, limit=args.trial)
            for d in details:
                print()
                print_walk_detail(d)

        if details:
            write_walks_csv(details, WALKS_CSV)
            print(f"\nWrote walks CSV: {WALKS_CSV} ({len(details)} rows)")

        print(f"\n--- Downloading GPX files into {GPX_DIR} ---")
        results = download_gpx_files(details)
        successes = sum(1 for ok, _ in results.values() if ok)
        total_bytes = sum(size for ok, size in results.values() if ok)
        print(
            f"GPX download complete: {successes}/{len(results)} succeeded, "
            f"{total_bytes:,} bytes total"
        )

        # Re-write walks.csv so gpx_available reflects actual download outcome
        if details:
            write_walks_csv(details, WALKS_CSV)
            print(f"Re-wrote walks CSV with corrected gpx_available flags: {WALKS_CSV}")


if __name__ == "__main__":
    main()