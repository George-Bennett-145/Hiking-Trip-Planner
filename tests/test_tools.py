"""Smoke tests for server/tools.py.

Run from the project root:
    python -m tests.test_tools
"""

from server.tools import (
    get_walk_details,
    search_accommodation,
    search_walks,
)


# Approximate centre of Keswick — handy reference point for proximity tests.
KESWICK_LAT, KESWICK_LON = 54.6013, -3.1346


def test_search_walks_no_filters_returns_default_limit():
    results = search_walks()
    assert len(results) == 10
    print(f"[ok] no-filter search returns default limit ({len(results)})")


def test_search_walks_difficulty_filter():
    results = search_walks(difficulty=["easy"], limit=300)
    assert results, "expected at least one easy walk"
    assert all(r["grade"] == "easy" for r in results)
    print(f"[ok] difficulty filter: {len(results)} 'easy' walks")


def test_search_walks_distance_filter():
    results = search_walks(distance_miles_max=3.0, limit=300)
    assert results, "expected at least one walk under 3 miles"
    assert all(r["distance_miles"] <= 3.0 for r in results)
    print(f"[ok] distance filter: {len(results)} walks under 3 miles")


def test_search_walks_ascent_filter():
    results = search_walks(ascent_metres_max=300, limit=300)
    assert results, "expected at least one walk under 300m ascent"
    assert all(r["ascent_metres"] <= 300 for r in results)
    print(f"[ok] ascent filter: {len(results)} walks under 300m ascent")


def test_search_walks_area_exact_match():
    results = search_walks(areas=["Far Eastern Fells"], limit=300)
    assert results, "expected matches for Far Eastern Fells"
    assert all(r["area"] == "Far Eastern Fells" for r in results)
    print(f"[ok] area exact match: {len(results)} walks in Far Eastern Fells")


def test_search_walks_area_fuzzy_match():
    # Lower-case partial query should still resolve to the canonical area.
    fuzzy = search_walks(areas=["far eastern"], limit=300)
    exact = search_walks(areas=["Far Eastern Fells"], limit=300)
    assert fuzzy, "fuzzy match returned no results"
    assert {w["walk_id"] for w in fuzzy} == {w["walk_id"] for w in exact}
    print("[ok] area fuzzy match resolves to same set as exact match")


def test_search_walks_query_matches_title():
    results = search_walks(query="scafell", limit=50)
    assert results, "expected at least one Scafell walk"
    assert all("scafell" in r["title"].lower() for r in results)
    print(f"[ok] query='scafell' returns {len(results)} walks, all matching by title")


def test_search_walks_query_combines_with_filters():
    # No Scafell walk is graded "easy" — combining should return [].
    results = search_walks(query="scafell", difficulty=["easy"], limit=50)
    assert results == []
    # Without the difficulty filter, we get the full set back.
    relaxed = search_walks(query="scafell", limit=50)
    assert len(relaxed) >= 5
    print(f"[ok] query combines with filters; relaxing returns {len(relaxed)} walks")


def test_search_walks_query_is_case_insensitive():
    lower = {w["walk_id"] for w in search_walks(query="helvellyn", limit=50)}
    upper = {w["walk_id"] for w in search_walks(query="HELVELLYN", limit=50)}
    assert lower and lower == upper
    print(f"[ok] query is case-insensitive ({len(lower)} Helvellyn walks)")


def test_search_walks_proximity():
    results = search_walks(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=5, limit=50
    )
    assert results, "expected walks within 5km of Keswick"
    print(f"[ok] proximity filter near Keswick: {len(results)} walks within 5km")


def test_search_walks_short_description_truncated():
    results = search_walks(limit=1)
    walk = results[0]
    assert "short_description" in walk
    assert len(walk["short_description"]) <= 400
    # Make sure it's not the full description (we have walks with >400 char desc)
    full = get_walk_details(walk["walk_id"])
    if len(full["description"]) > 400:
        assert walk["short_description"] != full["description"]
    print(f"[ok] short_description capped at 400 chars")


def test_get_walk_details_returns_full_row():
    walk = get_walk_details(1143)  # Hallin Fell from Martindale Church
    assert walk["walk_id"] == 1143
    assert walk["title"].startswith("Hallin Fell")
    assert walk["area"] == "Far Eastern Fells"
    assert isinstance(walk["distance_miles"], float)
    assert isinstance(walk["ascent_metres"], int)
    assert len(walk["description"]) > 400
    print(f"[ok] get_walk_details returns full {len(walk['description'])}-char description")


def test_get_walk_details_unknown_raises():
    try:
        get_walk_details(99999)
    except KeyError:
        print("[ok] unknown walk_id raises KeyError")
        return
    raise AssertionError("expected KeyError for unknown walk_id")


def test_search_accommodation_basic():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=5
    )
    assert results, "expected accommodation within 5km of Keswick"
    assert all(r["distance_km"] <= 5 for r in results)
    distances = [r["distance_km"] for r in results]
    assert distances == sorted(distances), "results not sorted by distance"
    print(f"[ok] accommodation search: {len(results)} results, sorted by distance")


def test_search_accommodation_type_filter():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=20, types=["hotel"]
    )
    assert results, "expected hotels within 20km of Keswick"
    assert all(r["tourism"] == "hotel" for r in results)
    print(f"[ok] type filter: {len(results)} hotels within 20km of Keswick")


def test_search_accommodation_osm_id_format():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=5, limit=1
    )
    assert results
    osm_id = results[0]["id"]
    prefix, sep, num = osm_id.partition("/")
    assert sep == "/", f"expected OSM-style id, got {osm_id!r}"
    assert prefix in ("node", "way", "relation")
    assert num.isdigit()
    print(f"[ok] accommodation id is OSM-style ({osm_id})")


def test_search_accommodation_respects_limit():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=50, limit=5
    )
    assert len(results) <= 5
    print(f"[ok] limit respected: returned {len(results)} of requested 5")


# -------- new behaviour: search_walks proximity sorting + distance field --------

def test_search_walks_proximity_includes_distance_field():
    results = search_walks(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=10, limit=20
    )
    assert results
    assert all("distance_from_query_km" in r for r in results), \
        "every proximity result should expose distance_from_query_km"
    print(f"[ok] proximity results include distance_from_query_km field")


def test_search_walks_proximity_sorted_nearest_first():
    results = search_walks(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=10, limit=20
    )
    distances = [r["distance_from_query_km"] for r in results]
    assert distances == sorted(distances), \
        f"expected ascending distance order, got {distances}"
    print(f"[ok] proximity results sorted nearest-first ({len(results)} walks)")


def test_search_walks_no_distance_field_without_proximity():
    # When the model uses non-proximity filters, the distance field should
    # not appear (it would be misleading without an anchor).
    results = search_walks(difficulty=["easy"], limit=5)
    assert results
    assert not any("distance_from_query_km" in r for r in results)
    print(f"[ok] non-proximity searches omit distance_from_query_km")


# ---- new behaviour: search_accommodation routed distance + crow-flies field ---

def test_search_accommodation_includes_both_distance_fields():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=5, limit=10
    )
    assert results
    for r in results:
        assert "distance_km" in r, "missing distance_km (routed)"
        assert "crow_flies_km" in r, "missing crow_flies_km"
    print(f"[ok] accommodation results expose distance_km and crow_flies_km")


def test_search_accommodation_routed_at_least_as_far_as_crow_flies():
    # By geometry, a real route can never be shorter than the straight line.
    # If routing failed for some entry it falls back to crow-flies (equal).
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=10, limit=15
    )
    assert results
    for r in results:
        # Allow small floating-point slack (~50 m) since the graph snaps to
        # the nearest node which may sit a few metres inside the crow-flies line.
        assert r["distance_km"] + 0.05 >= r["crow_flies_km"], (
            f"routed {r['distance_km']} < crow-flies {r['crow_flies_km']} "
            f"for {r['name']!r}"
        )
    print(f"[ok] all routed distances >= crow-flies (geometry sanity check)")


def test_search_accommodation_sorted_by_routed_distance():
    results = search_accommodation(
        near_lat=KESWICK_LAT, near_lon=KESWICK_LON, radius_km=10, limit=15
    )
    distances = [r["distance_km"] for r in results]
    assert distances == sorted(distances), \
        f"expected sort by routed distance, got {distances}"
    print(f"[ok] accommodation sorted by routed distance ({len(results)} entries)")


def test_search_accommodation_lake_detour_case():
    # Hallin Fell trailhead at Martindale Church sits on the east side of
    # Ullswater. With a 15 km crow-flies radius we should pull in some
    # accommodation on the west side (Glenridding/Patterdale) where the
    # routed distance is materially larger than the straight line because
    # the route has to skirt the lake.
    HALLIN_FELL_TH_LAT, HALLIN_FELL_TH_LON = 54.5631, -2.8728
    results = search_accommodation(
        near_lat=HALLIN_FELL_TH_LAT,
        near_lon=HALLIN_FELL_TH_LON,
        radius_km=15,
        limit=30,
    )
    assert results, "expected accommodation around Ullswater"
    # We expect at least one entry whose routed distance is materially
    # longer than the crow-flies distance — the lake-detour signature.
    detours = [r for r in results if r["distance_km"] > r["crow_flies_km"] + 2.0]
    assert detours, (
        "expected at least one entry where routed distance exceeds "
        "crow-flies by >2 km (lake detour) — none found"
    )
    worst = max(detours, key=lambda r: r["distance_km"] - r["crow_flies_km"])
    print(
        f"[ok] lake detour detected: {worst['name']!r} "
        f"crow-flies {worst['crow_flies_km']:.1f} km vs routed "
        f"{worst['distance_km']:.1f} km"
    )


def test_search_accommodation_walking_vs_driving_modes():
    HALLIN_FELL_TH_LAT, HALLIN_FELL_TH_LON = 54.5631, -2.8728
    walking = search_accommodation(
        near_lat=HALLIN_FELL_TH_LAT, near_lon=HALLIN_FELL_TH_LON,
        radius_km=10, limit=10, mode="walking",
    )
    driving = search_accommodation(
        near_lat=HALLIN_FELL_TH_LAT, near_lon=HALLIN_FELL_TH_LON,
        radius_km=10, limit=10, mode="driving",
    )
    assert walking and driving
    # Both modes should return results; driving routes are usually a bit
    # longer because they're constrained to roads, but we don't enforce a
    # strict inequality (a few entries can tie).
    walking_by_id = {r["id"]: r["distance_km"] for r in walking}
    driving_by_id = {r["id"]: r["distance_km"] for r in driving}
    common = set(walking_by_id) & set(driving_by_id)
    assert common, "expected overlap between walking and driving result sets"
    print(
        f"[ok] both modes return results; "
        f"{len(common)} entries appear in both (walking/driving)"
    )


if __name__ == "__main__":
    test_search_walks_no_filters_returns_default_limit()
    test_search_walks_difficulty_filter()
    test_search_walks_distance_filter()
    test_search_walks_ascent_filter()
    test_search_walks_area_exact_match()
    test_search_walks_area_fuzzy_match()
    test_search_walks_query_matches_title()
    test_search_walks_query_combines_with_filters()
    test_search_walks_query_is_case_insensitive()
    test_search_walks_proximity()
    test_search_walks_short_description_truncated()
    test_get_walk_details_returns_full_row()
    test_get_walk_details_unknown_raises()
    test_search_accommodation_basic()
    test_search_accommodation_type_filter()
    test_search_accommodation_osm_id_format()
    test_search_accommodation_respects_limit()
    test_search_walks_proximity_includes_distance_field()
    test_search_walks_proximity_sorted_nearest_first()
    test_search_walks_no_distance_field_without_proximity()
    test_search_accommodation_includes_both_distance_fields()
    test_search_accommodation_routed_at_least_as_far_as_crow_flies()
    test_search_accommodation_sorted_by_routed_distance()
    test_search_accommodation_lake_detour_case()
    test_search_accommodation_walking_vs_driving_modes()
    print("\nAll tools tests passed.")
