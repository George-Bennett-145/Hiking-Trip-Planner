"""Route between two lat/lon points using a cached OSM network.

Two modes are supported:
  walking  - uses output/lake_district_walking_graph.graphml
             (footpaths, bridleways, roads usable on foot)
  driving  - uses output/lake_district_driving_graph.graphml
             (drivable roads only)

Each graph is loaded lazily on first use of its mode and held in memory
for the lifetime of the process. The first call pays the load cost
(~8s walking, ~2s driving); subsequent calls reuse the cached graph.

Run directly to test:
    python -m server.routing
"""

import math
import time
from pathlib import Path
from typing import Literal, Optional

import networkx as nx
import osmnx as ox

Mode = Literal["walking", "driving"]

# Anchor to this file's location so the module works no matter what cwd
# the server was started from.
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
GRAPH_PATHS: dict[str, Path] = {
    "walking": _OUTPUT_DIR / "lake_district_walking_graph.graphml",
    "driving": _OUTPUT_DIR / "lake_district_driving_graph.graphml",
}

_BUILD_HINT: dict[str, str] = {
    "walking": "build_walking_graph.py",
    "driving": "build_driving_graph.py",
}

# Module-level cache, one slot per mode. Populated by _get_graph().
_graphs: dict[str, Optional[nx.MultiDiGraph]] = {"walking": None, "driving": None}


def _get_graph(mode: Mode = "walking") -> nx.MultiDiGraph:
    if mode not in _graphs:
        raise ValueError(f"Unknown mode {mode!r}, expected 'walking' or 'driving'")
    if _graphs[mode] is None:
        path = GRAPH_PATHS[mode]
        if not path.exists():
            raise FileNotFoundError(
                f"{mode.capitalize()} graph not found at {path}. "
                f"Run {_BUILD_HINT[mode]} first."
            )
        _graphs[mode] = ox.load_graphml(path)
    return _graphs[mode]


def prewarm() -> None:
    """Pre-load both graphs into memory.

    Call this at server startup so the first /api/connector request does
    not pay the load cost (~8s walking, ~2s driving). Safe to call after
    the graphs are already cached: subsequent calls are effectively no-ops.
    """
    _get_graph("walking")
    _get_graph("driving")


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _node_ids_between(
    G: nx.MultiDiGraph, lat1: float, lon1: float, lat2: float, lon2: float
) -> list[int]:
    """Return the list of OSM node IDs for the shortest path in graph G."""
    start_node = ox.distance.nearest_nodes(G, lon1, lat1)
    end_node = ox.distance.nearest_nodes(G, lon2, lat2)
    if start_node == end_node:
        return [start_node]
    return nx.shortest_path(G, start_node, end_node, weight="length")


def route_between(
    lat1: float, lon1: float, lat2: float, lon2: float,
    mode: Mode = "walking",
) -> list[tuple[float, float]]:
    """Return the shortest path from (lat1, lon1) to (lat2, lon2).

    `mode` selects the underlying network: 'walking' uses footpaths and
    walkable roads, 'driving' uses drivable roads only.

    Result is a list of (lat, lon) tuples following the chosen network.
    The first and last entries are the snapped graph nodes, which may be
    a short distance from the requested input points.

    Raises FileNotFoundError if the graph cache for that mode is missing,
    and networkx.NetworkXNoPath if no route exists between the snapped
    nodes (e.g. one is on an isolated island in the network).
    """
    G = _get_graph(mode)
    node_ids = _node_ids_between(G, lat1, lon1, lat2, lon2)
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_ids]


def route_highway_breakdown(
    lat1: float, lon1: float, lat2: float, lon2: float,
    mode: Mode = "walking",
) -> dict[str, float]:
    """Return metres on each OSM highway type along the shortest route.

    Keys are highway type strings (e.g. 'footway', 'residential', 'primary').
    Values are total metres of that type in the route.

    OSM sometimes stores highway as a list (when a way has multiple tags);
    in that case each tag in the list is credited with the full edge length
    so the totals add up to more than the route distance. This is rare.

    Useful for auditing whether a route uses sensible road types for the
    given mode (e.g. driving routes should mostly be primary/secondary/
    tertiary/residential, walking routes should favour footway/path).
    """
    G = _get_graph(mode)
    node_ids = _node_ids_between(G, lat1, lon1, lat2, lon2)

    totals: dict[str, float] = {}
    for u, v in zip(node_ids[:-1], node_ids[1:]):
        # MultiDiGraph: G[u][v] is a dict of parallel edges keyed 0, 1, ...
        # Take the one with the shortest length (same choice Dijkstra made).
        edge_data = min(G[u][v].values(), key=lambda d: d.get("length", 0))
        length = edge_data.get("length", 0.0)
        highway = edge_data.get("highway", "unknown")
        # highway can be a list when OSM way has multiple tags
        tags = highway if isinstance(highway, list) else [highway]
        for tag in tags:
            totals[tag] = totals.get(tag, 0.0) + length

    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def snap_distance_m(lat: float, lon: float, mode: Mode = "walking") -> float:
    """How far the nearest graph node is from a given lat/lon, in metres.

    Useful as a sanity check before routing: if this is large (say > 500m),
    the input point is far from any path/road in the chosen network and
    the route's start or end will be visibly disconnected.
    """
    G = _get_graph(mode)
    node_id = ox.distance.nearest_nodes(G, lon, lat)
    node = G.nodes[node_id]
    return _haversine_m(lat, lon, node["y"], node["x"])


# --------------------------------------------------------------------- main

def _polyline_length_m(polyline: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(polyline)):
        total += _haversine_m(
            polyline[i - 1][0], polyline[i - 1][1],
            polyline[i][0], polyline[i][1],
        )
    return total


def _run_test(label: str, lat1: float, lon1: float, lat2: float, lon2: float, mode: Mode = "walking") -> None:
    print(f"\n--- {label}  [mode={mode}] ---")
    print(f"  from ({lat1:.5f}, {lon1:.5f})")
    print(f"  to   ({lat2:.5f}, {lon2:.5f})")
    crow = _haversine_m(lat1, lon1, lat2, lon2)
    print(f"  crow-flies distance: {crow:.0f} m")

    snap1 = snap_distance_m(lat1, lon1, mode)
    snap2 = snap_distance_m(lat2, lon2, mode)
    print(f"  snap distances: start {snap1:.0f}m, end {snap2:.0f}m")

    t0 = time.perf_counter()
    poly = route_between(lat1, lon1, lat2, lon2, mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    route_len = _polyline_length_m(poly)
    print(f"  route: {len(poly)} points, {route_len:.0f} m")
    if crow > 0:
        print(f"  detour ratio: {route_len / crow:.2f}x crow-flies")
    else:
        print(f"  detour ratio: n/a (identical points)")
    print(f"  query time: {elapsed_ms:.0f} ms")

    breakdown = route_highway_breakdown(lat1, lon1, lat2, lon2, mode)
    total_m = sum(breakdown.values())
    print(f"  highway breakdown:")
    for htype, metres in breakdown.items():
        pct = metres / total_m * 100 if total_m > 0 else 0
        print(f"    {htype:<20} {metres:>7.0f} m  ({pct:.0f}%)")


if __name__ == "__main__":
    for mode in ("walking", "driving"):
        print(f"\nLoading {mode} graph...")
        t0 = time.perf_counter()
        G = _get_graph(mode)  # type: ignore[arg-type]
        print(f"  loaded in {time.perf_counter() - t0:.1f}s "
              f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")

    # Same Ambleside -> Catbells route compared in both modes.
    _run_test(
        "Ambleside -> Catbells (walk_2036) start",
        54.4313791, -2.9621712,
        54.563958, -2.8752953,
        mode="walking",
    )
    _run_test(
        "Ambleside -> Catbells (walk_2036) start",
        54.4313791, -2.9621712,
        54.563958, -2.8752953,
        mode="driving",
    )

    # Driving-only test: same as above but a longer cross-district drive.
    _run_test(
        "Ambleside -> Keswick centre",
        54.4313791, -2.9621712,
        54.5996,   -3.1347,
        mode="driving",
    )

    # Edge case kept on walking mode.
    _run_test(
        "Identical start and end",
        54.4313791, -2.9621712,
        54.4313791, -2.9621712,
        mode="walking",
    )
