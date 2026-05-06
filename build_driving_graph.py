"""Download and cache the Lake District driving network.

Sibling to build_walking_graph.py. Pulls the OSM driving network (roads
that cars can use, excluding footpaths/bridleways) for the same Lake
District bounding box, saves it as GraphML to output/, then reloads to
confirm the cache.

Used for routing the connector segment when the user is driving from
their accommodation to the trailhead.

Run from the project root:
    python build_driving_graph.py
"""

import time
from pathlib import Path

import osmnx as ox

BBOX_WEST = -3.45
BBOX_SOUTH = 54.35
BBOX_EAST = -2.75
BBOX_NORTH = 54.80

OUTPUT_PATH = Path("output/lake_district_driving_graph.graphml")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading driving network for Lake District bbox...")
    print(f"  bbox (W,S,E,N) = ({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH})")
    t0 = time.perf_counter()
    G = ox.graph_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        network_type="drive",
    )
    download_secs = time.perf_counter() - t0
    print(f"  download took {download_secs:.1f}s")
    print(f"  nodes: {G.number_of_nodes():,}")
    print(f"  edges: {G.number_of_edges():,}")

    print(f"\nSaving to {OUTPUT_PATH}...")
    t0 = time.perf_counter()
    ox.save_graphml(G, OUTPUT_PATH)
    save_secs = time.perf_counter() - t0
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  saved in {save_secs:.1f}s, file size {size_mb:.1f} MB")

    print("\nReloading from disk to verify cache...")
    t0 = time.perf_counter()
    G2 = ox.load_graphml(OUTPUT_PATH)
    load_secs = time.perf_counter() - t0
    print(f"  reload took {load_secs:.1f}s")
    print(f"  nodes: {G2.number_of_nodes():,}")
    print(f"  edges: {G2.number_of_edges():,}")

    if G.number_of_nodes() == G2.number_of_nodes() and G.number_of_edges() == G2.number_of_edges():
        print("\nOK: cache round-trip preserves graph size.")
    else:
        print("\nWARNING: node or edge count differs after reload.")


if __name__ == "__main__":
    main()
