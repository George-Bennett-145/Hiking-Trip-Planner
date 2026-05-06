"""Step 1: download and cache the Lake District walking network.

One-off setup script. Pulls the OSM walking network (footpaths,
bridleways, roads usable on foot) for the Lake District bounding box,
saves it as GraphML to output/, then reloads it to confirm the cache
works.

The bbox matches the one used for accommodation in CLAUDE.md:
    south 54.35, west -3.45, north 54.80, east -2.75

Run from the project root:
    python build_walking_graph.py
"""

import time
from pathlib import Path

import osmnx as ox

BBOX_WEST = -3.45
BBOX_SOUTH = 54.35
BBOX_EAST = -2.75
BBOX_NORTH = 54.80

OUTPUT_PATH = Path("output/lake_district_walking_graph.graphml")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading walking network for Lake District bbox...")
    print(f"  bbox (W,S,E,N) = ({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH})")
    t0 = time.perf_counter()
    G = ox.graph_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        network_type="walk",
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
