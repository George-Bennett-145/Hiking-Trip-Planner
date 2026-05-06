import requests
import json
from collections import Counter

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

QUERY = """
[out:json][timeout:90];
area["name"="Lake District National Park"]->.searchArea;
(
  node["tourism"="hotel"](area.searchArea);
  node["tourism"="guest_house"](area.searchArea);
  node["tourism"="hostel"](area.searchArea);
  node["tourism"="camp_site"](area.searchArea);
  node["tourism"="inn"](area.searchArea);
  node["tourism"="apartment"](area.searchArea);
  node["tourism"="caravan_site"](area.searchArea);
  way["tourism"="hotel"](area.searchArea);
  way["tourism"="guest_house"](area.searchArea);
  way["tourism"="hostel"](area.searchArea);
  way["tourism"="camp_site"](area.searchArea);
  way["tourism"="inn"](area.searchArea);
  way["tourism"="apartment"](area.searchArea);
  way["tourism"="caravan_site"](area.searchArea);
);
out body center;
"""

HEADERS = {"User-Agent": "UKHikingTripPlanner/0.1 (capstone project)"}


def fetch_accommodation():
    print("Querying Overpass API...")
    response = requests.post(OVERPASS_URL, data={"data": QUERY}, headers=HEADERS)

    if response.status_code != 200:
        print(f"Error: received status code {response.status_code}")
        print(response.text)
        return None

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: response was not valid JSON")
        print(response.text[:500])
        return None

    print(f"Raw results: {len(data['elements'])} elements returned")
    return data


def flatten(data):
    flattened = []

    for place in data["elements"]:
        tags = place.get("tags", {})

        # Nodes have lat/lon directly; ways have them under "center"
        lat = place.get("lat") or place.get("center", {}).get("lat")
        lon = place.get("lon") or place.get("center", {}).get("lon")

        flattened.append({
            "id": place.get("id"),
            "type": place.get("type"),
            "tourism": tags.get("tourism"),
            "name": tags.get("name", "Unnamed"),
            "lat": lat,
            "lon": lon,
            "amenity": tags.get("amenity"),
            "brand": tags.get("brand"),
            "brand_wikidata": tags.get("brand:wikidata"),
            "fhrs_id": tags.get("fhrs:id"),
            "addr_housename": tags.get("addr:housename"),
            "addr_street": tags.get("addr:street"),
            "addr_city": tags.get("addr:city"),
            "addr_postcode": tags.get("addr:postcode"),
            "addr_country": tags.get("addr:country"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website"),
        })

    return flattened


def save(flattened, output_path="output/accommodation.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flattened, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(flattened)} places to {output_path}")


if __name__ == "__main__":
    data = fetch_accommodation()

    if data:
        flattened = flatten(data)
        save(flattened)

        # Print a summary breakdown by tourism type
        types = Counter(p["tourism"] for p in flattened)
        print("\nBreakdown by type:")
        for tourism_type, count in types.most_common():
            print(f"  {tourism_type}: {count}")
