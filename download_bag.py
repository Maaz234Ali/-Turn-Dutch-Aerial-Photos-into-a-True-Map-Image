import argparse, json, urllib.parse, urllib.request
import geopandas as gpd

BASE = "https://api.pdok.nl/kadaster/bag/ogc/v2/collections/pand/items"
CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"
DROP_STATUS = ("gesloopt", "niet gerealiseerd", "ten onrechte", "bouwvergunning")

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "true-ortho/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--margin", type=float, default=50.0)
    ap.add_argument("--out", default="data/bag_pand.gpkg")
    a = ap.parse_args()
    x0, y0, x1, y1 = a.bbox
    bbox = f"{x0-a.margin},{y0-a.margin},{x1+a.margin},{y1+a.margin}"
    q = urllib.parse.urlencode({"f": "json", "limit": 1000, "bbox": bbox,
                                "bbox-crs": CRS_URI, "crs": CRS_URI})
    url, feats = f"{BASE}?{q}", []
    while url:
        d = get(url)
        feats += d.get("features", [])
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
        print(len(feats), "features...")
    if not feats:
        raise SystemExit("No features returned.")
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:28992")
    
    print("\n--- Raw BAG Status Breakdown ---")
    if "status" in gdf.columns:
        print(gdf["status"].value_counts(dropna=False))
        s = gdf["status"].astype(str).str.lower()
        gdf = gdf[~s.apply(lambda v: any(k in v for k in DROP_STATUS))]
    
    print("\nTotal valid features:", len(gdf))
    print("Bounds in RD metres:", gdf.total_bounds)
    gdf.to_file(a.out, driver="GPKG")
    print("Saved", len(gdf), "->", a.out)

if __name__ == "__main__":
    main()
