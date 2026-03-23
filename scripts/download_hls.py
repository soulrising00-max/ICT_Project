import requests
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import rasterio

# ── Config ──────────────────────────────────────────────────────────────
PROJECTS_DIR = Path("data/polygons")
HLS_DIR = Path("data/raw/hls")
HLS_DIR.mkdir(parents=True, exist_ok=True)

GEOJSON_FILES = sorted(PROJECTS_DIR.glob("*.geojson"))

TIME_POINTS = {
    "T1_2020": ("2020-01-01", "2020-04-30"),
    "T2_2023": ("2023-01-01", "2023-04-30"),
}

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
TARGET_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
DEBUG = True  # Set True to inspect granules before download


# ── Helpers ──────────────────────────────────────────────────────────────
def prepare_gdf(geojson_path):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def get_bbox(gdf):
    """Return bounding box: minx, miny, maxx, maxy"""
    return gdf.total_bounds


def search_hls_granules(bounds, start_date, end_date, product="HLSL30"):
    """
    Search CMR for HLS granules using a bounding box.
    Returns list of granules (possibly empty).
    """
    bbox_str = ",".join(map(str, bounds))
    params = {
        "short_name": product,
        "version": "2.0",
        "temporal": f"{start_date},{end_date}",
        "bounding_box": bbox_str,
        "page_size": 10,
        "sort_key": "-start_date",
    }

    try:
        r = requests.get(CMR_URL, params=params)
        r.raise_for_status()
        resp = r.json()
        granules = resp.get("feed", {}).get("entry", [])
    except Exception as e:
        print(f"  ❌ Error fetching granules for {product} ({bbox_str}): {e}")
        granules = []

    print(f"  [{product}] {len(granules)} granules found")
    if DEBUG and granules:
        print("    First granules IDs:")
        for g in granules[:5]:
            print("     -", g.get("producer_granule_id", "N/A"))
    return granules


def download_granule(granule, project_id, time_label):
    """Download all target bands from a granule"""
    out_dir = HLS_DIR / project_id / time_label
    out_dir.mkdir(parents=True, exist_ok=True)

    links = [
        l["href"]
        for l in granule.get("links", [])
        if l["href"].endswith(".tif")
        and any(band in l["href"] for band in TARGET_BANDS)
    ]

    if not links:
        print(f"  ⚠  No .tif links found for {granule.get('title', 'N/A')}")
        return

    for url in links:
        fname = url.split("/")[-1]
        out_path = out_dir / fname
        if out_path.exists():
            print(f"  ✓ Cached: {fname}")
            continue
        print(f"  ↓ Downloading: {fname}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"  ✅ Done: {project_id}/{time_label}")


def check_coverage(project_id, polygon):
    """Check coverage % of the polygon for each time point based on B04"""
    for time_label in TIME_POINTS:
        tifs = list((HLS_DIR / project_id / time_label).glob("*B04.tif"))
        if not tifs:
            print(
                f"  ⚠ {time_label}: No Red band (B04) found — skipping coverage check"
            )
            continue

        with rasterio.open(tifs[0]) as src:
            b = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
            tile_box = box(*b)
            pct = (tile_box.intersection(polygon).area / polygon.area) * 100
            flag = "✅" if pct >= 80 else "⚠ NEED ADJACENT TILE"
            print(f"  {time_label}: {pct:.1f}% coverage  {flag}")



def main():
    for geojson_path in GEOJSON_FILES:
        project_id = geojson_path.stem
        print(f"\n{'='*60}\nProject: {project_id}")

        gdf = prepare_gdf(geojson_path)
        polygon = gdf.geometry.union_all()
        bbox = get_bbox(gdf)

        for time_label, (start, end) in TIME_POINTS.items():
            print(f"\n  → {time_label}")

            granules = search_hls_granules(bbox, start, end, product="HLSL30")
            if not granules:
                print("    ↩ No Landsat results — trying Sentinel-2 (HLSS30)...")
                granules = search_hls_granules(bbox, start, end, product="HLSS30")

            if not granules:
                print(f"  ❌ No granules found for {time_label} — skipping")
                continue

            # Take most recent granule only
            download_granule(granules[0], project_id, time_label)

        print("\n  Coverage check:")
        check_coverage(project_id, polygon)

    print(f"\n{'='*60}\nDay 3 complete. Check any ⚠ flags above before Day 4.")


if __name__ == "__main__":
    main()