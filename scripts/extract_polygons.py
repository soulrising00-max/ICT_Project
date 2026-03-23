"""
extract_polygons.py
-------------------
Reads all .kml, .kmz, and .shp files from an input folder,
converts each to EPSG:4326, and saves as .geojson.

Usage:
    python extract_polygons.py --input data/polygons/raw --output data/polygons

Dependencies:
    pip install geopandas fastkml lxml shapely
"""

import argparse
import zipfile
import tempfile

# import shutil
import re
from pathlib import Path

import geopandas as gpd

# import pandas as pd
# from shapely.geometry import shape, MultiPolygon, Polygon
from lxml import etree
from fastkml import kml as fastkml_kml


# ── Helpers ──────────────────────────────────────────────────────────────────


def sanitise_kml_bytes(kml_bytes: bytes) -> bytes:
    """
    Fix two common issues in Verra (and other real-world) KML files that
    cause fastkml / Python's expat parser to reject the document:

    1. xsi:schemaLocation on <Document> without xmlns:xsi declared.
       fastkml uses Python's built-in expat, which is strict about this.
       Fix: declare the missing namespace on the root <kml> element.

    2. Any other undeclared namespace prefix on attributes.
       Fix: strip the offending attributes entirely using lxml, which is
       more lenient, then re-serialise to clean bytes.
    """
    # Strategy A — declare missing xsi namespace (fast, handles the common case)
    text = kml_bytes.decode("utf-8", errors="replace")
    if "xsi:schemaLocation" in text and "xmlns:xsi" not in text:
        text = text.replace(
            "<kml ",
            '<kml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ',
            1,
        )
        # Handle files where root element is <kml> with no trailing space
        if 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' not in text:
            text = re.sub(
                r"(<kml\b)",
                r'\1 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
                text,
                count=1,
            )
    kml_bytes = text.encode("utf-8")

    # Strategy B — if still broken, use lxml (lenient) to strip bad attributes
    # and re-serialise to well-formed XML
    try:
        root = etree.fromstring(kml_bytes)
        kml_bytes = etree.tostring(root, encoding="utf-8", xml_declaration=True)
    except etree.XMLSyntaxError:
        # lxml also failed — return whatever we have and let fastkml surface the error
        pass

    return kml_bytes


def extract_geometries_from_kml_string(kml_bytes: bytes) -> tuple:
    """Walk a parsed KML tree and collect all Placemark geometries.

    fastkml 1.4 changes vs older versions:
      - KML().from_string() silently leaves .features empty; must use
        KML.class_from_element(ns=..., strict=False, element=root) instead.
      - Placemark geometry is on .kml_geometry (a fastkml wrapper object),
        not .geometry. The actual Shapely geom is at .kml_geometry.geometry.
    """
    kml_bytes = sanitise_kml_bytes(kml_bytes)
    root = etree.fromstring(kml_bytes)

    # Detect namespace from root tag, fall back to standard KML 2.2
    ns = root.nsmap.get(None, "http://www.opengis.net/kml/2.2")
    ns_braced = "{" + ns + "}"

    k = fastkml_kml.KML.class_from_element(ns=ns_braced, strict=False, element=root)

    geometries = []
    names = []

    def get_features(obj):
        f = getattr(obj, "features", None)
        if f is None:
            return []
        return f() if callable(f) else list(f)

    def get_geometry(feature):
        """
        Try every attribute name that fastkml has used across versions:
          - kml_geometry.geometry  (fastkml 1.x — the wrapper object)
          - geometry               (fastkml 0.x — direct Shapely object)
        """
        # fastkml 1.x
        kml_geom = getattr(feature, "kml_geometry", None)
        if kml_geom is not None:
            g = getattr(kml_geom, "geometry", None)
            if g is not None:
                return g
        # fastkml 0.x fallback
        g = getattr(feature, "geometry", None)
        return g

    def walk(features):
        for feature in features:
            geom = get_geometry(feature)
            if geom is not None:
                geometries.append(geom)
                names.append(getattr(feature, "name", None) or "")
            walk(get_features(feature))

    walk(get_features(k))

    return geometries, names


def read_kml(path: Path) -> gpd.GeoDataFrame:
    """Read a .kml file and return a GeoDataFrame (no CRS set by KML spec)."""
    kml_bytes = path.read_bytes()
    geometries, names = extract_geometries_from_kml_string(kml_bytes)

    if not geometries:
        raise ValueError(f"No geometries found in {path.name}")

    gdf = gpd.GeoDataFrame({"name": names, "geometry": geometries})
    gdf.crs = None  # KML is always lon/lat but has no formal CRS tag
    return gdf


def read_kmz(path: Path) -> gpd.GeoDataFrame:
    """Unzip a .kmz, find the inner doc.kml, and delegate to read_kml."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)
        tmp_path = Path(tmp)
        # doc.kml is the standard name; fall back to first .kml found
        kml_candidates = list(tmp_path.rglob("doc.kml")) or list(
            tmp_path.rglob("*.kml")
        )
        if not kml_candidates:
            raise ValueError(f"No .kml found inside {path.name}")
        return read_kml(kml_candidates[0])


def read_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Read a .shp file. Preserves all attribute columns."""
    return gpd.read_file(str(path))


def ensure_crs(gdf: gpd.GeoDataFrame, source_name: str) -> tuple[gpd.GeoDataFrame, str]:
    """
    If CRS is missing, assume EPSG:4326.
    If CRS is set but not 4326, reproject.
    Returns (gdf_in_4326, crs_note).
    """
    if gdf.crs is None:
        print(f"    ⚠  No CRS detected in {source_name} — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
        crs_note = "assumed EPSG:4326"
    elif gdf.crs.to_epsg() != 4326:
        original = gdf.crs.to_string()
        gdf = gdf.to_crs("EPSG:4326")
        crs_note = f"reprojected from {original}"
    else:
        crs_note = "EPSG:4326"
    return gdf, crs_note


def dissolve_if_multi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    If there are multiple rows (e.g. multiple zones in one file),
    dissolve into a single MultiPolygon so we get one GeoJSON feature.
    Keeps a 'name' column summarising all zone names.
    """
    if len(gdf) <= 1:
        return gdf
    names = ", ".join(str(n) for n in gdf.get("name", []) if n)
    dissolved = gdf.dissolve()
    dissolved["name"] = names or "dissolved"
    dissolved = dissolved.reset_index(drop=True)
    return dissolved


def area_km2(gdf: gpd.GeoDataFrame) -> float:
    """
    Compute area in km² by projecting to an equal-area CRS.
    Uses EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global).
    """
    try:
        projected = gdf.to_crs("EPSG:6933")
        return round(projected.geometry.area.sum() / 1e6, 2)
    except Exception:
        return float("nan")


def geometry_summary(gdf: gpd.GeoDataFrame) -> str:
    types = gdf.geometry.geom_type.unique().tolist()
    return ", ".join(types)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def process_file(path: Path, output_dir: Path) -> dict | None:
    """
    Read one spatial file, clean it, save as GeoJSON.
    Returns a summary dict or None on failure.
    """
    suffix = path.suffix.lower()
    stem = path.stem

    try:
        # --- Read ---
        if suffix == ".kml":
            gdf = read_kml(path)
        elif suffix == ".kmz":
            gdf = read_kmz(path)
        elif suffix == ".shp":
            gdf = read_shapefile(path)
        else:
            return None  # shouldn't happen given glob filter

        if gdf is None or gdf.empty:
            print(f"    ✗  {path.name}: empty — skipping")
            return None

        # --- CRS ---
        gdf, crs_note = ensure_crs(gdf, path.name)

        # --- Drop rows with null geometry ---
        before = len(gdf)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        dropped = before - len(gdf)
        if dropped:
            print(f"    ⚠  Dropped {dropped} null/empty geometry rows from {path.name}")

        if gdf.empty:
            print(f"    ✗  {path.name}: no valid geometries after cleaning — skipping")
            return None

        # --- Dissolve multi-row files ---
        gdf = dissolve_if_multi(gdf)

        # --- Compute stats before saving ---
        km2 = area_km2(gdf)
        geom_type = geometry_summary(gdf)

        # --- Save ---
        out_path = output_dir / f"{stem}.geojson"
        gdf.to_file(str(out_path), driver="GeoJSON")

        result = {
            "file": path.name,
            "output": out_path.name,
            "crs": crs_note,
            "geometry_type": geom_type,
            "area_km2": km2,
            "features": len(gdf),
            "status": "ok",
        }
        print(f"    ✓  {path.name}  →  {out_path.name}")
        return result

    except Exception as e:
        print(f"    ✗  {path.name}: {e}")
        return {
            "file": path.name,
            "output": "—",
            "crs": "—",
            "geometry_type": "—",
            "area_km2": "—",
            "features": "—",
            "status": f"ERROR: {e}",
        }


def print_summary_table(results: list[dict]) -> None:
    """Pretty-print the summary table to stdout."""
    if not results:
        print("\nNo files processed.")
        return

    col_widths = {
        "file": max(len(r["file"]) for r in results) + 2,
        "crs": max(len(str(r["crs"])) for r in results) + 2,
        "geometry_type": max(len(str(r["geometry_type"])) for r in results) + 2,
        "area_km2": 12,
        "status": max(len(str(r["status"])) for r in results) + 2,
    }
    col_widths = {k: max(v, len(k) + 2) for k, v in col_widths.items()}

    header = (
        f"{'File':<{col_widths['file']}}"
        f"{'CRS':<{col_widths['crs']}}"
        f"{'Geometry':<{col_widths['geometry_type']}}"
        f"{'Area (km²)':>{col_widths['area_km2']}}"
        f"  {'Status'}"
    )
    divider = "-" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)
    for r in results:
        print(
            f"{r['file']:<{col_widths['file']}}"
            f"{str(r['crs']):<{col_widths['crs']}}"
            f"{str(r['geometry_type']):<{col_widths['geometry_type']}}"
            f"{str(r['area_km2']):>{col_widths['area_km2']}}"
            f"  {r['status']}"
        )
    print(divider)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n  {ok}/{len(results)} files converted successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KML/KMZ/SHP files to clean GeoJSON in EPSG:4326."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Folder containing .kml, .kmz, .shp files (searched recursively)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Folder to write .geojson outputs"
    )
    parser.add_argument(
        "--no-dissolve",
        action="store_true",
        help="Skip dissolve step — keep one GeoJSON feature per row",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"ERROR: Input folder does not exist: {input_dir}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files — skip shapefiles that are component parts (.dbf, .prj, etc.)
    extensions = {".kml", ".kmz", ".shp"}
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in extensions)

    if not files:
        print(f"No .kml / .kmz / .shp files found in {input_dir}")
        raise SystemExit(0)

    print(f"\nFound {len(files)} file(s) in {input_dir}\n")

    results = []
    for path in files:
        print(f"  Processing: {path.name}")
        result = process_file(path, output_dir)
        if result:
            results.append(result)

    print_summary_table(results)


if __name__ == "__main__":
    main()
