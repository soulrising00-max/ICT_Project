import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from pathlib import Path
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────
POLYGONS_DIR = Path("data/polygons")
HLS_DIR = Path("data/raw/hls")
CLIPPED_DIR = Path("data/clipped")
FEATURES_DIR = Path("data/features")

CLIPPED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DATES = ["T1_2020", "T2_2023"]


# ── Clip ─────────────────────────────────────────────────────────────────
def clip_raster(raster_path, gdf, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        print(f"  ✓ Cached: {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        shapes = [mapping(geom) for geom in gdf_proj.geometry]
        clipped, transform = mask(src, shapes, crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
            }
        )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)
    print(f"  ✂  Clipped: {out_path.name}")


def clip_all():
    for geojson in sorted(POLYGONS_DIR.glob("*.geojson")):
        project_id = geojson.stem
        gdf = gpd.read_file(geojson).to_crs("EPSG:4326")
        print(f"\n{'='*50}\nClipping: {project_id}")
        for date in DATES:
            tifs = list((HLS_DIR / project_id / date).glob("*.tif"))
            if not tifs:
                print(f"  ⚠  No tiles found for {date} — skipping")
                continue
            for tif in tifs:
                out_path = CLIPPED_DIR / project_id / date / tif.name
                clip_raster(tif, gdf, out_path)


# ── Indices ───────────────────────────────────────────────────────────────
# def read_band(project_id, date, band_key):
#     """Find and read a band by keyword (e.g. B04, B05, B06)."""
#     folder = CLIPPED_DIR / project_id / date
#     matches = list(folder.glob(f"*{band_key}*.tif"))
#     if not matches:
#         raise FileNotFoundError(f"No {band_key} band found in {folder}")
#     with rasterio.open(matches[0]) as src:
#         data = src.read(1).astype(float)
#         nodata = src.nodata
#     if nodata is not None:
#         data[data == nodata] = np.nan
#     return data
def read_band(project_id, date, band_key):
    folder = CLIPPED_DIR / project_id / date
    matches = list(folder.glob(f"*{band_key}*.tif"))
    if not matches:
        raise FileNotFoundError(f"No {band_key} band found in {folder}")
    with rasterio.open(matches[0]) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    # HLS specific: mask fill value -9999 and unrealistic values
    data[data == -9999] = np.nan
    data[data > 10000] = np.nan  # HLS reflectance is scaled 0–10000
    data[data < -1000] = np.nan
    return data


def compute_indices(project_id, date):
    red = read_band(project_id, date, "B04")
    nir = read_band(project_id, date, "B05")
    swir = read_band(project_id, date, "B06")

    # HLS reflectance scaling — divide by 10000 to get 0–1 range
    red = red / 10000.0
    nir = nir / 10000.0
    swir = swir / 10000.0

    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (nir - swir) / (nir + swir + 1e-10)

    return ndvi, ndwi


# ── Feature matrix ────────────────────────────────────────────────────────
def build_features(project_id):
    print(f"\n  Building features: {project_id}")

    ndvi_t1, ndwi_t1 = compute_indices(project_id, "T1_2020")
    ndvi_t2, ndwi_t2 = compute_indices(project_id, "T2_2023")
    delta_ndvi = ndvi_t2 - ndvi_t1

    # flatten
    df = pd.DataFrame(
        {
            "ndvi_t1": ndvi_t1.flatten(),
            "ndwi_t1": ndwi_t1.flatten(),
            "ndvi_t2": ndvi_t2.flatten(),
            "ndwi_t2": ndwi_t2.flatten(),
            "delta_ndvi": delta_ndvi.flatten(),
        }
    )

    # drop nodata pixels
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    df = df[df["ndvi_t1"] != 0].reset_index(drop=True)
    print(f"  Pixels: {before} raw → {len(df)} valid")
    print(
        f"  NDVI T1 mean: {df['ndvi_t1'].mean():.3f}  T2 mean: {df['ndvi_t2'].mean():.3f}"
    )

    out_path = FEATURES_DIR / f"{project_id}_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  ✅ Saved: {out_path.name}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Step 1 — Clipping")
    clip_all()

    print("\nStep 2 — Feature extraction")
    for geojson in sorted(POLYGONS_DIR.glob("*.geojson")):
        build_features(geojson.stem)

    print(f"\n{'='*50}")
    print("Done. Check data/features/ for .parquet files.")
    print("Next step: add Hansen labels and train the classifier.")


if __name__ == "__main__":
    main()
