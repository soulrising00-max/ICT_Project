# import numpy as np
# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask
# from shapely.geometry import mapping
# from pathlib import Path
# import pandas as pd

# # ── Config ───────────────────────────────────────────────────────────────
# POLYGONS_DIR = Path("data/polygons")
# CLIPPED_DIR = Path("data/clipped")
# FEATURES_DIR = Path("data/features")
# HANSEN_DIR = Path("data/hansen")

# HANSEN_DIR.mkdir(parents=True, exist_ok=True)

# PROJECTS = {
#     "1147_amazon_rio_redd": None,  # auto-calculate
#     "562_KasigauCorridorREDD": "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_00N_030E.tif",
# }
# # PROJECTS = ["1147_amazon_rio_redd", "562_KasigauCorridorREDD"]


# # ── Download Hansen tiles ─────────────────────────────────────────────────
# def get_hansen_url(lat, lon):
#     lat_tile = int(np.ceil(lat / 10)) * 10
#     # Fix: lon tile should snap to the tile that CONTAINS the point
#     lon_tile = int(np.floor(lon / 10)) * 10
#     # Hansen east tiles are labeled by their western edge rounded to nearest 10
#     # lon=38.771 → floor(38.771/10)*10 = 30 → but 30E tile only goes to 40E
#     # so we need ceil for eastern hemisphere positive longitudes
#     if lon >= 0:
#         lon_tile = int(np.ceil(lon / 10)) * 10
#         # edge case: if exactly on boundary
#         if lon_tile == lon:
#             lon_tile += 10

#     lat_str = f"{abs(lat_tile):02d}{'N' if lat_tile >= 0 else 'S'}"
#     lon_str = f"{abs(lon_tile):03d}{'E' if lon_tile >= 0 else 'W'}"

#     base = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
#     return f"{base}/Hansen_GFC-2023-v1.11_lossyear_{lat_str}_{lon_str}.tif"


# def download_hansen(project_id, gdf):
#     import requests

#     bounds = gdf.total_bounds  # minx, miny, maxx, maxy
#     # Use centroid of polygon to pick the tile
#     center_lat = (bounds[1] + bounds[3]) / 2
#     center_lon = (bounds[0] + bounds[2]) / 2

#     print(f"  Polygon centroid: lat={center_lat:.3f}, lon={center_lon:.3f}")

#     url = get_hansen_url(center_lat, center_lon)
#     out_path = HANSEN_DIR / f"{project_id}_lossyear.tif"

#     print(f"  Hansen tile URL: {url}")

#     # Delete cached file if it exists — may be wrong tile
#     if out_path.exists():
#         out_path.unlink()
#         print(f"  Deleted cached tile (may have been wrong tile)")

#     print(f"  ↓ Downloading Hansen tile...")
#     response = requests.get(url, stream=True)
#     if response.status_code != 200:
#         raise RuntimeError(f"Hansen download failed: {response.status_code}\n{url}")

#     with open(out_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     print(f"  ✅ Saved: {out_path.name}")
#     return out_path


# # ── Clip Hansen to polygon ────────────────────────────────────────────────
# def clip_hansen(hansen_path, gdf, project_id):
#     out_path = HANSEN_DIR / f"{project_id}_lossyear_clipped.tif"
#     if out_path.exists():
#         out_path.unlink()  # always reclip to avoid stale cache

#     with rasterio.open(hansen_path) as src:
#         print(f"  Hansen tile CRS: {src.crs}")
#         print(f"  Hansen tile bounds: {src.bounds}")

#         # Hansen is always EPSG:4326 — use gdf as-is, no reprojection
#         gdf_4326 = gdf.to_crs("EPSG:4326")
#         print(f"  Polygon bounds: {gdf_4326.total_bounds}")

#         shapes = [mapping(geom) for geom in gdf_4326.geometry]
#         clipped, transform = mask(src, shapes, crop=True)
#         meta = src.meta.copy()
#         meta.update(
#             {
#                 "height": clipped.shape[1],
#                 "width": clipped.shape[2],
#                 "transform": transform,
#             }
#         )

#     with rasterio.open(out_path, "w", **meta) as dst:
#         dst.write(clipped)
#     print(f"  ✂  Clipped Hansen: {out_path.name}")
#     return out_path


# # ── Add labels to feature matrix ─────────────────────────────────────────
# def add_labels(project_id, hansen_clipped_path):
#     from rasterio.enums import Resampling
#     from rasterio.warp import reproject

#     features_path = FEATURES_DIR / f"{project_id}_features.parquet"
#     df = pd.read_parquet(features_path)

#     # Get shape of the HLS clipped raster to match exactly
#     hls_ref = list((Path("data/clipped") / project_id / "T1_2020").glob("*B04*.tif"))[0]

#     with rasterio.open(hls_ref) as hls_src:
#         hls_shape = (hls_src.height, hls_src.width)
#         hls_crs = hls_src.crs
#         hls_transform = hls_src.transform

#     with rasterio.open(hansen_clipped_path) as hansen_src:
#         hansen_reprojected = np.zeros(hls_shape, dtype=np.float32)
#         reproject(
#             source=rasterio.band(hansen_src, 1),
#             destination=hansen_reprojected,
#             src_transform=hansen_src.transform,
#             src_crs=hansen_src.crs,
#             dst_transform=hls_transform,
#             dst_crs=hls_crs,
#             resampling=Resampling.nearest,
#         )

#     labels = (hansen_reprojected.flatten() > 0).astype(int)

#     # Apply same nodata mask as features
#     with rasterio.open(hls_ref) as src:
#         raw = src.read(1).astype(float)
#         nodata = src.nodata
#         if nodata is not None:
#             raw[raw == nodata] = np.nan
#         raw[raw == -9999] = np.nan
#         raw[raw > 10000] = np.nan
#         raw[raw < -1000] = np.nan

#     valid_mask = ~np.isnan(raw.flatten())
#     labels_filtered = labels[valid_mask]
#     red_vals = raw.flatten()[valid_mask]
#     nonzero_mask = red_vals != 0
#     labels_final = labels_filtered[nonzero_mask]

#     df["label"] = labels_final[: len(df)]
#     loss_rate = df["label"].mean()
#     label_source = "Hansen GFC lossyear"

#     print(
#         f"  Forest loss rate: {loss_rate:.2%}  ({df['label'].sum()} / {len(df)} pixels)"
#     )

#     # ── NDVI-change fallback for biomes where Hansen detects little loss ──
#     if loss_rate < 0.005:
#         print(
#             f"  ⚠  Hansen loss rate too low ({loss_rate:.2%}) — switching to NDVI-change labels"
#         )
#         ndvi_threshold = -0.15
#         df["label"] = (df["delta_ndvi"] < ndvi_threshold).astype(int)
#         loss_rate = df["label"].mean()
#         label_source = f"NDVI delta < {ndvi_threshold} (degradation proxy)"
#         print(
#             f"  NDVI-change loss rate: {loss_rate:.2%}  ({df['label'].sum()} / {len(df)} pixels)"
#         )
#         print(f"  ℹ  Label source: {label_source}")

#     if loss_rate == 0.0:
#         print("  ⚠  Still 0% after fallback — check data manually.")
#     if loss_rate > 0.5:
#         print("  ⚠  Loss rate >50% — verify labels.")

#     df["label_source"] = label_source

#     out_path = FEATURES_DIR / f"{project_id}_labeled.parquet"
#     df.to_parquet(out_path, index=False)
#     print(f"  ✅ Saved: {out_path.name}")
#     return df


# # ── Main ──────────────────────────────────────────────────────────────────
# def main():
#     for project_id, hansen_override in PROJECTS.items():
#         print(f"\n{'='*50}\n{project_id}")
#         gdf = gpd.read_file(POLYGONS_DIR / f"{project_id}.geojson").to_crs("EPSG:4326")

#         if hansen_override:
#             # download directly without URL calculation
#             import requests

#             out_path = HANSEN_DIR / f"{project_id}_lossyear.tif"
#             if out_path.exists():
#                 out_path.unlink()
#             print(f"  Using hardcoded tile: {hansen_override}")
#             response = requests.get(hansen_override, stream=True)
#             response.raise_for_status()
#             with open(out_path, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print(f"  ✅ Saved: {out_path.name}")
#             hansen_path = out_path
#         else:
#             hansen_path = download_hansen(project_id, gdf)

#         hansen_clipped = clip_hansen(hansen_path, gdf, project_id)
#         add_labels(project_id, hansen_clipped)


# if __name__ == "__main__":
#     main()
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import mapping
from pathlib import Path
import pandas as pd
import requests

# ── Config ───────────────────────────────────────────────────────────────
POLYGONS_DIR = Path("data/polygons")
CLIPPED_DIR = Path("data/clipped")
FEATURES_DIR = Path("data/features")
HANSEN_DIR = Path("data/hansen")

HANSEN_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = {
    "1147_amazon_rio_redd": None,
    "562_KasigauCorridorREDD": "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_00N_030E.tif",
}


# ── Hansen URL calculation ────────────────────────────────────────────────
def get_hansen_url(lat, lon):
    lat_tile = int(np.ceil(lat / 10)) * 10
    if lon >= 0:
        lon_tile = int(np.ceil(lon / 10)) * 10
        if lon_tile == lon:
            lon_tile += 10
    else:
        lon_tile = int(np.floor(lon / 10)) * 10

    lat_str = f"{abs(lat_tile):02d}{'N' if lat_tile >= 0 else 'S'}"
    lon_str = f"{abs(lon_tile):03d}{'E' if lon_tile >= 0 else 'W'}"
    base = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
    return f"{base}/Hansen_GFC-2023-v1.11_lossyear_{lat_str}_{lon_str}.tif"


# ── Download ──────────────────────────────────────────────────────────────
def download_hansen(project_id, gdf):
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    print(f"  Polygon centroid: lat={center_lat:.3f}, lon={center_lon:.3f}")

    url = get_hansen_url(center_lat, center_lon)
    out_path = HANSEN_DIR / f"{project_id}_lossyear.tif"
    print(f"  Hansen tile URL: {url}")

    if out_path.exists():
        out_path.unlink()

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Hansen download failed: {response.status_code}\n{url}")
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  ✅ Saved: {out_path.name}")
    return out_path


def download_hansen_hardcoded(project_id, url):
    out_path = HANSEN_DIR / f"{project_id}_lossyear.tif"
    if out_path.exists():
        out_path.unlink()
    print(f"  Using hardcoded tile: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  ✅ Saved: {out_path.name}")
    return out_path


# ── Clip Hansen to polygon ────────────────────────────────────────────────
def clip_hansen(hansen_path, gdf, project_id):
    out_path = HANSEN_DIR / f"{project_id}_lossyear_clipped.tif"
    if out_path.exists():
        out_path.unlink()

    gdf_4326 = gdf.to_crs("EPSG:4326")
    print(f"  Hansen tile bounds: {rasterio.open(hansen_path).bounds}")
    print(f"  Polygon bounds:     {gdf_4326.total_bounds}")

    with rasterio.open(hansen_path) as src:
        shapes = [mapping(geom) for geom in gdf_4326.geometry]
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
    print(f"  ✂  Clipped Hansen: {out_path.name}")
    return out_path


# ── Fmask cloud masking ───────────────────────────────────────────────────
def get_clear_mask(project_id, date, shape_2d):
    folder = CLIPPED_DIR / project_id / date
    fmask_files = list(folder.glob("*Fmask*.tif"))
    if not fmask_files:
        print(f"  ⚠  No Fmask for {date} — assuming all clear")
        return np.ones(shape_2d, dtype=bool)

    with rasterio.open(fmask_files[0]) as src:
        fmask = src.read(1)

    # Fmask values: 0=clear land, 1=clear water, 2+=cloud/shadow/snow
    clear = (fmask == 0) | (fmask == 1)
    print(f"  Fmask {date}: {clear.mean():.1%} clear pixels")
    return clear


# ── Add labels ────────────────────────────────────────────────────────────
def add_labels(project_id, hansen_clipped_path):
    features_path = FEATURES_DIR / f"{project_id}_features.parquet"
    df = pd.read_parquet(features_path)

    hls_ref = list((CLIPPED_DIR / project_id / "T1_2020").glob("*B04*.tif"))[0]

    with rasterio.open(hls_ref) as hls_src:
        hls_shape = (hls_src.height, hls_src.width)
        hls_crs = hls_src.crs
        hls_transform = hls_src.transform

    with rasterio.open(hansen_clipped_path) as hansen_src:
        hansen_reprojected = np.zeros(hls_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(hansen_src, 1),
            destination=hansen_reprojected,
            src_transform=hansen_src.transform,
            src_crs=hansen_src.crs,
            dst_transform=hls_transform,
            dst_crs=hls_crs,
            resampling=Resampling.nearest,
        )

    with rasterio.open(hls_ref) as src:
        raw = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw[raw == -9999] = np.nan
        raw[raw > 10000] = np.nan
        raw[raw < -1000] = np.nan

    valid_mask = ~np.isnan(raw.flatten())
    red_vals = raw.flatten()[valid_mask]
    nonzero_mask = red_vals != 0

    labels_flat = (hansen_reprojected.flatten() > 0).astype(int)
    labels_final = labels_flat[valid_mask][nonzero_mask]

    df["label"] = labels_final[: len(df)]
    loss_rate = df["label"].mean()
    label_source = "Hansen GFC lossyear"

    print(
        f"  Forest loss rate: {loss_rate:.2%}  ({df['label'].sum()} / {len(df)} pixels)"
    )

    # NDVI-change fallback — no cloud masking, just threshold
    if loss_rate < 0.005:
        print(
            f"  ⚠  Hansen loss rate too low — using NDVI-change fallback (no cloud mask)"
        )
        ndvi_threshold = -0.30
        df["label"] = (df["delta_ndvi"] < ndvi_threshold).astype(int)
        loss_rate = df["label"].mean()
        label_source = f"NDVI delta < {ndvi_threshold} degradation proxy"
        print(
            f"  NDVI-change loss rate: {loss_rate:.2%}  ({df['label'].sum()} / {len(df)} pixels)"
        )

    if loss_rate == 0.0:
        print("  ⚠  Still 0% — check data manually.")
    if loss_rate > 0.5:
        print(f"  ⚠  Loss rate >50% — threshold may be too loose for this biome.")

    df["label_source"] = label_source

    out_path = FEATURES_DIR / f"{project_id}_labeled.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  ✅ Saved: {out_path.name}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    for project_id, hansen_override in PROJECTS.items():
        print(f"\n{'='*50}\n{project_id}")
        gdf = gpd.read_file(POLYGONS_DIR / f"{project_id}.geojson").to_crs("EPSG:4326")

        if hansen_override:
            hansen_path = download_hansen_hardcoded(project_id, hansen_override)
        else:
            hansen_path = download_hansen(project_id, gdf)

        hansen_clipped = clip_hansen(hansen_path, gdf, project_id)
        add_labels(project_id, hansen_clipped)

    print(f"\n{'='*50}")
    print("Done. Check loss rates above before training.")
    print("Next: train_classifier.py")


if __name__ == "__main__":
    main()
