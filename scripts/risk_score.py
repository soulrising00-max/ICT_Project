import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import rasterio
from rasterio.enums import Resampling
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

# ── Config ────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
RESULTS_DIR = Path("data/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["ndvi_t1", "ndwi_t1", "ndvi_t2", "ndwi_t2", "delta_ndvi"]

# Verra claimed annual offsets in tCO2/year — fill from project pages
# https://registry.verra.org
CLAIMED_OFFSETS = {
    "1147_amazon_rio_redd": 400000,  # approximate — update with real value
    "562_KasigauCorridorREDD": 50000,  # approximate — update with real value
}

# Carbon density assumption: ~100 tCO2/ha over 20yr crediting period = 5/yr
CARBON_DENSITY_TCO2_PER_HA = 5.0

# HLS pixel resolution in metres (Landsat 30m)
PIXEL_SIZE_M = 30.0
PIXEL_AREA_HA = (PIXEL_SIZE_M**2) / 10000  # 0.09 ha per pixel


# ── Risk score ────────────────────────────────────────────────────────────
def compute_risk_score(forest_loss_ha, claimed_offset_tco2_yr):
    """
    Risk score = estimated lost sequestration / claimed annual offset.
    Numerator: forest_loss_ha * carbon_density (tCO2/ha/yr).
    Returns None if claimed offset is zero or missing.
    """
    if claimed_offset_tco2_yr is None or claimed_offset_tco2_yr <= 0:
        return None, "DATA_MISSING"

    lost_sequestration = forest_loss_ha * CARBON_DENSITY_TCO2_PER_HA
    risk_score = lost_sequestration / claimed_offset_tco2_yr
    risk_label = "HIGH" if risk_score > 0.05 else "LOW"
    return round(risk_score, 4), risk_label


# ── Run inference + compute loss ─────────────────────────────────────────
def analyze_project(project_id, model):
    print(f"\n{'='*50}\n{project_id}")

    path = FEATURES_DIR / f"{project_id}_labeled.parquet"
    df = pd.read_parquet(path)
    X = df[FEATURES].values

    # Model inference
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    df["pred"] = y_pred
    df["prob"] = y_prob

    # Forest loss in hectares
    loss_pixels = int(y_pred.sum())
    forest_loss_ha = round(loss_pixels * PIXEL_AREA_HA, 2)
    total_area_ha = round(len(df) * PIXEL_AREA_HA, 2)
    loss_pct = round((loss_pixels / len(df)) * 100, 3)

    print(f"  Total area:    {total_area_ha:,.1f} ha")
    print(f"  Loss pixels:   {loss_pixels:,}")
    print(f"  Forest loss:   {forest_loss_ha:,.2f} ha  ({loss_pct}%)")

    # Risk score
    claimed_offset = CLAIMED_OFFSETS.get(project_id)
    risk_score, risk_label = compute_risk_score(forest_loss_ha, claimed_offset)

    print(
        f"  Claimed offset: {claimed_offset:,} tCO2/yr"
        if claimed_offset
        else "  Claimed offset: MISSING"
    )
    print(f"  Risk score:    {risk_score}  →  {risk_label}")

    return {
        "project_id": project_id,
        "total_area_ha": total_area_ha,
        "loss_pixels": loss_pixels,
        "forest_loss_ha": forest_loss_ha,
        "loss_pct": loss_pct,
        "claimed_offset": claimed_offset,
        "carbon_density": CARBON_DENSITY_TCO2_PER_HA,
        "risk_score": risk_score,
        "risk_label": risk_label,
    }, df


# ── Forest loss map ───────────────────────────────────────────────────────
def plot_loss_map(project_id, df, result):
    """Reshape predictions back to raster and save PNG."""
    from pathlib import Path as P
    import numpy as np

    # Get raster shape from clipped B04
    b04_files = list((P("data/clipped") / project_id / "T1_2020").glob("*B04*.tif"))
    if not b04_files:
        print(f"  ⚠  No B04 found for loss map — skipping")
        return

    with rasterio.open(b04_files[0]) as src:
        h, w = src.height, src.width
        nodata = src.nodata
        raw = src.read(1).astype(float)
        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw[raw == -9999] = np.nan
        raw[raw > 10000] = np.nan
        raw[raw < -1000] = np.nan

    # Rebuild full pixel array with NaN for masked pixels
    valid_mask = ~np.isnan(raw.flatten())
    red_vals = raw.flatten()[valid_mask]
    nonzero_mask = red_vals != 0
    full_valid = valid_mask.copy()
    full_valid[valid_mask] = nonzero_mask

    pred_full = np.full(h * w, np.nan)
    pred_full[full_valid] = df["pred"].values[: full_valid.sum()]
    pred_map = pred_full.reshape(h, w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # NDVI T1
    ndvi_full = np.full(h * w, np.nan)
    ndvi_full[full_valid] = df["ndvi_t1"].values[: full_valid.sum()]
    axes[0].imshow(ndvi_full.reshape(h, w), cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[0].set_title(f"{project_id}\nNDVI 2020")
    axes[0].axis("off")

    # Predicted loss
    im = axes[1].imshow(pred_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title(
        f"Predicted forest loss\n"
        f"{result['forest_loss_ha']:,.1f} ha  |  risk: {result['risk_label']}"
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Loss probability")

    plt.tight_layout()
    out_path = REPORTS_DIR / f"{project_id}_loss_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading model...")
    model = joblib.load(MODELS_DIR / "random_forest.joblib")

    results = []
    dfs = {}

    for project_id in CLAIMED_OFFSETS:
        result, df = analyze_project(project_id, model)
        results.append(result)
        dfs[project_id] = df
        plot_loss_map(project_id, df, result)

    # Risk table
    risk_df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "risk_scores.csv"
    risk_df.to_csv(out_csv, index=False)

    print(f"\n{'='*50}")
    print("\nRisk score table:")
    print(
        risk_df[
            [
                "project_id",
                "forest_loss_ha",
                "claimed_offset",
                "risk_score",
                "risk_label",
            ]
        ].to_string(index=False)
    )
    print(f"\n✅ Saved: {out_csv}")

    # Log to MLflow
    mlflow.set_experiment("carbon-land-monitor")
    with mlflow.start_run(run_name="risk_scoring"):
        for r in results:
            prefix = r["project_id"][:20]
            mlflow.log_metric(f"{prefix}_forest_loss_ha", r["forest_loss_ha"])
            mlflow.log_metric(
                f"{prefix}_risk_score", r["risk_score"] if r["risk_score"] else -1
            )
        mlflow.log_artifact(str(out_csv))
        for project_id in CLAIMED_OFFSETS:
            map_path = REPORTS_DIR / f"{project_id}_loss_map.png"
            if map_path.exists():
                mlflow.log_artifact(str(map_path))

    print("\nNext: python scripts/api.py")


if __name__ == "__main__":
    main()
