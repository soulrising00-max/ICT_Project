from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI()

# --- CORS (for Streamlit) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load ML model ---
model_path = Path("data/models/random_forest.joblib")
model = joblib.load(model_path)

# --- Load risk table (fallback/demo only) ---
risk_table_path = Path("data/results/risk_scores.csv")
df = pd.read_csv(risk_table_path)


# ==============================
# 🔹 Request Schema
# ==============================
class AnalyzeRequest(BaseModel):
    geojson: dict
    baseline_year: int
    analysis_year: int
    claimed_offset: float


# ==============================
# 🔹 Core Pipeline Functions
# ==============================


def compute_project_metrics(geojson, baseline_year, analysis_year):
    """
    TEMP IMPLEMENTATION

    Replace this with:
    - Raster clipping (Hansen / Sentinel)
    - Pixel counting
    """

    # --- Placeholder logic (using mean of dataset for now) ---
    total_area = df["total_area_ha"].mean()
    forest_loss = df["forest_loss_ha"].mean()

    loss_pct = forest_loss / total_area

    return {
        "total_area": float(total_area),
        "forest_loss": float(forest_loss),
        "loss_pct": float(loss_pct),
    }


def build_features(metrics, claimed_offset):
    """
    Converts computed metrics → model input
    """
    return [
        metrics["total_area"],
        metrics["forest_loss"],
        metrics["loss_pct"],
        claimed_offset,
    ]


def get_risk_tier(risk_score):
    """
    Simple interpretable risk bands
    """
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


# ==============================
# 🔹 Endpoints
# ==============================


# --- Health check ---
@app.get("/")
def home():
    return {"message": "API is running"}


# --- New: Full pipeline endpoint ---
@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    # 1. Compute metrics
    metrics = compute_project_metrics(
        request.geojson, request.baseline_year, request.analysis_year
    )

    # 2. Build features
    features = build_features(metrics, request.claimed_offset)

    # 3. Model prediction
    data = np.array(features).reshape(1, -1)
    risk_score = float(model.predict(data)[0])

    # 4. Risk classification
    risk_tier = get_risk_tier(risk_score)

    return {
        "total_area_ha": metrics["total_area"],
        "forest_loss_ha": metrics["forest_loss"],
        "loss_pct": metrics["loss_pct"],
        "risk_score": risk_score,
        "risk_tier": risk_tier,
    }


# --- Old endpoint (kept for demo/debug only) ---
@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction}


# --- Optional: static project lookup ---
@app.get("/projects/{project_id}/results")
def get_project_results(project_id: str):
    """
    NOTE:
    This returns precomputed values.
    Use /analyze for real pipeline.
    """
    row = df[df["project_id"] == project_id]
    if row.empty:
        return {"error": "Project not found"}
    return row.to_dict(orient="records")[0]
