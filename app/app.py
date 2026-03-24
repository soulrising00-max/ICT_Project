import streamlit as st
import pandas as pd
from pathlib import Path
import requests
import json

API_URL = "http://localhost:8000/analyze"
# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Carbon Project Monitor", page_icon="🌿", layout="wide")
st.title("🌿 Carbon Project Land Cover Monitor")

# ── Load risk table ───────────────────────────────────────────
RISK_TABLE = Path("data/results/risk_scores.csv")  # ← your actual filename


@st.cache_data
def load_risk_table():
    if not RISK_TABLE.exists():
        st.error(f"Not found: {RISK_TABLE} — run risk_score.py first.")
        st.stop()
    df = pd.read_csv(RISK_TABLE)

    # Projects excluded due to unreliable model results
    # Add project_ids here as you validate more projects
    EXCLUDED_PROJECTS = [
        "562_KasigauCorridorREDD",  # zero loss — tile coverage unverified
    ]
    df = df[~df["project_id"].isin(EXCLUDED_PROJECTS)].reset_index(drop=True)

    # Human-readable display name
    df["project_name"] = df["project_id"].str.replace("_", " ", regex=False).str.title()
    return df


df = load_risk_table()

# ── Sidebar: project selector ─────────────────────────────────
st.sidebar.header("🔍 Filters")
selected_name = st.sidebar.selectbox(
    "Select Project", options=df["project_name"].tolist()
)

row = df[df["project_name"] == selected_name].iloc[0]
project_id = row["project_id"]

# ── Risk table (all projects) ─────────────────────────────────
st.subheader("📊 Risk Score Summary — All Projects")

RISK_THRESHOLD = 0.05  # 5% — matches your README assumption


def highlight_risk(r):
    if r["risk_label"] == "DATA_MISSING":
        return ["background-color: #fff3cd"] * len(r)
    if r["risk_score"] > RISK_THRESHOLD:
        return ["background-color: #ffe0e0"] * len(r)
    return [""] * len(r)


# Choose which columns to display and in what order
display_cols = [
    "project_name",
    "total_area_ha",
    "forest_loss_ha",
    "loss_pct",
    "claimed_offset",
    "risk_score",
    "risk_label",
]

# With this — no jinja2 needed
st.dataframe(
    df[display_cols],
    use_container_width=True,
    column_config={
        "project_name": st.column_config.TextColumn("Project"),
        "total_area_ha": st.column_config.NumberColumn(
            "Total Area (ha)", format="%.1f"
        ),
        "forest_loss_ha": st.column_config.NumberColumn(
            "Forest Loss (ha)", format="%.1f"
        ),
        "loss_pct": st.column_config.NumberColumn("Loss %", format="%.2f%%"),
        "claimed_offset": st.column_config.NumberColumn(
            "Claimed Offset (tCO₂/yr)", format="%d"
        ),
        "risk_score": st.column_config.NumberColumn("Risk Score", format="%.4f"),
        "risk_label": st.column_config.TextColumn("Risk"),
    },
)

# ── Selected project KPIs ─────────────────────────────────────
st.subheader(f"📍 {selected_name}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Area", f"{row['total_area_ha']:,.1f} ha")
c2.metric("Forest Loss", f"{row['forest_loss_ha']:,.1f} ha ({row['loss_pct']:.1f}%)")
c3.metric(
    "Risk Score",
    f"{row['risk_score']:.4f}",
    delta=(
        "Above threshold" if row["risk_score"] > RISK_THRESHOLD else "Within threshold"
    ),
    delta_color="inverse",
)
c4.metric("Claimed Annual Offset", f"{row['claimed_offset']:,} tCO₂/yr")

# ── Forest loss PNG ────────────────────────────────────────────
st.subheader("🗺️ Forest Loss Map (2020 → 2023)")

png_path = Path("data/reports") / f"{project_id}.png"

if png_path.exists():
    st.image(
        str(png_path),
        use_container_width=True,
        caption=f"{selected_name} — forest change overlay",
    )
else:
    st.warning(
        f"`data/reports/{project_id}.png` not found. "
        "Generate it from the evaluation notebook."
    )

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Risk threshold: {RISK_THRESHOLD:.0%}  •  "
    "Carbon density: 5 tCO₂/ha/yr (flat rate)  •  "
    "Median compositing: not applied (known limitation)  •  "
    "Run `mlflow ui` to view experiment logs."
)


# ── Live Analysis (NEW) ───────────────────────────────────────
st.subheader("⚡ Run Dynamic Analysis")

with st.expander("Run analysis using GeoJSON input"):

    uploaded_file = st.file_uploader("Upload GeoJSON", type=["geojson", "json"])

    baseline_year = st.number_input("Baseline Year", value=2020)
    analysis_year = st.number_input("Analysis Year", value=2023)
    claimed_offset = st.number_input(
        "Claimed Offset (tCO₂/yr)", value=int(row["claimed_offset"])
    )

    if st.button("Run Analysis"):

        if uploaded_file is None:
            st.warning("Please upload a GeoJSON file")
        else:
            try:
                geojson_data = json.load(uploaded_file)

                response = requests.post(
                    API_URL,
                    json={
                        "geojson": geojson_data,
                        "baseline_year": baseline_year,
                        "analysis_year": analysis_year,
                        "claimed_offset": claimed_offset,
                    },
                    timeout=30,
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    result = response.json()

                    st.success("Analysis complete")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("Total Area", f"{result['total_area_ha']:.1f} ha")
                    c2.metric("Forest Loss", f"{result['forest_loss_ha']:.1f} ha")
                    c3.metric("Loss %", f"{result['loss_pct']:.2%}")
                    c4.metric("Risk Score", f"{result['risk_score']:.4f}")

                    st.write(f"**Risk Tier:** {result['risk_tier']}")

            except json.JSONDecodeError:
                st.error("The uploaded file is not valid GeoJSON/JSON.")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
