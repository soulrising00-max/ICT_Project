import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import joblib

# ── Config ────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PROJECT = "1147_amazon_rio_redd"
TEST_PROJECT = "562_KasigauCorridorREDD"
FEATURES = ["ndvi_t1", "ndwi_t1", "ndvi_t2", "ndwi_t2", "delta_ndvi"]


# ── Load data ─────────────────────────────────────────────────────────────
def load_data(project_id):
    path = FEATURES_DIR / f"{project_id}_labeled.parquet"
    df = pd.read_parquet(path)
    X = df[FEATURES].values
    y = df["label"].values
    print(f"  {project_id}: {len(df)} pixels, loss rate {y.mean():.2%}")
    return X, y


# ── NDVI baseline ─────────────────────────────────────────────────────────
def ndvi_baseline_predict(X, threshold=0.3):
    ndvi_t2_idx = FEATURES.index("ndvi_t2")
    return (X[:, ndvi_t2_idx] < threshold).astype(int)


# ── Plot confusion matrix ─────────────────────────────────────────────────
def plot_confusion_matrix(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No loss", "Loss"])
    ax.set_yticklabels(["No loss", "Loss"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    mlflow.set_experiment("carbon-land-monitor")

    print("Loading training data...")
    X_train_full, y_train_full = load_data(TRAIN_PROJECT)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42,
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    print("\nLoading test data (Kasigau — out of sample)...")
    X_test, y_test = load_data(TEST_PROJECT)

    with mlflow.start_run(run_name="random_forest_v1"):

        # ── Train ─────────────────────────────────────────────────────────
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("train_project", TRAIN_PROJECT)
        mlflow.log_param("test_project", TEST_PROJECT)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        print("\nTraining random forest...")
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # ── Validation ────────────────────────────────────────────────────
        print("\n── Validation (Amazon held-out 20%) ──")
        y_val_pred = rf.predict(X_val)
        y_val_prob = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        val_cm = confusion_matrix(y_val, y_val_pred)

        print(
            classification_report(y_val, y_val_pred, target_names=["No loss", "Loss"])
        )
        print(f"  AUC: {val_auc:.3f}")

        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)

        plot_confusion_matrix(
            val_cm, "Validation — Amazon", REPORTS_DIR / "confusion_val.png"
        )
        mlflow.log_artifact(str(REPORTS_DIR / "confusion_val.png"))

        # ── NDVI baseline comparison ──────────────────────────────────────
        print("\n── NDVI baseline vs Random Forest (validation) ──")
        y_ndvi = ndvi_baseline_predict(X_val)
        ndvi_precision = precision_score(y_val, y_ndvi, zero_division=0)
        ndvi_recall = recall_score(y_val, y_ndvi, zero_division=0)
        ndvi_f1 = f1_score(y_val, y_ndvi, zero_division=0)

        print(
            classification_report(
                y_val, y_ndvi, target_names=["No loss", "Loss"], zero_division=0
            )
        )
        print(
            f"  RF   — precision: {val_precision:.3f}  recall: {val_recall:.3f}  f1: {val_f1:.3f}"
        )
        print(
            f"  NDVI — precision: {ndvi_precision:.3f}  recall: {ndvi_recall:.3f}  f1: {ndvi_f1:.3f}"
        )

        mlflow.log_metric("ndvi_baseline_precision", ndvi_precision)
        mlflow.log_metric("ndvi_baseline_recall", ndvi_recall)
        mlflow.log_metric("ndvi_baseline_f1", ndvi_f1)

        # ── Feature importances ───────────────────────────────────────────
        importances = dict(zip(FEATURES, rf.feature_importances_))
        print("\n── Feature importances ──")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.3f}")
            mlflow.log_metric(f"importance_{feat}", imp)

        fig, ax = plt.subplots(figsize=(6, 4))
        sorted_feats = sorted(importances.items(), key=lambda x: x[1])
        ax.barh(
            [f for f, _ in sorted_feats], [v for _, v in sorted_feats], color="#1d9e75"
        )
        ax.set_xlabel("Importance")
        ax.set_title("Feature importances — random forest")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(REPORTS_DIR / "feature_importance.png"))

        # ── Out-of-sample test on Kasigau ─────────────────────────────────
        print("\n── Out-of-sample test (Kasigau, Kenya) ──")
        print("  Note: Kasigau Hansen labels are sparse (6 loss pixels)")
        print("  Result is indicative only — image quality is a known limitation")
        y_test_pred = rf.predict(X_test)
        test_cm = confusion_matrix(y_test, y_test_pred)

        print(
            classification_report(
                y_test, y_test_pred, target_names=["No loss", "Loss"], zero_division=0
            )
        )

        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        plot_confusion_matrix(
            test_cm, "Out-of-sample — Kasigau", REPORTS_DIR / "confusion_test.png"
        )
        mlflow.log_artifact(str(REPORTS_DIR / "confusion_test.png"))

        # ── Save model ────────────────────────────────────────────────────
        model_path = MODELS_DIR / "random_forest.joblib"
        joblib.dump(rf, model_path)
        mlflow.sklearn.log_model(rf, "random_forest_model")
        print(f"\n  ✅ Model saved: {model_path}")

    print(f"\n{'='*50}")
    print("Training complete. Check data/reports/ for plots.")
    print("Next: python scripts/risk_score.py")
    print("MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
