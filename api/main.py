from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model once
model = joblib.load("data/models/random_forest.joblib")


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction}
