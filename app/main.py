from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from .schemas import HeartDiseaseInput
import pathlib

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

templates = Jinja2Templates(directory="app/templates")

# Load the ML model
model = joblib.load("model/heart_model.joblib")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
    }


@app.post("/predict")
def predict(features: HeartDiseaseInput):
    data = np.array([[
        features.age, features.sex, features.cp, features.trestbps, features.chol,
        features.fbs, features.restecg, features.thalach, features.exang,
        features.oldpeak, features.slope, features.ca, features.thal
    ]])
    prediction = model.predict(data)[0]
    return {"heart_disease": bool(prediction)}


# -----------------------------
# FRONTEND ROUTES
# -----------------------------
# Serve the new merged HTML at "/"
@app.get("/", response_class=HTMLResponse)
def home():
    html_file = pathlib.Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"), status_code=200)
