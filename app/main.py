from fastapi import FastAPI
import joblib
import numpy as np
from .schemas import HeartDiseaseInput as HeartFeatures

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# Load model
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
def predict(features: HeartFeatures):
    data = np.array([[features.age, features.sex, features.cp, features.trestbps, features.chol,
                      features.fbs, features.restecg, features.thalach, features.exang,
                      features.oldpeak, features.slope, features.ca, features.thal]])
    
    prediction = model.predict(data)[0]
    return {"heart_disease": bool(prediction)}
