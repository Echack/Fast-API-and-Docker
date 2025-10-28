from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import joblib
import numpy as np
import os

MODEL_BUNDLE_PATH = os.getenv("MODEL_PATH", "models/diabetes_model.pkl")
bundle = None
model = None
feature_names: List[str] = []

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

class DiabetesFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

class PredictRequest(BaseModel):
    data: List[DiabetesFeatures]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: Optional[str] = "1.0.0"
    feature_order: List[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "Diabetes Prediction API is running."}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        rows = []
        for item in payload.data:
            row = [getattr(item, fname) for fname in feature_names]
            rows.append(row)
        X = np.array(rows, dtype=float)
        preds = model.predict(X).tolist()
        return PredictResponse(predictions=preds, feature_order=feature_names)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.on_event("startup")
def _load_model():
    global bundle, model, feature_names
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_BUNDLE_PATH}. Train it first: python train_model.py"
        )
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
