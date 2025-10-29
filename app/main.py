# Import Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import joblib
import numpy as np
import os

# Load model path from environment variable (used in Docker) or default path
MODEL_BUNDLE_PATH = os.getenv("MODEL_PATH", "models/diabetes_model.pkl")

# Global variables for the model and feature names
bundle = None
model = None
feature_names: List[str] = []

# Initialize FastAPI application
app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# Define request and response data models using Pydantic
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


# Define the API endpoints
@app.get("/")
def root():
    return {"status": "ok", "message": "Diabetes Prediction API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        # Convert list of Pydantic objects into a NumPy array
        rows = []
        for item in payload.data:
            # Extract values from each feature in correct order
            row = [getattr(item, fname) for fname in feature_names]
            rows.append(row)

        # Convert list of rows into NumPy array for model input
        X = np.array(rows, dtype=float)

        # Make predictions using the trained model
        preds = model.predict(X).tolist()

        # Return predictions and feature order in response
        return PredictResponse(predictions=preds, feature_order=feature_names)

    except ValidationError as ve:
        # Validation errors (e.g., missing or wrong type of data)
        raise HTTPException(status_code=422, detail=str(ve))

    except Exception as e:
        # Catch all other exceptions (e.g., model or input issues)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# Event triggered when FastAPI app starts
@app.on_event("startup")
def _load_model():

    global bundle, model, feature_names

    # Check that the model file exists
    if not os.path.exists(MODEL_BUNDLE_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_BUNDLE_PATH}. "
            "Train it first using: python train_model.py"
        )

    # Load model and feature names from saved bundle
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Log message (useful for Docker startup confirmation)
    print(f" Model loaded successfully from {MODEL_BUNDLE_PATH}")
