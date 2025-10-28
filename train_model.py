"""
train_model.py
Trains a RandomForestRegressor on the scikit-learn diabetes dataset,
prints evaluation metrics, and saves the trained model to models/diabetes_model.pkl
"""

import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/diabetes_model.pkl"

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names

def build_model(random_state: int = 42) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            )),
        ]
    )
    return model

def train_and_evaluate() -> dict:
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "model_path": MODEL_PATH}
    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if k != "model_path":
            print(f"{k}: {v:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature order: {feature_names}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
