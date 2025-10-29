import os
from typing import Tuple

# Libraries for saving models and performing machine learning tasks
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Path to save the trained model
MODEL_PATH = "models/diabetes_model.pkl"

def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads the diabetes dataset from scikit-learn.
    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        feature_names (list): List of feature names
    """
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    return X, y, feature_names


def build_model(random_state: int = 42) -> Pipeline:

    model = Pipeline(
        steps=[
            # StandardScaler ensures all features are on the same scale
            ("scaler", StandardScaler(with_mean=True, with_std=True)),

            # RandomForestRegressor uses multiple decision trees for regression
            ("rf", RandomForestRegressor(
                n_estimators=300,   # number of trees
                max_depth=None,     # allows trees to grow fully
                random_state=random_state,
                n_jobs=-1           # use all CPU cores for faster training
            )),
        ]
    )
    return model


def train_and_evaluate() -> dict:
 
    # Load dataset and feature names
    X, y, feature_names = load_data()

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Evaluate model performance using common regression metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Create a 'models' directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save both model and feature names as a dictionary for easy loading later
    joblib.dump({"model": model, "feature_names": feature_names}, MODEL_PATH)

    # Print and return model performance metrics
    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse, "model_path": MODEL_PATH}
    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        if k != "model_path":
            print(f"{k}: {v:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Feature order: {feature_names}")
    return metrics


# This ensures that the training only runs when this script is executed directly,
# not when it’s imported as a module in another file.
if __name__ == "__main__":
    train_and_evaluate()
    
