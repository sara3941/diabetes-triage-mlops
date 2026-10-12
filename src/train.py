
import json
import os
import pickle
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)


def train(version="v0.1"):
    # 1. Load data
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    # 2. Deterministic train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # 3. Model selection based on version
    if version == "v0.1":
        # Baseline pipeline for v0.1
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        # Improved regularized model for v0.2
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))

    # 4. Train pipeline
    model.fit(X_train, y_train)

    # 5. Evaluate on held-out test split
    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # 6. Save model artifact and metrics
    os.makedirs("models", exist_ok=True)
    model_file = "models/model.bin"
    with open(model_file, "wb") as f_out:
        pickle.dump(model, f_out)

    metrics_file = "models/metrics.json"
    metrics = {
        "model_version": version,
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
    }
    with open(metrics_file, "w") as f_out:
        json.dump(metrics, f_out, indent=2)

    print(f"[{version}] Training finished successfully.")
    print(f"Saved model to: {model_file}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    version = os.getenv("MODEL_VERSION", "v0.1")
    train(version=version)