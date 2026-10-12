import json
import os
import pickle
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


def train(version="v0.2"):
    mlflow.set_tracking_uri("sqlite:////app/mlflow.db")
    mlflow.set_experiment("diabetes_triage_v0.1")

    # Load dataset
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    # Fixed reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if version == "v0.1":
        model_name = "LinearRegression"
        regressor = LinearRegression()
        run_name = "v0.1-baseline"
        extra_params = {}
    else:
        model_name = "RidgeCV"
        alphas = np.logspace(-3, 3, 50)
        regressor = RidgeCV(alphas=alphas, cv=5)
        run_name = "v0.2-ridge"
        extra_params = {"cv_folds": 5}

    with mlflow.start_run(run_name=run_name):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", regressor),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        train_r2 = float(r2_score(y_train, y_pred_train))
        test_r2 = float(r2_score(y_test, y_pred_test))

        mlflow.log_param("model_version", version)
        mlflow.log_param("model_type", model_name)
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

        if version != "v0.1":
            best_alpha = float(pipeline.named_steps["regressor"].alpha_)
            mlflow.log_param("selected_best_alpha", best_alpha)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.sklearn.log_model(pipeline, name="model")

        os.makedirs("models", exist_ok=True)
        with open("models/model.bin", "wb") as f:
            pickle.dump(pipeline, f)

        metrics = {
            "model_version": version,
            "model_type": model_name,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }
        if version != "v0.1":
            metrics["selected_best_alpha"] = float(
                pipeline.named_steps["regressor"].alpha_
            )

        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return pipeline, metrics


def main():
    version = os.getenv("MODEL_VERSION", "v0.2")
    pipeline, metrics = train(version=version)
    print(
        f"[{metrics['model_version']}] Training Complete | "
        f"Model: {metrics['model_type']} | "
        f"Test RMSE: {metrics['test_rmse']:.2f} | "
        f"Test R2: {metrics['test_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
