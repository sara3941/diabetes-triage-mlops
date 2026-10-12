import os
import pickle
import pandas as pd
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.schemas import DiabetesInput

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.bin")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

app = FastAPI(title="Diabetes Progression Triage API")

# Global reference for model
model = None


def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f_in:
            model = pickle.load(f_in)


# Load at startup
load_model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [
        {
            "field": ".".join([str(loc) for loc in err["loc"] if loc != "body"]),
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error: missing or invalid input features",
            "details": errors,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(payload: DiabetesInput):

    if model is None:
        load_model()
        if model is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error": "Model artifact not loaded"},
            )

    # Convert incoming payload to a single-row DataFrame matching training columns
    data = pd.DataFrame([payload.model_dump()])
    prediction_val = float(model.predict(data)[0])
    return {"prediction": prediction_val}
