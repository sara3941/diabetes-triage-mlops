# Diabetes Progression Triage MLOps Service

A containerized machine learning microservice for prioritized triage in virtual diabetes care.

---

## Architecture & Tech Stack

- **Framework**: FastAPI + Uvicorn (Python 3.11)
- **Model Pipeline**: scikit-learn (`StandardScaler` + `LinearRegression` for v0.1)
- **Input Validation**: Pydantic request models enforcing strict schema validation
- **Port**: 80
- **CI/CD**: GitHub Actions
  - `ci.yml`: Automated code quality (`black`, `flake8`), model smoke test, and test suite execution (`pytest`)
  - `release.yml`: Automated packaging, container smoke test, GitHub Releases, and image publication to GitHub Container Registry (GHCR)

---

## API Endpoints

### 1. `GET /health`
Returns service health status and the active model version.

- **Request**:
  curl http://localhost:80/health

- **Response (HTTP 200)**:
  {
    "status": "ok",
    "model_version": "v0.1"
  }

---

### 2. `POST /predict`
Scores short-term diabetes progression using 10 standardized clinical markers (`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`).

- **Request**:
  curl -X POST http://localhost:80/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'

- **Response (HTTP 200)**:
  {
    "prediction": 235.9496
  }

---

### 3. Error Handling (HTTP 422)

The service enforces strict validation on incoming clinical features. Malformed or incomplete requests return HTTP 422:

- **Missing Required Feature (e.g., omitting `s6`)**:
  curl -i -X POST http://localhost:80/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'

  *Response (HTTP 422)*:
  {
    "error": "Validation error: missing or invalid input features",
    "details": [
      {
        "field": "s6",
        "message": "Field required"
      }
    ]
  }

- **Invalid Data Type (e.g., `"bmi": "high"`)**: Rejects non-numeric values with validation error details.
- **Empty Payload (`{}`)**: Rejects missing fields with details for all 10 clinical features.

---

## Running the Container

### Pull and Run from GHCR
docker pull ghcr.io/sara3941/diabetes-triage-mlops:v0.1
docker run -d --name triage-service -p 80:80 ghcr.io/sara3941/diabetes-triage-mlops:v0.1

### Local Build and Execution
docker build -t diabetes-triage:v0.1 .
docker run -d --name triage-service -p 80:80 -e MODEL_VERSION=v0.1 diabetes-triage:v0.1

### Run Local Test Suite
pytest tests/ -v