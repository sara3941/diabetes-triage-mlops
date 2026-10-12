import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.train import train

VALID_PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}

PAYLOAD_MISSING_S6 = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
}

PAYLOAD_WRONG_TYPE = {
    **VALID_PAYLOAD,
    "bmi": "high",  # String instead of a numeric float
}


@pytest.fixture(scope="session", autouse=True)
def run_training():
    """Ensure a trained model artifact exists before executing tests."""
    train(version="v0.1")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_missing_feature_returns_422(client):
    response = client.post("/predict", json=PAYLOAD_MISSING_S6)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "details" in data
    field_errors = [item["field"] for item in data["details"]]
    assert "s6" in field_errors


def test_predict_invalid_data_type_returns_422(client):
    response = client.post("/predict", json=PAYLOAD_WRONG_TYPE)
    assert response.status_code == 422
    data = response.json()
    field_errors = [item["field"] for item in data["details"]]
    assert "bmi" in field_errors


def test_predict_empty_payload_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert len(data["details"]) == 10  # All 10 fields are missing
