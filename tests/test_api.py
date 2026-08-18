import pytest
from litestar.testing import TestClient

from app.main import app


def valid_payload():
    return {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654,
    }


@pytest.fixture
def client():
    with TestClient(app=app) as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "parkinson-ml-api"


def test_predict_valid_request(client):
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "label" in data


def test_prediction_value_is_valid(client):
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    data = response.json()

    assert data["prediction"] in [0, 1]


def test_probability_is_valid(client):
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    data = response.json()

    assert 0.0 <= data["probability"] <= 1.0


def test_prediction_label_is_valid(client):
    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    data = response.json()

    assert data["label"] in ["Parkinson", "Healthy"]


def test_missing_feature_is_rejected(client):
    payload = valid_payload()
    payload.pop("MDVP:Fo(Hz)")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code in [400, 422]


def test_invalid_feature_type_is_rejected(client):
    payload = valid_payload()
    payload["MDVP:Fo(Hz)"] = "not-a-number"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code in [400, 422]
