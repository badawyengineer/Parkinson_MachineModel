import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequest


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


def test_valid_prediction_request():
    request = PredictionRequest(**valid_payload())

    assert request.mdvp_fo == 119.992
    assert len(request.to_model_features()) == 22


def test_missing_feature_is_rejected():
    payload = valid_payload()
    payload.pop("MDVP:Fo(Hz)")

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_string_number_is_converted():
    payload = valid_payload()
    payload["MDVP:Fo(Hz)"] = "119.992"

    request = PredictionRequest(**payload)

    assert request.mdvp_fo == 119.992
