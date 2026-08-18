from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "parkinson_model.joblib"
)


SAMPLE = {
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
    "spread2": -0.137177,
    "D2": 3.413649,
    "PPE": 0.284654,
}


def load_model():
    return joblib.load(MODEL_PATH)


def load_sample():
    return pd.DataFrame([SAMPLE])


def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = load_model()

    assert model is not None


def test_model_prediction():
    model = load_model()
    sample = load_sample()

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_model_probability():
    model = load_model()
    sample = load_sample()

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)

    probability = probabilities[0][1]

    assert 0.0 <= probability <= 1.0
