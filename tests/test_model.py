from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "parkinson_model.joblib"
)

DATA_PATH = Path(
    "/home/badawy/.cache/kagglehub/datasets/"
    "debasisdotcom/parkinson-disease-detection/"
    "versions/2/Parkinsson disease.csv"
)


def load_model():
    return joblib.load(MODEL_PATH)


def load_sample():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["name", "status"],
        errors="ignore",
    )

    return X.iloc[[0]]


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
