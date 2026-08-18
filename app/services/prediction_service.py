import logging
from pathlib import Path

import joblib
import pandas as pd


logger = logging.getLogger(__name__)


class PredictionService:
    """Service responsible for loading the ML model and making predictions."""

    def __init__(self):
        self.model_path = (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "parkinson_model.joblib"
        )

        self.model = self._load_model()

        logger.info(
            "ML model loaded successfully from %s",
            self.model_path,
        )

    def _load_model(self):
        """Load the serialized ML pipeline."""

        if not self.model_path.exists():
            logger.error(
                "Model file not found: %s",
                self.model_path,
            )

            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        return joblib.load(self.model_path)

    def predict(self, features: dict) -> dict:
        """Make a Parkinson prediction from input features."""

        logger.info(
            "Prediction request received with %d features",
            len(features),
        )

        data = pd.DataFrame([features])

        prediction = int(self.model.predict(data)[0])
        probability = float(self.model.predict_proba(data)[0][1])

        result = {
            "prediction": prediction,
            "probability": probability,
            "label": (
                "Parkinson"
                if prediction == 1
                else "Healthy"
            ),
        }

        logger.info(
            "Prediction completed: label=%s probability=%.4f",
            result["label"],
            result["probability"],
        )

        return result
