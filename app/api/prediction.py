from litestar import post

from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import PredictionService


prediction_service = PredictionService()


@post("/predict", status_code=200)
async def predict(data: PredictionRequest) -> dict:
    """Predict Parkinson's disease from biomedical features."""

    features = data.to_model_features()

    return prediction_service.predict(features)
