import logging

from litestar import Litestar, get

from app.api.prediction import predict
from app.logging_config import configure_logging


configure_logging()

logger = logging.getLogger(__name__)


@get("/health")
async def health() -> dict:
    logger.info("Health check requested")

    return {
        "status": "healthy",
        "service": "parkinson-ml-api",
    }


app = Litestar(
    route_handlers=[
        health,
        predict,
    ]
)
