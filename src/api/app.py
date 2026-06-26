from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerData
from src.api.predictor import ChurnPredictor
from src.utils.config import (
    PROJECT_NAME,
    API_VERSION
)
from src.utils.logger import get_logger

logger = get_logger()

app = FastAPI(
    title=PROJECT_NAME,
    version=API_VERSION
)

predictor = ChurnPredictor()


@app.get("/")
def home():

    logger.info(
        "Health check endpoint called."
    )

    return {
        "status": "healthy",
        "project": PROJECT_NAME,
        "version": API_VERSION
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:

        return predictor.predict(customer)

    except Exception as e:

        logger.error(
            f"Prediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )