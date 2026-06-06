from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from src.utils.logger import get_logger
from src.utils.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PROJECT_NAME,
    API_VERSION
)

logger = get_logger()

app = FastAPI(
    title=PROJECT_NAME,
    version=API_VERSION
)

try:

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    logger.info(
        "Model artifacts loaded successfully."
    )

except Exception as e:

    logger.error(
        f"Artifact loading failed: {e}"
    )

    raise e


class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():

    logger.info(
        "Health check endpoint called."
    )

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:

        data = pd.DataFrame(
            [customer.model_dump()]
        )

        transformed_data = (
            preprocessor.transform(data)
        )

        prediction = model.predict(
            transformed_data
        )[0]

        probability = (
            model.predict_proba(
                transformed_data
            )[0][1]
        )

        logger.info(
            f"Prediction generated: {prediction}"
        )

        return {
            "prediction": int(prediction),
            "churn_probability": round(
                float(probability),
                4
            )
        }

    except Exception as e:

        logger.error(
            f"Prediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )