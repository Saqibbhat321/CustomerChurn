import os
from datetime import datetime

import joblib
import pandas as pd

from src.utils.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    API_VERSION
)
from src.utils.logger import get_logger

logger = get_logger()

PREDICTION_LOG_FILE = "prediction_logs/prediction_history.csv"


class ChurnPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        logger.info(
            "Model artifacts loaded successfully."
        )

    def predict(self, customer):

        data = pd.DataFrame(
            [customer.model_dump()]
        )

        transformed_data = (
            self.preprocessor.transform(data)
        )

        prediction = self.model.predict(
            transformed_data
        )[0]

        probability = (
            self.model.predict_proba(
                transformed_data
            )[0][1]
        )

        probability = round(
            float(probability),
            4
        )

        if prediction == 1:

            prediction_label = "Churn"

            confidence = (
                "High"
                if probability >= 0.80
                else "Medium"
                if probability >= 0.60
                else "Low"
            )

            message = (
                "This customer is likely to churn. "
                "Consider offering a retention plan or discount."
            )

        else:

            prediction_label = "No Churn"

            confidence_score = (
                1 - probability
            )

            confidence = (
                "High"
                if confidence_score >= 0.80
                else "Medium"
                if confidence_score >= 0.60
                else "Low"
            )

            message = (
                "This customer is likely to remain with the company."
            )

        self.save_prediction(
            customer,
            prediction_label,
            probability,
            confidence
        )

        logger.info(
            f"Prediction generated: {prediction_label}"
        )

        return {

            "prediction": int(prediction),

            "prediction_label": prediction_label,

            "churn_probability": probability,

            "confidence": confidence,

            "message": message,

            "model": "Random Forest Classifier",

            "api_version": API_VERSION

        }

    def save_prediction(

        self,

        customer,

        prediction,

        probability,

        confidence

    ):

        log_entry = {

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "prediction": prediction,

            "churn_probability": probability,

            "confidence": confidence,

            **customer.model_dump()

        }

        log_df = pd.DataFrame(
            [log_entry]
        )

        if os.path.exists(
            PREDICTION_LOG_FILE
        ):

            log_df.to_csv(

                PREDICTION_LOG_FILE,

                mode="a",

                header=False,

                index=False

            )

        else:

            log_df.to_csv(

                PREDICTION_LOG_FILE,

                index=False

            )