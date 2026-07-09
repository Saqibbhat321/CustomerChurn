from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")

API_VERSION = os.getenv("API_VERSION", "1.0.0")
PROJECT_NAME = os.getenv("PROJECT_NAME", "Customer Churn Prediction API")

RANDOM_FOREST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}