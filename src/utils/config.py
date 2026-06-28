from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH")
API_VERSION = os.getenv("API_VERSION")
PROJECT_NAME = os.getenv("PROJECT_NAME")
RANDOM_FOREST_CONFIG = {

    "n_estimators": 300,

    "max_depth": 12,

    "min_samples_split": 5,

    "min_samples_leaf": 2,

    "random_state": 42,

    "n_jobs": -1

}