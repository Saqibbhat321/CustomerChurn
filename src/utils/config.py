from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH")
API_VERSION = os.getenv("API_VERSION")
PROJECT_NAME = os.getenv("PROJECT_NAME")