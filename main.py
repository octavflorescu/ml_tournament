import os

from dotenv import load_dotenv
from evaluator import Evaluator
from api_client import APIClient

load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

evaluator = Evaluator(
    API_URL,
    API_KEY,
    APIClient(API_URL, API_KEY)
)

evaluator.evaluate()
