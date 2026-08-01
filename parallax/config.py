import os
from dotenv import load_dotenv

load_dotenv()

GEMMA_ENDPOINT = os.getenv("GEMMA_ENDPOINT", "https://ai.spuric.com/v1/chat/completions")
API_KEY = os.getenv("API_KEY", "")
USE_LIVE_GEMMA = os.getenv("USE_LIVE_GEMMA", "true").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-27b-it")

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "..", "scenarios")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")
CACHED_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "cached_outputs")
