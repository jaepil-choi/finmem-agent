import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory (3 levels up from src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    
    MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
    MONGODB_PORT = int(os.getenv("MONGODB_PORT", 27017))
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "reports")
    
    VECTOR_DB_ROOT = os.getenv("VECTOR_DB_ROOT", "data/vector_db")

settings = Settings()
