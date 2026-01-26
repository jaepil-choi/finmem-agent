import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

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

    # Path to YAML configs
    PROMPT_CONFIG_DIR = PROJECT_ROOT / "configs" / "prompts"

    def __init__(self):
        self._factor_expertise = None
        self._risk_profiles = None

    @property
    def factor_expertise(self) -> Dict[str, Any]:
        """Lazy load factor expertise from YAML."""
        if self._factor_expertise is None:
            config_path = self.PROMPT_CONFIG_DIR / "factor_expertise.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._factor_expertise = yaml.safe_load(f).get("factor_expertise", {})
            else:
                self._factor_expertise = {}
        return self._factor_expertise

    @property
    def risk_profiles(self) -> Dict[str, Any]:
        """Lazy load risk profiles from YAML."""
        if self._risk_profiles is None:
            config_path = self.PROMPT_CONFIG_DIR / "risk_profiles.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._risk_profiles = yaml.safe_load(f).get("risk_profiles", {})
            else:
                self._risk_profiles = {}
        return self._risk_profiles

settings = Settings()
