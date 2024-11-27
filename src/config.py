# config.py
import os
from dotenv import load_dotenv


class Config:
    """Configuration management using environment variables."""

    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.LITELLM_LOG = os.getenv("LITELLM_LOG", "INFO")
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.events']
        self.CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "../google_credentials.json")