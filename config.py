import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "").strip()
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", "").strip()
HINDSIGHT_BANK_ID = os.getenv("HINDSIGHT_BANK_ID", "").strip()


def groq_ready() -> bool:
    return bool(GROQ_API_KEY)


def hindsight_ready() -> bool:
    return bool(HINDSIGHT_API_URL and HINDSIGHT_API_KEY and HINDSIGHT_BANK_ID)
