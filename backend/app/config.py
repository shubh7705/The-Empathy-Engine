"""
config.py — Central configuration for The Empathy Engine.
Loads API keys from .env and defines voice/emotion catalogs.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# Load .env from root or backend directory
if (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env")
elif (BACKEND_DIR / ".env").exists():
    load_dotenv(BACKEND_DIR / ".env")
else:
    load_dotenv()

# ── ElevenLabs ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID: str = "eleven_multilingual_v2"

# ── Voice catalog ─────────────────────────────────────────────────────────────
# VERIFIED: These 15 voices all return HTTP 200 on the free ElevenLabs plan.
# Voices that return 402 (Rachel, Domi, Josh, Aria, Charlotte) are excluded.
# Each voice is genuinely distinct — different gender, accent, and timbre.
VOICE_CATALOG: list[dict] = [
    # ── Female voices ──────────────────────────────────────────────────────
    {
        "id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Sarah",
        "description": "Warm, soft American female",
        "gender": "female",
    },
    {
        "id": "FGY2WhTYpPnrIDTdsKH5",
        "name": "Laura",
        "description": "Upbeat, expressive American female",
        "gender": "female",
    },
    {
        "id": "Xb7hH8MSUJpSbSDYk0k2",
        "name": "Alice",
        "description": "Confident, crisp British female",
        "gender": "female",
    },
    {
        "id": "SAz9YHcvj6GT2YYXdXww",
        "name": "River",
        "description": "Calm, androgynous American — neutral reads",
        "gender": "neutral",
    },
    # ── British male voices ────────────────────────────────────────────────
    {
        "id": "JBFqnCBsd6RMkjVDRZzb",
        "name": "George",
        "description": "Warm, authoritative British male",
        "gender": "male",
    },
    {
        "id": "N2lVS1w4EtoT3dr4eOWO",
        "name": "Callum",
        "description": "Intense, gritty British male",
        "gender": "male",
    },
    {
        "id": "SOYHLrjzK2X1ezoPC6cr",
        "name": "Harry",
        "description": "Anxious, youthful British male",
        "gender": "male",
    },
    # ── American male voices ───────────────────────────────────────────────
    {
        "id": "TX3LPaxmHKxFdv7VOQHJ",
        "name": "Liam",
        "description": "Energetic, articulate young American male",
        "gender": "male",
    },
    {
        "id": "onwK4e9ZLuTAKqWW03F9",
        "name": "Daniel",
        "description": "Deep, authoritative British/American male",
        "gender": "male",
    },
    {
        "id": "pqHfZKP75CvOlQylNhV4",
        "name": "Bill",
        "description": "Older, trustworthy American male",
        "gender": "male",
    },
    {
        "id": "nPczCjzI2devNBz1zQrb",
        "name": "Brian",
        "description": "Resonant, natural middle-aged American male",
        "gender": "male",
    },
    # ── Australian / International voices ──────────────────────────────────
    {
        "id": "IKne3meq5aSn9XLyUdCD",
        "name": "Charlie",
        "description": "Casual, friendly Australian male",
        "gender": "male",
    },
    {
        "id": "bIHbv24MWmeRgasZH58o",
        "name": "Will",
        "description": "Relaxed, conversational American male",
        "gender": "male",
    },
    {
        "id": "cgSgspJ2msm6clMCkdW9",
        "name": "Jessica",
        "description": "Playful, bright young American female",
        "gender": "female",
    },
    {
        "id": "iP95p4xoKVk53GoZ742B",
        "name": "Chris",
        "description": "Casual, everyday American male",
        "gender": "male",
    },
]

DEFAULT_VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"


def fetch_voices_from_api():
    """
    Try to fetch the caller's actual ElevenLabs voice library via REST.
    Returns a list of voice dicts on success, None if the API key lacks
    voices_read permission or the request fails.
    """
    if not ELEVENLABS_API_KEY:
        return None
    try:
        import requests
        resp = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        voices = []
        for v in data.get("voices", []):
            labels = v.get("labels") or {}
            voices.append({
                "id": v["voice_id"],
                "name": v["name"],
                "description": labels.get("description") or labels.get("use_case") or "ElevenLabs voice",
                "gender": labels.get("gender", "unknown"),
            })
        return voices if voices else None
    except Exception:
        return None


# ── HuggingFace ───────────────────────────────────────────────────────────────
EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"

# All supported emotion labels from the model
SUPPORTED_EMOTIONS: list[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

# ── Audio output ──────────────────────────────────────────────────────────────
OUTPUTS_DIR: Path = BACKEND_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTL in seconds (24 hours)
CACHE_TTL_SECONDS: int = 86400

# ── Intensity thresholds ──────────────────────────────────────────────────────
INTENSITY_HIGH: float = 0.85
INTENSITY_MEDIUM: float = 0.65
