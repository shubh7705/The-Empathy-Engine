"""
main.py — FastAPI application entry point for The Empathy Engine.

Routes:
  POST /api/generate     — Full emotion-to-speech pipeline
  GET  /api/voices       — List available voices
  GET  /api/audio/{name} — Serve generated audio files
  GET  /api/health       — System health check
  GET  /                 — Serve frontend index.html
"""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Local imports ─────────────────────────────────────────────────────────────
from .config import (
    DEFAULT_VOICE_ID,
    ELEVENLABS_API_KEY,
    OUTPUTS_DIR,
    VOICE_CATALOG,
    fetch_voices_from_api,
)
from .emotion import detect_emotions, get_emotion_metadata, is_model_loaded
from .models import (
    BlendedEmotion,
    EmotionResult,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    VoiceInfo,
    VoiceListResponse,
    VoiceSettings,
)
from .tts_engine import generate_audio
from .voice_mapper import (
    blend_voice_profile,
    build_ssml_preview,
    modify_text_blend,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("empathy_engine.main")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="The Empathy Engine",
    description="AI-powered emotionally expressive text-to-speech pipeline",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
STATIC_ASSETS_DIR = FRONTEND_DIR / "assets"

# Mount audio output directory
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount frontend static assets (css, js)
if STATIC_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_ASSETS_DIR)), name="assets")


# ── Frontend serving ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        elevenlabs_configured=bool(ELEVENLABS_API_KEY),
        output_dir=str(OUTPUTS_DIR),
    )


# ── Voice list ────────────────────────────────────────────────────────────────
@app.get("/api/voices", response_model=VoiceListResponse, tags=["Voices"])
async def list_voices():
    # Try live fetch from ElevenLabs account first (uses voices_read permission).
    # Falls back to curated premade catalog if permission is missing.
    live_voices = fetch_voices_from_api()
    source = live_voices if live_voices else VOICE_CATALOG
    if live_voices:
        logger.info(f"Serving {len(live_voices)} voices from ElevenLabs account.")
    else:
        logger.info(f"voices_read not available — serving {len(VOICE_CATALOG)} premade voices from catalog.")
    voices = [VoiceInfo(**v) for v in source]
    return VoiceListResponse(voices=voices, default_voice_id=DEFAULT_VOICE_ID)


# ── Main pipeline ─────────────────────────────────────────────────────────────
@app.post("/api/generate", response_model=GenerateResponse, tags=["Engine"])
async def generate(request: GenerateRequest):
    """
    Full blended pipeline: Text → Top-2 Emotions → Blend → Voice Mapping → TTS → Audio
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    logger.info("Pipeline start | text='%s...' | voice=%s", text[:60], request.voice_id)

    # ── Step 1: Detect top-2 emotions + compute blend ─────────────────────
    primary, secondary, blend = detect_emotions(text)

    # ── Step 2: Blend-aware voice profile ────────────────────────────────
    voice_profile = blend_voice_profile(
        primary_label=primary["label"],
        primary_score=primary["score"],
        secondary_label=secondary["label"],
        secondary_score=secondary["score"],
        blend_label=blend["label"],
        is_blended=blend["is_blended"],
    )

    # ── Step 3: Build blended expressive text ────────────────────────────
    expressive_text = modify_text_blend(
        text=text,
        primary_label=primary["label"],
        primary_intensity=primary["intensity"],
        secondary_label=secondary["label"],
        secondary_score=secondary["score"],
        is_blended=blend["is_blended"],
    )
    ssml_preview = build_ssml_preview(expressive_text, voice_profile)

    # ── Step 4: Resolve voice ────────────────────────────────────────────
    voice_id = request.voice_id or DEFAULT_VOICE_ID
    voice_name = next(
        (v["name"] for v in VOICE_CATALOG if v["id"] == voice_id),
        voice_id[:8] + "...",
    )

    # ── Step 5: Generate audio ───────────────────────────────────────────
    audio_path, engine_used = generate_audio(
        text=expressive_text,
        profile=voice_profile,
        voice_id=voice_id,
        output_format=request.output_format,
    )

    if audio_path is None or engine_used == "failed":
        raise HTTPException(status_code=500, detail="Audio generation failed. Check server logs.")

    audio_filename = audio_path.name
    audio_url = "/outputs/{}".format(audio_filename)

    logger.info(
        "Pipeline complete | blend='%s' blended=%s | engine=%s | file=%s",
        blend["label"], blend["is_blended"], engine_used, audio_filename,
    )

    # Build shared EmotionResult for primary (also used as legacy .emotion)
    primary_result = EmotionResult(
        label=primary["label"],
        score=primary["score"],
        intensity=primary["intensity"],
        emoji=primary["emoji"],
        color=primary["color"],
        description=primary["description"],
    )
    secondary_result = EmotionResult(
        label=secondary["label"],
        score=secondary["score"],
        intensity=secondary["intensity"],
        emoji=secondary["emoji"],
        color=secondary["color"],
        description=secondary["description"],
    )

    return GenerateResponse(
        success=True,
        primary_emotion=primary_result,
        secondary_emotion=secondary_result,
        blended_emotion=BlendedEmotion(
            label=blend["label"],
            emoji=blend["emoji"],
            color=blend["color"],
            description=blend["description"],
            is_blended=blend["is_blended"],
        ),
        emotion=primary_result,   # backward-compatible legacy field
        voice_settings=VoiceSettings(
            stability=voice_profile.stability,
            similarity_boost=voice_profile.similarity_boost,
            style=voice_profile.style,
            use_speaker_boost=voice_profile.use_speaker_boost,
            pitch_label=voice_profile.pitch_label,
            rate_label=voice_profile.rate_label,
        ),
        ssml_preview=ssml_preview,
        audio_url=audio_url,
        audio_filename=audio_filename,
        voice_used=voice_id,
        voice_name=voice_name,
        tts_engine_used=engine_used,
        cached=False,
    )


# ── Audio file serving ────────────────────────────────────────────────────────
@app.get("/api/audio/{filename}", tags=["Audio"])
async def serve_audio(filename: str):
    """Serve a generated audio file by name."""
    # Security: only allow safe filenames
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    audio_path = OUTPUTS_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")

    media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"

    return FileResponse(
        path=str(audio_path),
        media_type=media_type,
        filename=filename,
    )
