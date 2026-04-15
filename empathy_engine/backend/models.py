"""
models.py — Pydantic request/response models for The Empathy Engine API
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Input text to synthesize")
    voice_id: Optional[str] = Field(None, description="ElevenLabs voice ID (uses default if omitted)")
    output_format: str = Field("mp3", description="Desired output format: 'mp3' or 'wav'")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I got the job but I'll really miss my team.",
                "voice_id": "EXAVITQu4vr4xnSDxMaL",
                "output_format": "mp3",
            }
        }


class EmotionResult(BaseModel):
    label: str = Field(..., description="Detected emotion label")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1)")
    intensity: str = Field(..., description="Intensity tier: low / medium / high")
    emoji: str = Field(..., description="Emoji representation")
    color: str = Field(..., description="Hex color for UI badge")
    description: str = Field(default="", description="Human-readable description")


class BlendedEmotion(BaseModel):
    """
    Represents the blended/combined emotional tone derived from
    the top-2 detected emotions.
    """
    label: str = Field(..., description="Combined tone label, e.g. 'bittersweet'")
    emoji: str = Field(..., description="Representative emoji")
    color: str = Field(..., description="Hex color for blend badge")
    description: str = Field(..., description="Human-readable blend description")
    is_blended: bool = Field(..., description="True if two emotions were meaningfully combined")


class VoiceSettings(BaseModel):
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    pitch_label: str
    rate_label: str


class GenerateResponse(BaseModel):
    success: bool

    # ── Blended emotion (new) ──────────────────────────────────────────────
    primary_emotion: EmotionResult = Field(..., description="Dominant detected emotion")
    secondary_emotion: EmotionResult = Field(..., description="Supporting emotion (may equal primary if no blend)")
    blended_emotion: BlendedEmotion = Field(..., description="Computed combined tone")

    # ── Legacy field kept for backward compatibility ───────────────────────
    emotion: EmotionResult = Field(..., description="Primary emotion (same as primary_emotion)")

    voice_settings: VoiceSettings
    ssml_preview: str = Field(..., description="SSML preview of what was sent to TTS")
    audio_url: str = Field(..., description="URL path to the generated audio file")
    audio_filename: str
    voice_used: str = Field(..., description="Voice ID used for generation")
    voice_name: str = Field(..., description="Human-readable voice name")
    tts_engine_used: str = Field(..., description="'elevenlabs' or 'pyttsx3'")
    cached: bool = Field(default=False)


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str
    gender: str


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]
    default_voice_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    elevenlabs_configured: bool
    output_dir: str
