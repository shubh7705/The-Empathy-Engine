"""
tts_engine.py — Text-to-Speech engine with ElevenLabs (primary) and pyttsx3 (fallback).
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    DEFAULT_VOICE_ID,
    OUTPUTS_DIR,
)
from .voice_mapper import VoiceProfile

logger = logging.getLogger("empathy_engine.tts")


def _generate_elevenlabs(
    text: str,
    voice_id: str,
    profile: VoiceProfile,
    output_path: Path,
    output_format: str = "mp3",
) -> bool:
    """
    Attempt audio generation via ElevenLabs SDK.

    Returns True on success, False on failure.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("No ElevenLabs API key configured. Skipping ElevenLabs.")
        return False

    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        voice_settings = VoiceSettings(
            stability=profile.stability,
            similarity_boost=profile.similarity_boost,
            style=profile.style,
            use_speaker_boost=profile.use_speaker_boost,
        )

        logger.info(f"Calling ElevenLabs: voice={voice_id}, stability={profile.stability}, style={profile.style}, format={output_format}")

        if output_format == "wav":
            audio_stream = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL_ID,
                voice_settings=voice_settings,
                output_format="pcm_44100_16",
            )
            raw_pcm = b"".join(chunk for chunk in audio_stream if chunk)
            import wave
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(44100)
                wav_file.writeframes(raw_pcm)
        else:
            audio_stream = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL_ID,
                voice_settings=voice_settings,
                output_format="mp3_44100_128",
            )
            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)

        logger.info(f"ElevenLabs audio saved: {output_path}")
        return True

    except Exception as exc:
        logger.error(f"ElevenLabs generation failed: {exc}", exc_info=True)
        return False


def _generate_pyttsx3(text: str, output_path: Path, profile: VoiceProfile, output_format: str = "mp3") -> bool:
    """
    Fallback TTS using pyttsx3 (offline).
    Saves as .wav then converts path reference.
    """
    try:
        import pyttsx3

        engine = pyttsx3.init()

        # Apply modulation: rate and volume
        base_rate = 180  # words per minute
        if profile.rate_label in ("Fast", "Very Fast"):
            rate = int(base_rate * 1.35)
        elif profile.rate_label in ("Slow", "Slightly Slow"):
            rate = int(base_rate * 0.70)
        else:
            rate = base_rate

        engine.setProperty("rate", rate)
        engine.setProperty("volume", 1.0)

        # pyttsx3 saves as .wav internally
        wav_path = output_path.with_suffix(".wav")
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()

        if output_format == "mp3":
            # Rename to mp3 path (still wav content but acceptable for most players)
            import shutil
            shutil.copy(wav_path, output_path)
            wav_path.unlink(missing_ok=True)
        else:
            # If the output_path itself isn't already the wav_path (should be in generate_audio)
            if wav_path != output_path:
                import shutil
                shutil.copy(wav_path, output_path)
                wav_path.unlink(missing_ok=True)

        logger.info(f"pyttsx3 audio saved: {output_path}")
        return True

    except Exception as exc:
        logger.error(f"pyttsx3 generation failed: {exc}", exc_info=True)
        return False


def generate_audio(
    text: str,
    profile: VoiceProfile,
    voice_id: Optional[str] = None,
    filename: Optional[str] = None,
    output_format: str = "mp3",
) -> Tuple[Optional[Path], str]:
    """
    Generate audio for the given text using the best available engine.

    Args:
        text:      Text to synthesize.
        profile:   Voice modulation settings.
        voice_id:  ElevenLabs voice ID (uses default if None).
        filename:  Output filename (auto-generated UUID if None).

    Returns:
        Tuple of (output_path | None, engine_used: 'elevenlabs' | 'pyttsx3' | 'failed')
    """
    effective_voice_id = voice_id or DEFAULT_VOICE_ID
    output_filename = filename or f"audio_{uuid.uuid4()}.{output_format}"
    output_path = OUTPUTS_DIR / output_filename

    # Primary: ElevenLabs
    if _generate_elevenlabs(text, effective_voice_id, profile, output_path, output_format):
        return output_path, "elevenlabs"

    # Fallback: pyttsx3
    logger.warning("ElevenLabs failed; falling back to pyttsx3.")
    if _generate_pyttsx3(text, output_path, profile, output_format):
        return output_path, "pyttsx3"

    logger.error("Both TTS engines failed.")
    return None, "failed"
