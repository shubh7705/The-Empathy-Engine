"""
voice_mapper.py — Maps detected emotion(s) to ElevenLabs voice settings.

Supports blended voice profiles: when two emotions are close in score,
their voice settings are interpolated to produce a mixed, nuanced delivery.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import INTENSITY_HIGH, INTENSITY_MEDIUM

logger = logging.getLogger("empathy_engine.voice_mapper")


# ── Voice settings dataclass ──────────────────────────────────────────────────
@dataclass
class VoiceProfile:
    stability: float          # 0.0 = very expressive, 1.0 = very stable
    similarity_boost: float   # Voice clarity / likeness
    style: float              # Style exaggeration (0.0–1.0)
    use_speaker_boost: bool
    pitch_label: str          # UI label e.g. "High", "Low", "Balanced"
    rate_label: str           # UI label e.g. "Fast", "Slow", "Normal"
    ssml_rate: str            # e.g. "fast", "medium", "slow"
    ssml_pitch: str           # e.g. "+3st", "-2st", "0st"


# ── Base emotion profiles ─────────────────────────────────────────────────────
_BASE_PROFILES: Dict[str, VoiceProfile] = {
    "joy": VoiceProfile(
        stability=0.25, similarity_boost=0.75, style=0.60,
        use_speaker_boost=True,
        pitch_label="High", rate_label="Fast",
        ssml_rate="fast", ssml_pitch="+3st",
    ),
    "sadness": VoiceProfile(
        stability=0.80, similarity_boost=0.65, style=0.20,
        use_speaker_boost=False,
        pitch_label="Low", rate_label="Slow",
        ssml_rate="slow", ssml_pitch="-3st",
    ),
    "anger": VoiceProfile(
        stability=0.30, similarity_boost=0.80, style=0.70,
        use_speaker_boost=True,
        pitch_label="Low-Mid", rate_label="Fast",
        ssml_rate="fast", ssml_pitch="-1st",
    ),
    "fear": VoiceProfile(
        stability=0.60, similarity_boost=0.70, style=0.45,
        use_speaker_boost=True,
        pitch_label="High-Mid", rate_label="Slightly Slow",
        ssml_rate="medium-slow", ssml_pitch="+1st",
    ),
    "surprise": VoiceProfile(
        stability=0.20, similarity_boost=0.75, style=0.65,
        use_speaker_boost=True,
        pitch_label="Very High", rate_label="Fast",
        ssml_rate="fast", ssml_pitch="+4st",
    ),
    "disgust": VoiceProfile(
        stability=0.55, similarity_boost=0.70, style=0.40,
        use_speaker_boost=False,
        pitch_label="Low", rate_label="Slightly Slow",
        ssml_rate="medium-slow", ssml_pitch="-2st",
    ),
    "neutral": VoiceProfile(
        stability=0.50, similarity_boost=0.75, style=0.15,
        use_speaker_boost=False,
        pitch_label="Balanced", rate_label="Normal",
        ssml_rate="medium", ssml_pitch="0st",
    ),
}

_SSML_RATE_MAP: Dict[str, str] = {
    "fast": "fast",
    "medium-fast": "+15%",
    "medium": "medium",
    "medium-slow": "-15%",
    "slow": "slow",
}

# ── Blend thresholds ──────────────────────────────────────────────────────────
_BLEND_DOMINANCE_THRESHOLD = 0.75
_BLEND_PRIMARY_WEIGHT = 0.70


# ── Internal helpers ──────────────────────────────────────────────────────────
def _scale_by_intensity(profile: VoiceProfile, score: float) -> VoiceProfile:
    """
    Scale expressiveness based on confidence score.
    """
    if score >= INTENSITY_HIGH:
        multiplier = 1.2
    elif score >= INTENSITY_MEDIUM:
        multiplier = 1.0
    else:
        multiplier = 0.7

    new_style = min(1.0, profile.style * multiplier)
    stability_delta = (1.0 - profile.stability) * (multiplier - 1.0) * 0.3
    new_stability = max(0.1, min(1.0, profile.stability - stability_delta))

    return dataclasses.replace(
        profile,
        stability=round(new_stability, 3),
        style=round(new_style, 3),
    )


def _interpolate_profiles(
    p1: VoiceProfile, w1: float,
    p2: VoiceProfile, w2: float,
    blend_label: str,
) -> VoiceProfile:
    """
    Linearly interpolate two VoiceProfiles by given weights (w1 + w2 = 1.0).
    """
    def lerp(a: float, b: float) -> float:
        return round(a * w1 + b * w2, 3)

    return VoiceProfile(
        stability=lerp(p1.stability, p2.stability),
        similarity_boost=lerp(p1.similarity_boost, p2.similarity_boost),
        style=lerp(p1.style, p2.style),
        use_speaker_boost=p1.use_speaker_boost,
        pitch_label=f"{p1.pitch_label} / {p2.pitch_label}",
        rate_label=f"{p1.rate_label} / {p2.rate_label}",
        ssml_rate=p1.ssml_rate,
        ssml_pitch=p1.ssml_pitch,
    )


# ── Public API ────────────────────────────────────────────────────────────────
def map_emotion_to_voice(emotion: str, score: float) -> VoiceProfile:
    """Single-emotion voice mapping (backward compatible)."""
    base = _BASE_PROFILES.get(emotion, _BASE_PROFILES["neutral"])
    scaled = _scale_by_intensity(base, score)
    logger.debug("Single map: '%s'(%.2f) → stability=%.2f style=%.2f",
                 emotion, score, scaled.stability, scaled.style)
    return scaled


def blend_voice_profile(
    primary_label: str, primary_score: float,
    secondary_label: str, secondary_score: float,
    blend_label: str,
    is_blended: bool,
) -> VoiceProfile:
    """Produce a VoiceProfile for blended or single emotions."""
    primary_base = _BASE_PROFILES.get(primary_label, _BASE_PROFILES["neutral"])
    primary_scaled = _scale_by_intensity(primary_base, primary_score)

    if not is_blended or primary_score >= _BLEND_DOMINANCE_THRESHOLD:
        logger.debug(
            "Blend map: primary '%s' dominates (%.2f >= %.2f)",
            primary_label, primary_score, _BLEND_DOMINANCE_THRESHOLD,
        )
        return primary_scaled

    total = primary_score + secondary_score
    w1 = primary_score / total if total > 0 else _BLEND_PRIMARY_WEIGHT
    w2 = 1.0 - w1

    secondary_base = _BASE_PROFILES.get(secondary_label, _BASE_PROFILES["neutral"])
    secondary_scaled = _scale_by_intensity(secondary_base, secondary_score)

    blended = _interpolate_profiles(primary_scaled, w1, secondary_scaled, w2, blend_label)
    logger.debug(
        "Blend map: '%s'(%.2f)*%.2f + '%s'(%.2f)*%.2f → stability=%.2f style=%.2f",
        primary_label, primary_score, w1,
        secondary_label, secondary_score, w2,
        blended.stability, blended.style,
    )
    return blended


# ── Blended text modifier ─────────────────────────────────────────────────────
_INTENSITY_PREFIX: Dict[str, Dict[str, str]] = {
    "joy": {
        "high": "Extremely excited! ",
        "medium": "Very happy! ",
        "low": "",
    },
    "sadness": {
        "high": "I'm deeply sorry... ",
        "medium": "I'm sorry... ",
        "low": "",
    },
    "anger": {
        "high": "This is absolutely unacceptable! ",
        "medium": "This is really frustrating! ",
        "low": "",
    },
    "fear": {
        "high": "This is terrifying... ",
        "medium": "This is concerning... ",
        "low": "",
    },
    "surprise": {
        "high": "Wow, I cannot believe this! ",
        "medium": "Oh wow! ",
        "low": "Oh! ",
    },
    "disgust": {
        "high": "This is absolutely appalling! ",
        "medium": "This is really unpleasant. ",
        "low": "",
    },
    "neutral": {"high": "", "medium": "", "low": ""},
}

_BLEND_CONNECTORS: Dict[frozenset, Dict[str, str]] = {
    frozenset({"joy", "sadness"}):   {"high": "I'm really happy, but also a bit emotional... ", "medium": "Happy yet a little sad... ", "low": ""},
    frozenset({"joy", "fear"}):      {"high": "I'm so excited, but honestly a little scared... ", "medium": "Excited yet nervous... ", "low": ""},
    frozenset({"joy", "surprise"}):  {"high": "I'm absolutely blown away and overjoyed! ", "medium": "Wow, this is amazing! ", "low": ""},
    frozenset({"joy", "anger"}):     {"high": "I'm thrilled, but still fired up... ", "medium": "Happy but still frustrated! ", "low": ""},
    frozenset({"joy", "disgust"}):   {"high": "Happy, though something about this feels wrong... ", "medium": "Glad, but a bit put off. ", "low": ""},
    frozenset({"anger", "sadness"}): {"high": "I'm furious and deeply hurt... ", "medium": "Frustrated and upset... ", "low": ""},
    frozenset({"anger", "fear"}):    {"high": "I'm angry but also scared... ", "medium": "On edge and defensive... ", "low": ""},
    frozenset({"anger", "surprise"}):{"high": "I can't believe this — I'm furious! ", "medium": "Shocked and angry! ", "low": ""},
    frozenset({"anger", "disgust"}): {"high": "This is revolting and I am furious! ", "medium": "Really disgusted and angry. ", "low": ""},
    frozenset({"sadness", "fear"}):  {"high": "I'm heartbroken and terrified... ", "medium": "Sad and afraid... ", "low": ""},
    frozenset({"sadness", "surprise"}):{"high": "I'm shocked and deeply saddened... ", "medium": "Surprised and a bit sad... ", "low": ""},
    frozenset({"sadness", "disgust"}):{"high": "This is bitter and disgusting... ", "medium": "Sad and repulsed. ", "low": ""},
    frozenset({"fear", "surprise"}): {"high": "I'm completely startled and terrified! ", "medium": "Frightened and caught off guard... ", "low": ""},
    frozenset({"fear", "disgust"}):  {"high": "This is horrifying and revolting... ", "medium": "Disturbed and disgusted. ", "low": ""},
    frozenset({"surprise", "disgust"}):{"high": "I'm appalled — I can't believe this! ", "medium": "Shocked and disgusted. ", "low": ""},
}


def modify_text_blend(
    text: str,
    primary_label: str,
    primary_intensity: str,
    secondary_label: str,
    secondary_score: float,
    is_blended: bool,
) -> str:
    """Generate expressive text reflecting blended or single emotions."""
    if not is_blended:
        prefix = _INTENSITY_PREFIX.get(primary_label, {}).get(primary_intensity, "")
        return f"{prefix}{text}"

    key = frozenset({primary_label, secondary_label})
    connector_map = _BLEND_CONNECTORS.get(key)

    if connector_map:
        connector = connector_map.get(primary_intensity, "")
    else:
        primary_meta_label = primary_label.capitalize()
        secondary_meta_label = secondary_label.capitalize()
        connector = f"Feeling {primary_meta_label.lower()}, and also a bit {secondary_meta_label.lower()}... "

    return f"{connector}{text}"


def build_expressive_text(text: str, emotion: str, intensity: str) -> str:
    """Backward-compatible single-emotion text builder."""
    prefix = _INTENSITY_PREFIX.get(emotion, {}).get(intensity, "")
    return f"{prefix}{text}"


def build_ssml_preview(text: str, profile: VoiceProfile) -> str:
    """Build an SSML string for display/preview purposes."""
    rate = _SSML_RATE_MAP.get(profile.ssml_rate, "medium")
    return (
        f'<speak>\n'
        f'  <prosody rate="{rate}" pitch="{profile.ssml_pitch}">\n'
        f'    {text}\n'
        f'  </prosody>\n'
        f'</speak>'
    )
