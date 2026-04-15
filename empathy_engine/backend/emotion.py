"""
emotion.py — Emotion detection engine using HuggingFace Transformers.

Uses j-hartmann/emotion-english-distilroberta-base for 7-class classification.
Now supports TOP-2 blended emotion detection for more human-like expressions.
"""

import logging
from functools import lru_cache
from typing import List, Tuple, Dict

from transformers import pipeline as hf_pipeline

from .config import EMOTION_MODEL, INTENSITY_HIGH, INTENSITY_MEDIUM

logger = logging.getLogger("empathy_engine.emotion")

# ── Emotion metadata (emoji + UI color) ──────────────────────────────────────
EMOTION_METADATA: Dict[str, dict] = {
    "joy": {
        "emoji": "😄",
        "color": "#FFD700",
        "display": "Joy",
        "description": "Positive and happy",
    },
    "sadness": {
        "emoji": "😢",
        "color": "#6495ED",
        "display": "Sadness",
        "description": "Sorrowful or melancholic",
    },
    "anger": {
        "emoji": "😠",
        "color": "#FF4444",
        "display": "Anger",
        "description": "Frustrated or enraged",
    },
    "fear": {
        "emoji": "😨",
        "color": "#9B59B6",
        "display": "Fear",
        "description": "Anxious or fearful",
    },
    "surprise": {
        "emoji": "😲",
        "color": "#FF8C00",
        "display": "Surprise",
        "description": "Unexpected or astonished",
    },
    "disgust": {
        "emoji": "🤢",
        "color": "#2ECC71",
        "display": "Disgust",
        "description": "Repulsed or appalled",
    },
    "neutral": {
        "emoji": "😐",
        "color": "#95A5A6",
        "display": "Neutral",
        "description": "Matter-of-fact or calm",
    },
}

# ── Blended emotion label lookup ─────────────────────────────────────────────
# Key: frozenset of two emotion labels → human-readable blend label
# Ordered so both (A,B) and (B,A) resolve to the same entry.
_BLEND_LABELS: Dict[frozenset, Dict] = {
    frozenset({"joy", "sadness"}): {
        "label": "bittersweet",
        "emoji": "🥹",
        "color": "#C8A2C8",
        "description": "Happy yet touched with sorrow",
    },
    frozenset({"joy", "fear"}): {
        "label": "excited but nervous",
        "emoji": "😬",
        "color": "#FFB347",
        "description": "Thrilled yet anxious about what comes next",
    },
    frozenset({"joy", "surprise"}): {
        "label": "elated",
        "emoji": "🤩",
        "color": "#FFD700",
        "description": "Joyfully overwhelmed with delight",
    },
    frozenset({"joy", "anger"}): {
        "label": "triumphant",
        "emoji": "😤",
        "color": "#FFA500",
        "description": "Victorious but still fired up",
    },
    frozenset({"joy", "disgust"}): {
        "label": "conflicted excitement",
        "emoji": "😅",
        "color": "#90EE90",
        "description": "Pleased but something feels off",
    },
    frozenset({"anger", "sadness"}): {
        "label": "frustrated and hurt",
        "emoji": "😤",
        "color": "#CD5C5C",
        "description": "Deeply upset with a mix of pain and rage",
    },
    frozenset({"anger", "fear"}): {
        "label": "defensive",
        "emoji": "😡",
        "color": "#FF6347",
        "description": "Angry and on high alert",
    },
    frozenset({"anger", "surprise"}): {
        "label": "outraged",
        "emoji": "🤬",
        "color": "#FF4444",
        "description": "Shocked and furious at the same time",
    },
    frozenset({"anger", "disgust"}): {
        "label": "contemptuous",
        "emoji": "😤",
        "color": "#8B0000",
        "description": "Furious and revolted",
    },
    frozenset({"sadness", "fear"}): {
        "label": "despair",
        "emoji": "😰",
        "color": "#4169E1",
        "description": "Sorrowful and deeply afraid",
    },
    frozenset({"sadness", "surprise"}): {
        "label": "dismayed",
        "emoji": "😢",
        "color": "#87CEEB",
        "description": "Shocked and saddened",
    },
    frozenset({"sadness", "disgust"}): {
        "label": "bitter",
        "emoji": "😞",
        "color": "#708090",
        "description": "Sad with an aftertaste of resentment",
    },
    frozenset({"fear", "surprise"}): {
        "label": "startled",
        "emoji": "😱",
        "color": "#DA70D6",
        "description": "Caught off guard and frightened",
    },
    frozenset({"fear", "disgust"}): {
        "label": "horrified",
        "emoji": "🫣",
        "color": "#800080",
        "description": "Deeply disturbed",
    },
    frozenset({"surprise", "disgust"}): {
        "label": "appalled",
        "emoji": "😳",
        "color": "#3CB371",
        "description": "Shocked by something revolting",
    },
}

# Threshold: if the gap between primary and secondary scores is this large,
# the secondary emotion is considered noise — single-emotion mode is used.
_BLEND_DOMINANCE_THRESHOLD = 0.45


@lru_cache(maxsize=1)
def _get_classifier():
    """Lazy singleton: loads the HuggingFace model exactly once, with top_k=2."""
    logger.info("Loading emotion model: %s (top_k=2)", EMOTION_MODEL)
    clf = hf_pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        top_k=2,           # Now returns top 2 emotions
        truncation=True,
        max_length=512,
    )
    logger.info("Emotion model loaded successfully.")
    return clf


def get_intensity_label(score: float) -> str:
    """Map confidence score to human-readable intensity."""
    if score >= INTENSITY_HIGH:
        return "high"
    elif score >= INTENSITY_MEDIUM:
        return "medium"
    return "low"


def get_emotion_metadata(label: str) -> dict:
    """Return display metadata for a given emotion label."""
    return EMOTION_METADATA.get(label, EMOTION_METADATA["neutral"])


def is_model_loaded() -> bool:
    """Check if the classifier has been initialized."""
    return _get_classifier.cache_info().currsize > 0


def blend_emotions(primary_label: str, primary_score: float,
                   secondary_label: str, secondary_score: float) -> Dict:
    """
    Compute a blended emotional tone from two emotions.

    Rules:
    - If the primary score dominates by > THRESHOLD → pure primary emotion
    - If both are neutral → pure neutral
    - Otherwise → look up the blend table and return a combined label

    Returns a dict with: label, emoji, color, description, is_blended (bool)
    """
    gap = primary_score - secondary_score

    # Dominant single emotion
    if gap >= _BLEND_DOMINANCE_THRESHOLD or secondary_label == "neutral":
        meta = EMOTION_METADATA.get(primary_label, EMOTION_METADATA["neutral"])
        return {
            "label": meta["display"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "description": meta["description"],
            "is_blended": False,
        }

    # Look up blend
    key = frozenset({primary_label, secondary_label})
    blend = _BLEND_LABELS.get(key)
    if blend:
        return {**blend, "is_blended": True}

    # Fallback: primary still wins but acknowledge the secondary
    meta = EMOTION_METADATA.get(primary_label, EMOTION_METADATA["neutral"])
    sec_meta = EMOTION_METADATA.get(secondary_label, EMOTION_METADATA["neutral"])
    return {
        "label": "{} with a hint of {}".format(
            meta["display"].lower(), sec_meta["display"].lower()
        ),
        "emoji": meta["emoji"],
        "color": meta["color"],
        "description": "{}, with some {}".format(meta["description"], sec_meta["description"]),
        "is_blended": True,
    }


def detect_emotions(text: str) -> Tuple[Dict, Dict, Dict]:
    """
    Detect the top 2 emotions and compute their blended tone.

    Returns:
        Tuple of (primary_dict, secondary_dict, blend_dict)

        Each emotion dict has: label, score, intensity, emoji, color, description
        Blend dict has: label, emoji, color, description, is_blended
    """
    try:
        classifier = _get_classifier()
        results = classifier(text)[0]   # list of top-2 dicts

        primary_raw = results[0]
        primary_label = primary_raw["label"].lower()
        primary_score = round(float(primary_raw["score"]), 4)
        primary_intensity = get_intensity_label(primary_score)
        primary_meta = EMOTION_METADATA.get(primary_label, EMOTION_METADATA["neutral"])

        primary = {
            "label": primary_label,
            "score": primary_score,
            "intensity": primary_intensity,
            "emoji": primary_meta["emoji"],
            "color": primary_meta["color"],
            "description": primary_meta["description"],
        }

        # Secondary (may not exist in edge cases)
        if len(results) >= 2:
            secondary_raw = results[1]
            secondary_label = secondary_raw["label"].lower()
            secondary_score = round(float(secondary_raw["score"]), 4)
            secondary_intensity = get_intensity_label(secondary_score)
            secondary_meta = EMOTION_METADATA.get(secondary_label, EMOTION_METADATA["neutral"])

            secondary = {
                "label": secondary_label,
                "score": secondary_score,
                "intensity": secondary_intensity,
                "emoji": secondary_meta["emoji"],
                "color": secondary_meta["color"],
                "description": secondary_meta["description"],
            }
        else:
            # Fallback: echo primary as secondary (backward compat)
            secondary = {**primary, "score": 0.0, "intensity": "low"}
            secondary_label = primary_label
            secondary_score = 0.0

        blend = blend_emotions(primary_label, primary_score, secondary_label, secondary_score)

        logger.info(
            "Emotions: primary=%s(%.2f) secondary=%s(%.2f) blend='%s' blended=%s",
            primary_label, primary_score,
            secondary_label, secondary_score,
            blend["label"], blend["is_blended"],
        )

        return primary, secondary, blend

    except Exception as exc:
        logger.error("Emotion detection failed: %s", exc, exc_info=True)
        neutral = {
            "label": "neutral", "score": 0.5, "intensity": "low",
            "emoji": "😐", "color": "#95A5A6", "description": "Matter-of-fact or calm",
        }
        blend_fallback = {
            "label": "Neutral", "emoji": "😐", "color": "#95A5A6",
            "description": "Matter-of-fact or calm", "is_blended": False,
        }
        return neutral, neutral, blend_fallback


# ── Backward-compatible single-emotion wrapper ────────────────────────────────
def detect_emotion(text: str) -> Tuple[str, float, str]:
    """
    Legacy wrapper — returns (label, score, intensity) for the primary emotion.
    Used internally; prefer detect_emotions() for new code.
    """
    primary, _, _ = detect_emotions(text)
    return primary["label"], primary["score"], primary["intensity"]
