"""
cache.py — MD5-based audio caching utility.
"""

import hashlib
import logging
import time
from pathlib import Path

from empathy_engine.backend.config import CACHE_TTL_SECONDS, OUTPUTS_DIR

logger = logging.getLogger("empathy_engine.cache")


def _cache_key(text: str, voice_id: str, emotion: str) -> str:
    raw = f"{text}::{voice_id}::{emotion}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_audio(text: str, voice_id: str, emotion: str) -> Path | None:
    """Return path to cached audio if it exists and is fresh."""
    key = _cache_key(text, voice_id, emotion)
    cache_path = OUTPUTS_DIR / f"cache_{key}.mp3"

    if not cache_path.exists():
        return None

    age = time.time() - cache_path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        logger.info(f"Cache expired for key={key[:8]}. Removing.")
        cache_path.unlink(missing_ok=True)
        return None

    logger.info(f"Cache hit: {cache_path.name}")
    return cache_path


def save_to_cache(text: str, voice_id: str, emotion: str, source_path: Path) -> Path:
    """Copy a generated file to the cache location."""
    import shutil
    key = _cache_key(text, voice_id, emotion)
    cache_path = OUTPUTS_DIR / f"cache_{key}.mp3"
    shutil.copy2(source_path, cache_path)
    logger.info(f"Saved to cache: {cache_path.name}")
    return cache_path


def cleanup_old_cache():
    """Delete all cache files older than CACHE_TTL_SECONDS."""
    now = time.time()
    removed = 0
    for f in OUTPUTS_DIR.glob("cache_*.mp3"):
        if now - f.stat().st_mtime > CACHE_TTL_SECONDS:
            f.unlink(missing_ok=True)
            removed += 1
    if removed:
        logger.info(f"Cache cleanup: removed {removed} files.")
