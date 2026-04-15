# 🎤 The Empathy Engine — Giving AI a Human Voice

> A production-grade AI system that detects blended emotions in text and generates highly expressive, human-like speech with context-aware modulation.

---

## ✨ Features

| Feature | Status |
|---|---|
| Blended Multi-Emotion Detection (Top-2 Mix) | ✅ |
| 15 Combined Emotional Profiles ("Bittersweet", etc.) | ✅ |
| Confidence-scaled voice interpolation | ✅ |
| ElevenLabs neural TTS (primary) | ✅ |
| pyttsx3 offline fallback | ✅ |
| Output formats: MP3 or natively constructed WAV | ✅ |
| Multi-voice selector (6 voices) | ✅ |
| Vercel-like Minimalist Web UI | ✅ |
| Real-time pipeline visualization & Audio history | ✅ |
| REST API with FastAPI | ✅ |

---

## 🏗️ Architecture

```
The Empathy Engine/
├── empathy_engine/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── config.py        ← All settings, voice catalog, env loading
│   │   ├── emotion.py       ← HuggingFace emotion classifier (w/ Blended logic)
│   │   ├── voice_mapper.py  ← Emotion interpolation + Blend-aware SSML Builder
│   │   ├── tts_engine.py    ← ElevenLabs + pyttsx3 fallback (MP3/WAV outputs)
│   │   ├── models.py        ← Pydantic API contracts
│   │   └── main.py          ← FastAPI app, routes, static serving
│   ├── frontend/
│   │   └── index.html       ← Simplistic, responsive single-page UI
│   ├── utils/
│   │   └── cache.py         ← MD5-based audio caching
│   └── outputs/             ← Generated audio files (auto-created)
├── run.py                   ← Convenience launcher
├── requirements.txt
└── .env                     ← API key (never commit this)
```

### Emotion Pipeline

```
Text Input
    │
    ▼
┌────────────────────────────────────┐
│  Emotion Detection                 │
│  (j-hartmann/distilroberta-base)  │
│  → Returns Top 2 emotions          │
│  → Determines Blended State        │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  Voice Mapper                      │
│  → Interpolates stability & style  │
│  → Context-aware phrases (prefix)  │
│  → SSML preview string            │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  TTS Engine                        │
│  Primary:  ElevenLabs API (MP3/PCM)│
│  Fallback: pyttsx3 (Native WAV)   │
└────────────────┬───────────────────┘
                 │
                 ▼
         Audio File (.mp3 / .wav)
         Served via /api/audio/
```

---

## 🎭 Emotion → Voice Mapping

When the system detects a single emotion (or one vastly overpowers the other), it uses pure voice profiles.

**Example Pure Baselines:**
| Emotion | Pitch | Rate | Stability | Style |
|---|---|---|---|---|
| Joy | High (+3st) | Fast | 0.25 | 0.60+ |
| Sadness | Low (-3st) | Slow | 0.80 | 0.20 |
| Anger | Low-Mid (-1st) | Fast | 0.30 | 0.70+ |
| Fear | High-Mid (+1st) | Slightly Slow | 0.60 | 0.45 |

**Blended Interpolation:**
If two emotions are present (e.g. Joy 50%, Sadness 40%), the `voice_mapper` calculates a weighted average of their baseline Stability and Style parameters. The system also maps this combination to a human-readable Tone like **"Bittersweet"** and attaches a connector phrase to improve TwelveLabs' inflection.

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- Internet connection (for ElevenLabs API & HuggingFace model download)

### 1. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the HuggingFace model (~330MB). Subsequent starts are instant.

### 3. Configure API key
Edit `.env`:
```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 4. Run the server
```bash
python run.py
```

Open your browser at **http://127.0.0.1:8000** (or `8001` depending on port settings).

---

## 🌐 API Reference

### `POST /api/generate`
Run the full emotion-to-speech pipeline.

**Request body (JSON):**
```json
{
  "text": "I got the job offer but I'll really miss my old team.",
  "voice_id": "EXAVITQu4vr4xnSDxMaL",
  "output_format": "mp3" 
}
```
*(Optionally set `output_format` to `"wav"` to receive parsed PCM WAV structures).*

**Response:**
```json
{
  "success": true,
  "emotion": {
    "label": "sadness",
    "score": 0.52,
    "intensity": "medium",
    "emoji": "😢",
    "color": "#6495ED"
  },
  "primary_emotion": {
    "label": "sadness",
    "score": 0.52,
    "intensity": "medium",
    "emoji": "😢"
  },
  "secondary_emotion": {
    "label": "joy",
    "score": 0.44,
    "intensity": "medium",
    "emoji": "😄"
  },
  "blended_emotion": {
    "is_blended": true,
    "label": "Bittersweet",
    "description": "Happy yet a little sad...",
    "emoji": "🥹"
  },
  "voice_settings": {
    "stability": 0.52,
    "similarity_boost": 0.75,
    "style": 0.40,
    "use_speaker_boost": true,
    "pitch_label": "Interpolated",
    "rate_label": "Interpolated"
  },
  "ssml_preview": "<speak>\n  <prosody rate=\"medium\" pitch=\"0st\">\n    Happy yet a little sad... I got the job offer but I'll really miss my old team.\n  </prosody>\n</speak>",
  "audio_url": "/outputs/audio_<uuid>.mp3",
  "audio_filename": "audio_<uuid>.mp3",
  "voice_used": "EXAVITQu4vr4xnSDxMaL",
  "voice_name": "Sarah",
  "tts_engine_used": "elevenlabs",
  "cached": false
}
```

---

### `GET /api/voices`
List all available voices.

```json
{
  "voices": [
    { "id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "description": "Warm & expressive", "gender": "female" }
  ],
  "default_voice_id": "EXAVITQu4vr4xnSDxMaL"
}
```

### `GET /api/audio/{filename}`
Stream or download a generated audio file cleanly serving correct `audio/mpeg` or `audio/wav` MIME types automatically.

---

## 🚀 Development

```bash
# Hot-reload mode
python run.py --reload

# Different port
python run.py --port 8080

# Public access
python run.py --host 0.0.0.0
```

API docs available at: `http://127.0.0.1:8000/docs`

---

## 📝 Notes

- Audio files are saved to `empathy_engine/outputs/` (auto-cleaned after 24h)
- If ElevenLabs fails or limits are hit, the system automatically falls back to pyttsx3 (offline, robotic) while seamlessly honoring your requested format wrapper (`WAV` or `MP3`).
- The HuggingFace model truncates inputs longer than 512 tokens.
