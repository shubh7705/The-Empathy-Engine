# The Empathy Engine - Giving AI a Human Voice

> A production grade AI system that detects blended emotions in text and generates highly expressive, human-like speech with context-aware modulation.

---

## ✨ Features

| Feature                                              | Status |
| ---------------------------------------------------- | ------ |
| Blended Multi-Emotion Detection (Top-2 Mix)          | ✅      |
| 15 Combined Emotional Profiles ("Bittersweet", etc.) | ✅      |
| Confidence-scaled voice interpolation                | ✅      |
| ElevenLabs neural TTS (primary)                      | ✅      |
| pyttsx3 offline fallback                             | ✅      |
| Output formats: MP3 or natively constructed WAV      | ✅      |
| Multi-voice selector (15 verified voices)            | ✅      |
| Modern Responsive Web UI (HTML5 / Vanilla CSS / JS)  | ✅      |
| Real-time pipeline visualization & Audio history     | ✅      |
| REST API with FastAPI & Modular Architecture         | ✅      |

---

## 🏗 Architecture & Project Structure

The project follows a standard **Frontend / Backend** separation of concerns:

```
The-Empathy-Engine/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py         ← Central settings, voice catalog, env loader
│   │   ├── emotion.py        ← HuggingFace emotion classifier (w/ Top-2 blend logic)
│   │   ├── models.py         ← Pydantic API contracts & validation schemas
│   │   ├── tts_engine.py     ← ElevenLabs + pyttsx3 fallback (MP3/WAV outputs)
│   │   └── voice_mapper.py   ← Emotion interpolation + Blend-aware SSML builder
│   ├── utils/
│   │   ├── __init__.py
│   │   └── cache.py          ← MD5-based audio caching & TTL cleanup
│   ├── outputs/              ← Generated and cached audio files
│   ├── main.py               ← FastAPI app, CORS, API routes & static mounting
│   ├── requirements.txt      ← Backend Python dependencies
│   └── __init__.py
├── frontend/
│   ├── index.html            ← Semantic HTML5 single-page application
│   ├── css/
│   │   └── style.css         ← Modern design system, glassmorphism & responsive layout
│   ├── js/
│   │   ├── app.js            ← Frontend application logic, audio playback & history
│   │   └── config.js         ← Dynamic API URL detection & configuration
│   └── assets/
│       └── samples/          ← Bundled sample audio files
├── sample_audio/             ← Reference audio examples
├── .env.example              ← Environment variables template
├── requirements.txt          ← Root dependencies pointer
├── run.py                    ← Convenience launcher
└── README.md
```

---

## 🔬 Emotion Pipeline

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
│  → SSML preview string             │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  TTS Engine                        │
│  Primary: ElevenLabs API (MP3/PCM) │
│  Fallback: pyttsx3 (Native WAV)    │
└────────────────┬───────────────────┘
                 │
                 ▼
        Audio File (.mp3 / .wav)
        Served via /api/audio/ or /outputs/
```

---

## 🎛 Emotion → Voice Mapping

When the system detects a single emotion (or one vastly overpowers the other), it uses pure voice profiles.

### Example Pure Baselines:

| Emotion | Pitch           | Rate          | Stability | Style |
| ------- | --------------- | ------------- | --------- | ----- |
| Joy     | High (+3st)     | Fast          | 0.25      | 0.60+ |
| Sadness | Low (-3st)      | Slow          | 0.80      | 0.20  |
| Anger   | Low-Mid (-1st)  | Fast          | 0.30      | 0.70+ |
| Fear    | High-Mid (+1st) | Slightly Slow | 0.60      | 0.45  |

### Blended Interpolation

If two emotions are present (e.g. Joy 50%, Sadness 40%), the `voice_mapper` calculates a weighted average of their baseline Stability and Style parameters.

The system also maps this combination to a human-readable tone like **"Bittersweet"** and attaches a connector phrase to improve natural vocal inflection.

---

## 🚀 Setup & Execution

### Prerequisites

* Python 3.10+
* Internet connection (for ElevenLabs API & HuggingFace model download)

### 1. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: The first run will download the HuggingFace model (~330MB). Subsequent starts are cached and instant.

### 3. Configure API Key (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

*(If omitted or unconfigured, the system automatically uses `pyttsx3` for offline text-to-speech).*

### 4. Run Server

#### Option A: Unified Launcher (Backend + Frontend)

```bash
python run.py
```

* **Frontend**: `http://127.0.0.1:8000/`
* **Swagger API Docs**: `http://127.0.0.1:8000/docs`

#### Option B: Standalone Frontend Development

You can run the backend:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
And serve the `frontend/` directory with any static file server (e.g. VS Code Live Server, Vite, `npx serve frontend`, or `python -m http.server 3000 --directory frontend`). The frontend automatically routes requests to `http://127.0.0.1:8000`.

---

## 📡 API Reference

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

**Response:**

```json
{
  "success": true,
  "primary_emotion": {
    "label": "sadness",
    "score": 0.52,
    "intensity": "medium",
    "emoji": "😢",
    "color": "#6495ED"
  },
  "secondary_emotion": {
    "label": "joy",
    "score": 0.44,
    "intensity": "medium",
    "emoji": "😄",
    "color": "#FFD700"
  },
  "blended_emotion": {
    "is_blended": true,
    "label": "bittersweet",
    "description": "Happy yet touched with sorrow",
    "emoji": "🥹",
    "color": "#C8A2C8"
  },
  "voice_settings": {
    "stability": 0.52,
    "similarity_boost": 0.75,
    "style": 0.40,
    "use_speaker_boost": true,
    "pitch_label": "Low / High",
    "rate_label": "Slow / Fast"
  },
  "ssml_preview": "<speak>\n  <prosody rate=\"slow\" pitch=\"-3st\">\n    Happy yet a little sad... I got the job offer but I'll really miss my old team.\n  </prosody>\n</speak>",
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

Returns available voice catalog.

---

### `GET /api/health`

Returns system status and model initialization state.

---

### `GET /api/audio/{filename}`

Stream or download a generated audio file.

---

## 🛠 Development Options

```bash
# Hot-reload mode
python run.py --reload

# Custom port
python run.py --port 8080

# Bind all interfaces
python run.py --host 0.0.0.0
```

---

## 📝 Notes

* Audio outputs are saved to [`backend/outputs/`](file:///c:/Users/shubh/OneDrive/Desktop/Gen%20AI%20Projects/The-Empathy-Engine/backend/outputs)
* If ElevenLabs fails or limits are exceeded, the system automatically falls back to `pyttsx3`
* HuggingFace emotion model supports up to 512 tokens with automatic truncation
