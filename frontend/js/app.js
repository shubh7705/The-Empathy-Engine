/**
 * app.js — Main application logic for The Empathy Engine frontend.
 */

import { API_BASE, APP_CONFIG } from "./config.js";

// ── DOM Element Selectors ───────────────────────────────────────────────────
const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const voiceSelect = document.getElementById("voice-select");
const formatSelect = document.getElementById("format-select");
const generateBtn = document.getElementById("generate-btn");
const errorBanner = document.getElementById("error-banner");
const loader = document.getElementById("loader");
const resultsCard = document.getElementById("results-card");
const audioPlayer = document.getElementById("audio-player");
const downloadLink = document.getElementById("download-link");
const ssmlCode = document.getElementById("ssml-code");
const historySection = document.getElementById("history-section");
const historyList = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const systemStatusDot = document.getElementById("system-status-dot");
const systemStatusText = document.getElementById("system-status-text");

// ── State ───────────────────────────────────────────────────────────────────
let history = JSON.parse(localStorage.getItem("empathy_history") || "[]");

// ── Helper Utilities ────────────────────────────────────────────────────────
function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : "";
}

function showError(msg) {
  if (!errorBanner) return;
  errorBanner.textContent = msg;
  errorBanner.style.display = "block";
}

function hideError() {
  if (!errorBanner) return;
  errorBanner.style.display = "none";
}

function escapeHtml(unsafe) {
  return (unsafe || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ── Initial Setup & Lifecycle ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initVoices();
  initPresets();
  initTextInput();
  renderHistory();
  initHistoryControls();
});

// ── Health Check ────────────────────────────────────────────────────────────
async function initHealthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      const data = await res.json();
      if (systemStatusDot) systemStatusDot.style.backgroundColor = "var(--success)";
      if (systemStatusText) {
        systemStatusText.textContent = data.elevenlabs_configured
          ? "Online • ElevenLabs Ready"
          : "Online • Offline Engine Ready";
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch {
    if (systemStatusDot) systemStatusDot.style.backgroundColor = "var(--warning)";
    if (systemStatusText) systemStatusText.textContent = "Connecting to Backend...";
  }
}

// ── Voice Catalog Loading ───────────────────────────────────────────────────
async function initVoices() {
  try {
    const resp = await fetch(`${API_BASE}/api/voices`);
    if (!resp.ok) throw new Error("Could not load voices");
    const data = await resp.json();

    voiceSelect.innerHTML = "";
    data.voices.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = `${v.name} (${capitalize(v.gender)}) — ${v.description}`;
      if (v.id === data.default_voice_id) opt.selected = true;
      voiceSelect.appendChild(opt);
    });
  } catch (err) {
    voiceSelect.innerHTML = '<option value="">Default Voice</option>';
  }
}

// ── Presets Handling ────────────────────────────────────────────────────────
function initPresets() {
  const chips = document.querySelectorAll(".preset-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const text = chip.getAttribute("data-text");
      if (text && textInput) {
        textInput.value = text;
        updateCharCount();
        textInput.focus();
      }
    });
  });
}

function initTextInput() {
  if (!textInput) return;
  textInput.addEventListener("input", updateCharCount);
  updateCharCount();
}

function updateCharCount() {
  if (!textInput || !charCount) return;
  const count = textInput.value.length;
  charCount.textContent = `${count} / 2000`;
}

// ── Generation Pipeline ─────────────────────────────────────────────────────
if (generateBtn) {
  generateBtn.addEventListener("click", async () => {
    const text = textInput ? textInput.value.trim() : "";
    if (!text) {
      showError("Please enter or select some text to synthesize.");
      return;
    }

    hideError();
    generateBtn.disabled = true;
    if (loader) loader.style.display = "block";
    if (resultsCard) resultsCard.style.display = "none";

    try {
      const payload = {
        text,
        voice_id: voiceSelect ? voiceSelect.value || undefined : undefined,
        output_format: formatSelect ? formatSelect.value : "mp3",
      };

      const resp = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      renderResult(data, text);

      saveHistory({
        text,
        primary: data.primary_emotion || data.emotion,
        secondary: data.secondary_emotion,
        blend: data.blended_emotion,
        audioUrl: data.audio_url,
        voiceName: data.voice_name,
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      showError(`Synthesis Error: ${err.message}`);
    } finally {
      generateBtn.disabled = false;
      if (loader) loader.style.display = "none";
    }
  });
}

// ── Render Results ──────────────────────────────────────────────────────────
function renderResult(data, text) {
  const primary = data.primary_emotion || data.emotion;
  const secondary = data.secondary_emotion;
  const blend = data.blended_emotion;

  // Primary Emotion
  const primaryIcon = document.getElementById("primary-icon");
  const primaryName = document.getElementById("primary-name");
  const primaryScore = document.getElementById("primary-score");
  if (primaryIcon) primaryIcon.textContent = primary.emoji || "😐";
  if (primaryName) primaryName.textContent = capitalize(primary.label);
  if (primaryScore) primaryScore.textContent = `${Math.round(primary.score * 100)}% Confidence`;

  // Secondary Emotion
  const secondaryCard = document.getElementById("secondary-card");
  const secondaryIcon = document.getElementById("secondary-icon");
  const secondaryName = document.getElementById("secondary-name");
  const secondaryScore = document.getElementById("secondary-score");

  if (blend && blend.is_blended && secondary && secondary.score > 0) {
    if (secondaryCard) secondaryCard.style.display = "flex";
    if (secondaryIcon) secondaryIcon.textContent = secondary.emoji || "😐";
    if (secondaryName) secondaryName.textContent = capitalize(secondary.label);
    if (secondaryScore) secondaryScore.textContent = `${Math.round(secondary.score * 100)}% Confidence`;
  } else {
    if (secondaryCard) secondaryCard.style.display = "none";
  }

  // Blend Details
  const blendTitle = document.getElementById("blend-title");
  const blendText = document.getElementById("blend-text");
  const blendDesc = document.getElementById("blend-desc");

  if (blend) {
    if (blendTitle) blendTitle.textContent = blend.is_blended ? "Blended Expression" : "Dominant Tone";
    if (blendText) blendText.textContent = `${blend.emoji} ${capitalize(blend.label)}`;
    if (blendDesc) blendDesc.textContent = blend.description || "";
  }

  // Metadata pills
  const metaEngine = document.getElementById("meta-engine");
  const metaVoice = document.getElementById("meta-voice");
  const metaIntensity = document.getElementById("meta-intensity");

  if (metaEngine) metaEngine.textContent = data.tts_engine_used.toUpperCase();
  if (metaVoice) metaVoice.textContent = data.voice_name;
  if (metaIntensity) metaIntensity.textContent = capitalize(primary.intensity);

  // Audio Player & Download
  const fullAudioUrl = `${API_BASE}${data.audio_url}`;
  if (audioPlayer) {
    audioPlayer.src = fullAudioUrl;
    audioPlayer.play().catch(() => {});
  }

  if (downloadLink) {
    downloadLink.href = fullAudioUrl;
    downloadLink.download = data.audio_filename || "speech.mp3";
  }

  // SSML Preview
  if (ssmlCode) {
    ssmlCode.textContent = data.ssml_preview || "";
  }

  if (resultsCard) resultsCard.style.display = "block";
}

// ── History Management ──────────────────────────────────────────────────────
function saveHistory(entry) {
  history.unshift(entry);
  if (history.length > APP_CONFIG.maxHistoryItems) {
    history = history.slice(0, APP_CONFIG.maxHistoryItems);
  }
  localStorage.setItem("empathy_history", JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  if (!historySection || !historyList) return;

  if (history.length === 0) {
    historySection.style.display = "none";
    return;
  }

  historySection.style.display = "block";
  historyList.innerHTML = history
    .map((item, idx) => {
      const emoji = item.blend?.emoji || item.primary?.emoji || "🎙️";
      const label = item.blend?.is_blended ? item.blend.label : item.primary?.label || "Generated Audio";
      return `
      <div class="history-item" data-idx="${idx}">
        <span class="history-icon">${emoji}</span>
        <div class="history-text">${escapeHtml(item.text)}</div>
        <span class="history-tag">${capitalize(label)}</span>
      </div>
    `;
    })
    .join("");

  // Attach click events
  historyList.querySelectorAll(".history-item").forEach((el) => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.getAttribute("data-idx"), 10);
      replayHistory(idx);
    });
  });
}

function replayHistory(idx) {
  const item = history[idx];
  if (item && item.audioUrl && audioPlayer) {
    audioPlayer.src = `${API_BASE}${item.audioUrl}`;
    audioPlayer.play().catch(() => {});
    if (resultsCard) resultsCard.style.display = "block";
  }
}

function initHistoryControls() {
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", () => {
      history = [];
      localStorage.removeItem("empathy_history");
      renderHistory();
    });
  }
}
