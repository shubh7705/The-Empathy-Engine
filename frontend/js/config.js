/**
 * config.js — Frontend configuration and API Base URL resolver.
 */

// If frontend is hosted directly on FastAPI (port 8000), use relative origin.
// If served from a separate static server (port 3000, 5500, etc.), default to http://127.0.0.1:8000
const isStandaloneDev = window.location.protocol === "file:" ||
  (window.location.port !== "8000" && window.location.port !== "" && !window.location.pathname.startsWith("/api"));

export const API_BASE = isStandaloneDev
  ? (localStorage.getItem("empathy_api_url") || "http://127.0.0.1:8000")
  : "";

export const APP_CONFIG = {
  appName: "The Empathy Engine",
  version: "2.0.0",
  maxHistoryItems: 10,
};
