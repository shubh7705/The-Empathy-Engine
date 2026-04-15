#!/usr/bin/env python3
"""
run.py -- Convenience launcher for The Empathy Engine.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --reload
"""

import argparse
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="The Empathy Engine -- Launch Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development")
    args = parser.parse_args()

    banner = (
        "\n"
        "  =========================================================\n"
        "   THE EMPATHY ENGINE v2.0 -- Giving AI a Human Voice\n"
        "  =========================================================\n"
        f"  Server   : http://{args.host}:{args.port}\n"
        f"  API Docs : http://{args.host}:{args.port}/docs\n"
        f"  Frontend : http://{args.host}:{args.port}/\n"
        "  =========================================================\n"
    )
    print(banner)

    try:
        import uvicorn
        uvicorn.run(
            "empathy_engine.backend.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except ImportError:
        print("ERROR: uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully.")


if __name__ == "__main__":
    main()
