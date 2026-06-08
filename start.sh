#!/bin/bash
# Quick-start script for local dev on macOS / Linux
set -e

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit as needed"
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment (Python 3.11)…"
  python3.11 -m venv venv
fi

source venv/bin/activate
pip install --quiet -r requirements.txt

echo "Starting ALPR OSS API on port ${PORT:-3000}…"
python main.py
