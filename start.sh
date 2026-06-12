```bash
#!/bin/bash
set -e

# Check for .env
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — update database credentials if needed"
fi

# Check for venv
if [ ! -d "venv" ]; then
  echo "Creating virtual environment (Python 3.11)…"
  python3.11 -m venv venv
fi

source venv/bin/activate
pip install --quiet -r requirements.txt