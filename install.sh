#!/usr/bin/env bash
set -e

echo "=== Patent Drawing Converter — Install Script ==="
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.9+ is required but was not found."
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ required. Found $PY_VERSION."
    exit 1
fi

echo "Python $PY_VERSION found."

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "Virtual environment already exists, skipping creation."
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip --quiet

# ── Python dependencies ───────────────────────────────────────────────────────
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ── Matplotlib font cache (avoids first-run delay) ───────────────────────────
echo "Building matplotlib font cache..."
python -c "import matplotlib; matplotlib.font_manager._load_fontmanager(try_read_cache=False)" 2>/dev/null || true

# ── .env file ─────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "  -> Edit .env to set SECRET_KEY before running in production."
else
    echo ".env already exists, skipping."
fi

echo ""
echo "=== Install complete ==="
echo ""
echo "To start the development server:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open http://localhost:5050 in your browser."
