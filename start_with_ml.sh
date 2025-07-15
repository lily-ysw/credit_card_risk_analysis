#!/bin/bash

# Start Onion Market Dashboard with ML Integration

# Kill any process using port 3000 (frontend) or 9000 (backend API)
for port in 3000 9000; do
  pid=$(lsof -ti tcp:$port)
  if [ -n "$pid" ]; then
    echo "[INFO] Killing process on port $port (PID $pid)"
    kill -9 $pid
  fi
done

# Check for required data file in new location
if [ ! -f backend/data/dark_market_output_v2.csv ]; then
  echo "❌ Error: dark_market_output_v2.csv not found in backend/data directory"
  exit 1
fi

# Set up Python venv for backend if needed
cd backend
if [ ! -d ".venv" ]; then
  echo "[INFO] Creating Python virtual environment in backend/.venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing/updating backend Python dependencies..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi
cd ..

# Start the backend ML API using venv Python
(
  cd backend/api
  ../.venv/bin/python ml_api.py &
)

# Start the frontend (unchanged)
(
  cd frontend
  npm run dev &
)

wait 