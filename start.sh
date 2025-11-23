#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $(jobs -p)
    exit
}

trap cleanup SIGINT SIGTERM

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirement.txt
else
    source venv/bin/activate
fi

# Start Backend (Training)
echo "Starting AI Training (Backend)..."
python main.py train &
BACKEND_PID=$!

# Start Frontend (Dashboard)
echo "Starting Dashboard (Frontend)..."
cd dashboard-ui
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
