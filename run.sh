#!/bin/bash
# Start Zettlecast services

set -e

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create logs directory
mkdir -p logs

echo "🧠 Starting Zettlecast..."

# Start Ollama if not running
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama..."
        ollama serve &> logs/ollama.log &
        sleep 2
    fi
fi

# Start FastAPI
echo "Starting API server on port ${API_PORT:-8000}..."
uvicorn zettlecast.main:app --host 0.0.0.0 --port ${API_PORT:-8000} > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:${API_PORT:-8000}/health > /dev/null 2>&1; then
        echo "✅ API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ API failed to start. Check logs/api.log for details"
        cat logs/api.log
        kill $API_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Start Next.js frontend (or Streamlit as fallback)
if [ -d "frontend" ] && [ -f "frontend/.env.local" ]; then
    echo "Starting Next.js frontend on port 3000..."
    cd frontend
    npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    sleep 3
    
    echo ""
    echo "✅ Zettlecast running!"
    echo "   API:      http://localhost:${API_PORT:-8000}"
    echo "   Frontend: http://localhost:3000"
    echo ""
    echo "Logs are in ./logs/"
    echo "Press Ctrl+C to stop"
    
    # Handle shutdown
    trap "echo 'Shutting down...'; kill $API_PID $FRONTEND_PID 2>/dev/null; exit" SIGINT SIGTERM
else
    # Fallback to Streamlit
    echo "Starting Streamlit UI on port ${UI_PORT:-8501}..."
    streamlit run src/zettlecast/ui/app.py --server.port ${UI_PORT:-8501} --server.headless true > logs/ui.log 2>&1 &
    UI_PID=$!
    
    sleep 2
    
    echo ""
    echo "✅ Zettlecast running!"
    echo "   API: http://localhost:${API_PORT:-8000}"
    echo "   UI:  http://localhost:${UI_PORT:-8501}"
    echo ""
    echo "Logs are in ./logs/"
    echo "Press Ctrl+C to stop"
    
    # Handle shutdown
    trap "echo 'Shutting down...'; kill $API_PID $UI_PID 2>/dev/null; exit" SIGINT SIGTERM
fi

# Wait
wait
