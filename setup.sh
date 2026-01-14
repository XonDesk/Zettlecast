#!/bin/bash
# Zettlecast Setup Script (macOS, Linux, WSL)
# Installs all dependencies including Ollama

set -e

echo "🧠 Zettlecast Setup"
echo "==================="

# Detect OS and environment
OS="$(uname -s)"
ARCH="$(uname -m)"

# Check if running in WSL
IS_WSL=false
if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    IS_WSL=true
    echo "Detected: Windows (WSL) - $OS ($ARCH)"
    echo ""
    echo "⚠️  WSL detected!"
    echo "   For best performance on Windows, consider using setup.ps1"
    echo "   This script will continue, but Ollama may have issues in WSL."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Run setup.ps1 in PowerShell instead."
        exit 0
    fi
else
    echo "Detected: $OS ($ARCH)"
fi

# Add Homebrew to PATH if on macOS
if [ "$OS" = "Darwin" ]; then
    if [ -d "/opt/homebrew/bin" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
    elif [ -d "/usr/local/bin" ]; then
        export PATH="/usr/local/bin:$PATH"
    fi
fi

# --- Python Environment ---
echo ""
echo "📦 Setting up Python environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
fi

source .venv/bin/activate
pip install --upgrade pip

# --- Install Zettlecast ---
echo ""
echo "📦 Installing Zettlecast dependencies..."
pip install -e ".[dev]"

# --- Install Ollama ---
echo ""
echo "🦙 Installing Ollama..."

install_ollama() {
    case "$OS" in
        Darwin)
            # macOS
            if command -v brew &> /dev/null; then
                echo "Installing via Homebrew..."
                brew install ollama
            else
                echo "⚠️  Homebrew not found. Installing Ollama manually:"
                echo "   Please download from: https://ollama.com/download/mac"
                echo ""
                echo "   After installing:"
                echo "   1. Open Ollama.app from Applications"
                echo "   2. Run: ollama pull llama3.2:3b"
                echo ""
                echo "   Alternatively, install Homebrew first:"
                echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                return 1
            fi
            ;;
        Linux)
            echo "Installing via curl..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "⚠️  Windows detected. Please install Ollama manually:"
            echo "   Download from: https://ollama.com/download/windows"
            echo ""
            echo "After installing, run: ollama pull llama3.2:3b"
            return 1
            ;;
        *)
            echo "⚠️  Unknown OS. Please install Ollama manually from https://ollama.com"
            return 1
            ;;
    esac
}

if command -v ollama &> /dev/null; then
    echo "✅ Ollama already installed: $(ollama --version)"
elif [ -f "/opt/homebrew/bin/ollama" ] || [ -f "/usr/local/bin/ollama" ]; then
    echo "✅ Ollama already installed (found in system paths)"
else
    if install_ollama; then
        echo "✅ Ollama installed successfully"
    else
        echo "⚠️  Ollama installation requires manual steps (see above)"
    fi
fi

# --- Pull default model ---
echo ""
echo "🤖 Pulling default LLM model (llama3.2:3b)..."

if command -v ollama &> /dev/null; then
    # Start Ollama service if not running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama service..."
        ollama serve &> /dev/null &
        sleep 2
    fi
    
    ollama pull llama3.2:3b
    echo "✅ Model ready"
else
    echo "⚠️  Skipping model pull (Ollama not available)"
fi

# --- Create directories ---
echo ""
echo "📁 Creating storage directories..."
mkdir -p ~/\_BRAIN_STORAGE
mkdir -p ~/\_BRAIN_STORAGE/.lancedb
echo "✅ Created ~/_BRAIN_STORAGE"

# --- Generate config ---
echo ""
echo "⚙️  Generating configuration..."

if [ ! -f ".env" ]; then
    API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    cat > .env << EOF
# Zettlecast Configuration
# Generated: $(date)

# API Security
API_TOKEN=$API_TOKEN

# Storage
STORAGE_PATH=$HOME/_BRAIN_STORAGE
LANCEDB_PATH=$HOME/_BRAIN_STORAGE/.lancedb

# Models
EMBEDDING_MODEL=google/embeddinggemma-300m
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
WHISPER_MODEL=medium

# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b

# Features
ENABLE_CONTEXT_ENRICHMENT=false

# Server
API_PORT=8000
UI_PORT=8501
EOF
    echo "✅ Created .env with secure token"
    echo ""
    echo "🔑 Your API token (save this for bookmarklet):"
    echo "   $API_TOKEN"
else
    echo "✅ .env already exists"
fi

# --- Summary ---
echo ""
echo "========================================="
echo "✅ Setup complete!"
echo ""
echo "To start Zettlecast:"
echo "  source .venv/bin/activate"
echo "  ./run.sh"
echo ""
echo "Or run components separately:"
echo "  uvicorn zettlecast.main:app --port 8000"
echo "  streamlit run src/zettlecast/ui/app.py"
echo "========================================="
