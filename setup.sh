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

# --- Check Python Version ---
echo ""
echo "📦 Checking Python version..."

PYTHON_CMD=""

# Try python3.11 first (preferred for NeMo compatibility)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Found Python 3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "✅ Found Python 3.12"
elif command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PY_VERSION" == "3.11" ]] || [[ "$PY_VERSION" == "3.12" ]]; then
        PYTHON_CMD="python3"
        echo "✅ Found Python $PY_VERSION"
    else
        echo "⚠️  Found Python $PY_VERSION, but 3.11 or 3.12 is recommended."
        echo "   Python 3.13+ is NOT supported due to NeMo/ML library compatibility."
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Please install Python 3.11:"
            echo "  macOS: brew install python@3.11"
            echo "  Linux: sudo apt install python3.11 python3.11-venv"
            exit 1
        fi
        PYTHON_CMD="python3"
    fi
else
    echo "❌ Python 3 not found!"
    echo "Please install Python 3.11:"
    echo "  macOS: brew install python@3.11"
    echo "  Linux: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# --- Python Environment ---
echo ""
echo "📦 Setting up Python environment..."

if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "Created virtual environment with $PYTHON_CMD"
fi

source .venv/bin/activate
pip install --upgrade pip --quiet

# --- Install Zettlecast ---
echo ""
echo "📦 Installing Zettlecast base dependencies..."
echo "(This may take a few minutes...)"

if ! pip install -e ".[dev]"; then
    echo "❌ Failed to install base dependencies!"
    echo ""
    echo "Common issues:"
    echo "  - PyTorch: May need manual installation from https://pytorch.org"
    echo "  - Build tools: May need to install Xcode Command Line Tools (macOS)"
    exit 1
fi

echo "✅ Base dependencies installed"

# --- Optional: Install Podcast/NeMo dependencies ---
echo ""

# Check if running on Apple Silicon (M-series Mac)
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # Check/install FFmpeg first (required for audio processing)
    # NOTE: torchcodec (used by pyannote.audio) requires FFmpeg 4-7, not 8
    echo "🔊 Checking FFmpeg..."
    if ! command -v ffmpeg &> /dev/null; then
        echo "   FFmpeg not found. Installing via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install ffmpeg@7 && brew link ffmpeg@7
            echo "   ✅ FFmpeg 7 installed"
        else
            echo ""
            echo "   ⚠️  FFmpeg is required but not installed!"
            echo "   Please install Homebrew first: https://brew.sh"
            echo "   Then run: brew install ffmpeg@7 && brew link ffmpeg@7"
            echo ""
        fi
    else
        echo "   ✅ FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
    fi
    echo ""
    
    echo "🎙️  Podcast Transcription Setup (Apple Silicon)"
    echo ""
    echo "   Choose a transcription backend for your M-series Mac:"
    echo ""
    echo "   [1] Parakeet-MLX (Recommended)"
    echo "       - Best accuracy for English podcasts"
    echo "       - Supports speakers diarization"
    echo "       - Optimized for Apple Silicon"
    echo ""
    echo "   [2] MLX-Whisper"
    echo "       - Very fast transcription"
    echo "       - No speaker diarization"
    echo "       - Simpler setup, fewer dependencies"
    echo ""
    echo "   [3] Faster-Whisper (Default)"
    echo "       - Already included in base install"
    echo "       - Good quality, runs on CPU"
    echo ""
    echo "   All options support speaker diarization with pyannote.audio!"
    echo ""
    read -p "Select option (1/2/3, default=3): " TRANSCRIPTION_CHOICE
    
    ASR_BACKEND="whisper"  # Default
    
    case "$TRANSCRIPTION_CHOICE" in
        1)
            echo ""
            echo "📦 Installing Parakeet-MLX + pyannote.audio..."
            echo "(This may take a few minutes...)"
            
            if pip install parakeet-mlx pyannote.audio torchaudio; then
                echo "✅ Parakeet-MLX installed with diarization support"
                ASR_BACKEND="parakeet-mlx"
                
                # Pre-download the parakeet model
                echo ""
                echo "📥 Pre-downloading Parakeet model (2.5GB)..."
                python -c "from parakeet_mlx import from_pretrained; from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')" 2>/dev/null && \
                    echo "   ✅ Parakeet model downloaded" || \
                    echo "   ⚠️  Model will download on first use"
            else
                echo "⚠️  Failed to install Parakeet-MLX."
                echo "   Falling back to faster-whisper (already installed)."
            fi
            ;;
        2)
            echo ""
            echo "📦 Installing MLX-Whisper + pyannote.audio..."
            echo "(This may take a few minutes...)"
            
            if pip install mlx-whisper pyannote.audio torchaudio; then
                echo "✅ MLX-Whisper installed with diarization support"
                ASR_BACKEND="mlx-whisper"
                
                # Pre-download the whisper model
                echo ""
                echo "📥 Pre-downloading Whisper model..."
                python -c "import mlx_whisper; mlx_whisper.transcribe('/dev/null', path_or_hf_repo='mlx-community/whisper-large-v3-turbo')" 2>/dev/null || true
                echo "   ✅ MLX-Whisper model downloaded"
            else
                echo "⚠️  Failed to install MLX-Whisper."
                echo "   Falling back to faster-whisper (already installed)."
            fi
            ;;
        3|"")
            echo ""
            echo "📦 Installing pyannote.audio for diarization..."
            
            if pip install pyannote.audio torchaudio; then
                echo "✅ Faster-Whisper ready with diarization support"
            else
                echo "⚠️  Failed to install pyannote.audio."
                echo "   Faster-Whisper will work without speaker detection."
            fi
            ASR_BACKEND="whisper"
            ;;
        *)
            echo ""
            echo "Invalid choice. Using faster-whisper (default)."
            ASR_BACKEND="whisper"
            ;;
    esac
    
    # Check if HF_TOKEN already exists in .env
    if [ -f ".env" ] && grep -q "^HF_TOKEN=" .env && ! grep -q "^HF_TOKEN=$" .env; then
        echo ""
        echo "   ✅ HuggingFace token found in .env - diarization ready!"
    else
        echo ""
        echo "   ⚠️  For speaker diarization, you'll need a HuggingFace token:"
        echo "   1. Accept the license at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "   2. Add your token to .env: HF_TOKEN=your_token_here"
    fi
    
    # Store the choice for .env generation
    export SELECTED_ASR_BACKEND="$ASR_BACKEND"
    echo ""
    echo "   Selected backend: $ASR_BACKEND"
else
    echo "🎙️  Do you want to install podcast transcription support (NVIDIA NeMo)?"
    echo "   This includes: parakeet-tdt-0.6b-v2 (transcription) + MSDD (diarization)"
    echo "   Requires ~5GB disk space and CUDA GPU recommended."
    read -p "Install podcast support? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📦 Installing NeMo toolkit and podcast dependencies..."
        echo "(This may take 5-10 minutes...)"
        
        if pip install -e ".[podcast]"; then
            echo "✅ Podcast/NeMo dependencies installed"
            export SELECTED_ASR_BACKEND="nemo"
            
            # Apply OS-specific NeMo patch if script exists
            if [ -f "scripts/patch_nemo_${OS}.py" ]; then
                echo ""
                echo "🔧 Applying $OS compatibility patch for NeMo..."
                python scripts/patch_nemo_${OS}.py
            fi
            
            echo ""
            echo "   Enable in .env: USE_NEMO=true"
        else
            echo "⚠️  Failed to install podcast dependencies."
            echo "   You can try again later with: pip install -e '.[podcast]'"
        fi
    fi
fi

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

# --- Configure storage path ---
echo ""
echo "📁 Storage Configuration"
echo ""

# Set default based on OS
if [ "$OS" = "Darwin" ]; then
    DEFAULT_STORAGE="$HOME/_BRAIN_STORAGE"
else
    DEFAULT_STORAGE="$HOME/_BRAIN_STORAGE"
fi

echo "   Where would you like to store your data?"
echo "   Default: $DEFAULT_STORAGE"
read -p "   Storage path (press Enter for default): " CUSTOM_STORAGE

if [ -n "$CUSTOM_STORAGE" ]; then
    STORAGE_PATH="$CUSTOM_STORAGE"
else
    STORAGE_PATH="$DEFAULT_STORAGE"
fi

# Expand ~ if used
STORAGE_PATH="${STORAGE_PATH/#\~/$HOME}"

# Create directories
echo ""
echo "📁 Creating storage directories..."
mkdir -p "$STORAGE_PATH"
mkdir -p "$STORAGE_PATH/.lancedb"
echo "✅ Created $STORAGE_PATH"

# --- Generate config ---
echo ""
echo "⚙️  Generating configuration..."

# Determine USE_NEMO based on backend selection
USE_NEMO="false"
if [ "${SELECTED_ASR_BACKEND:-auto}" = "nemo" ]; then
    USE_NEMO="true"
fi

if [ ! -f ".env" ]; then
    API_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    cat > .env << EOF
# Zettlecast Configuration
# Generated: $(date)

# API Security
API_TOKEN=$API_TOKEN

# Storage
STORAGE_PATH=$STORAGE_PATH
LANCEDB_PATH=$STORAGE_PATH/.lancedb

# Models
EMBEDDING_MODEL=google/embeddinggemma-300m
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
WHISPER_MODEL=large-v3-turbo

# ASR (Automatic Speech Recognition)
ASR_BACKEND=${SELECTED_ASR_BACKEND:-auto}
USE_NEMO=$USE_NEMO

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
    
    # Update ASR_BACKEND in existing .env if user selected a backend
    if [ -n "$SELECTED_ASR_BACKEND" ]; then
        if grep -q "^ASR_BACKEND=" .env; then
            # Update existing ASR_BACKEND line
            sed -i '' "s/^ASR_BACKEND=.*/ASR_BACKEND=$SELECTED_ASR_BACKEND/" .env
            echo "   Updated ASR_BACKEND to: $SELECTED_ASR_BACKEND"
        else
            # Add ASR_BACKEND if not present
            echo "" >> .env
            echo "# ASR (Automatic Speech Recognition)" >> .env
            echo "ASR_BACKEND=$SELECTED_ASR_BACKEND" >> .env
            echo "   Added ASR_BACKEND: $SELECTED_ASR_BACKEND"
        fi
    fi
    
    # Update storage path in existing .env if different
    if [ -n "$STORAGE_PATH" ] && ! grep -q "STORAGE_PATH=$STORAGE_PATH" .env; then
        if grep -q "^STORAGE_PATH=" .env; then
            sed -i '' "s|^STORAGE_PATH=.*|STORAGE_PATH=$STORAGE_PATH|" .env
            sed -i '' "s|^LANCEDB_PATH=.*|LANCEDB_PATH=$STORAGE_PATH/.lancedb|" .env
            echo "   Updated STORAGE_PATH to: $STORAGE_PATH"
        fi
    fi
fi

# --- Summary ---
echo ""
echo "========================================="
echo "✅ Setup complete!"
echo ""
echo "To start Zettlecast:"
echo "  ./run.sh"
echo ""
echo "Or run components separately:"
echo "  source .venv/bin/activate"
echo "  uvicorn zettlecast.main:app --port 8000"
echo "  streamlit run src/zettlecast/ui/app.py"
echo "========================================="
