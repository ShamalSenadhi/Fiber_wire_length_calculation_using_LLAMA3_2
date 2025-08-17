#!/bin/bash

echo "🚀 Setting up Fiber Length Extractor Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install Python requirements
echo "📦 Installing Python requirements..."
pip3 install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "🔧 Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama is already installed"
fi

# Start Ollama service (if not already running)
echo "🔄 Starting Ollama service..."
ollama serve &
sleep 5

# Pull the required model
echo "📥 Pulling llama3.2-vision:11b model (this may take a while)..."
ollama pull llama3.2-vision:11b

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 To run the application:"
echo "   streamlit run app.py"
echo ""
echo "📝 Make sure Ollama service is running before starting the app."
echo "   You can check with: ollama list"
echo ""
