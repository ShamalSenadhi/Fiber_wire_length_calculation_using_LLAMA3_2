#!/bin/bash

# Fiber Length Analyzer Setup Script
# This script sets up Ollama, CUDA drivers, and the required model for the Streamlit app

set -e  # Exit on any error

echo "🚀 Setting up Fiber Length Analyzer Environment..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  This script should not be run as root for security reasons"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install CUDA drivers if not already installed
if ! command_exists nvidia-smi; then
    echo "🎮 Installing CUDA drivers..."
    sudo apt-get install -y cuda-drivers
    echo "✅ CUDA drivers installed"
else
    echo "✅ CUDA drivers already installed"
fi

# Install Ollama if not already installed
if ! command_exists ollama; then
    echo "🤖 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama service is already running"
else
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5  # Wait for service to start
    echo "✅ Ollama service started"
fi

# Pull the required model
echo "📥 Pulling llama3.2-vision:11b model..."
if ollama list | grep -q "llama3.2-vision:11b"; then
    echo "✅ Model already exists"
else
    ollama pull llama3.2-vision:11b
    echo "✅ Model downloaded successfully"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command_exists pip; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "❌ pip not found. Please install Python and pip first"
    exit 1
fi

# Verify installation
echo "🔍 Verifying installation..."

# Check Ollama
if ollama list > /dev/null 2>&1; then
    echo "✅ Ollama is working"
else
    echo "❌ Ollama verification failed"
    exit 1
fi

# Check model
if ollama list | grep -q "llama3.2-vision:11b"; then
    echo "✅ Vision model is available"
else
    echo "❌ Vision model not found"
    exit 1
fi

# Check Python packages
python3 -c "import streamlit, ollama, PIL, re" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies verified"
else
    echo "❌ Python dependencies verification failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "  streamlit run app.py"
echo ""
echo "The app will be available at: http://localhost:8501"
echo ""
echo "📝 Note: Make sure to:"
echo "  1. Upload two images with handwritten fiber lengths"
echo "  2. Click 'Analyze Images' to process them"
echo "  3. View the results and calculated difference"
echo ""
echo "📋 Logs:"
echo "  - Ollama service log: ollama.log"
echo "  - Check Ollama status: ollama list"
echo "  - Stop Ollama: pkill -f 'ollama serve'"
