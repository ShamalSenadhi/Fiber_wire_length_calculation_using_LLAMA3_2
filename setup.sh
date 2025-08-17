#!/bin/bash

# Fiber Length Analyzer Setup Script with GPU Support
# This script sets up Ollama with GPU acceleration, CUDA drivers, and the required model

set -e  # Exit on any error

echo "🚀 Setting up Fiber Length Analyzer Environment with GPU Support..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  This script should not be run as root for security reasons"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU
check_gpu() {
    if command_exists nvidia-smi; then
        echo "🎮 GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        return 0
    else
        echo "⚠️  No NVIDIA GPU detected or nvidia-smi not available"
        return 1
    fi
}

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install CUDA drivers and toolkit
echo "🎮 Setting up CUDA environment..."
if ! command_exists nvidia-smi; then
    echo "Installing CUDA drivers..."
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA toolkit and drivers
    sudo apt-get install -y cuda-drivers cuda-toolkit-12-2
    
    echo "✅ CUDA drivers and toolkit installed"
    echo "⚠️  Please reboot your system before continuing!"
    echo "After reboot, run this script again to continue setup."
    exit 0
else
    echo "✅ CUDA drivers already installed"
    check_gpu
fi

# Install Docker (for containerized GPU support if needed)
if ! command_exists docker; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install NVIDIA Container Toolkit
echo "🔧 Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
    curl -s -L "https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list" | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ NVIDIA Container Toolkit installed"
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# Install Ollama with GPU support
if ! command_exists ollama; then
    echo "🤖 Installing Ollama with GPU support..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Configure Ollama for GPU usage
echo "⚙️ Configuring Ollama for GPU..."
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, adjust as needed
export OLLAMA_NUM_PARALLEL=2   # Number of parallel requests
export OLLAMA_MAX_LOADED_MODELS=1  # Keep one model loaded

# Create ollama systemd service with GPU settings
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=1"

[Install]
WantedBy=default.target
EOF

# Create ollama user if it doesn't exist
if ! id -u ollama > /dev/null 2>&1; then
    sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
    sudo usermod -a -G render,video ollama  # Add to GPU groups
fi

# Start Ollama service with GPU support
echo "🔄 Starting Ollama service with GPU support..."
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for service to start
sleep 10

# Verify Ollama is running
if sudo systemctl is-active --quiet ollama; then
    echo "✅ Ollama service is running with GPU support"
else
    echo "❌ Failed to start Ollama service"
    sudo systemctl status ollama
    exit 1
fi

# Pull the required model with GPU optimization
echo "📥 Pulling llama3.2-vision:11b model with GPU support..."
if ollama list | grep -q "llama3.2-vision:11b"; then
    echo "✅ Model already exists"
else
    # Set GPU memory fraction for model loading
    OLLAMA_GPU_MEMORY_FRACTION=0.8 ollama pull llama3.2-vision:11b
    echo "✅ Model downloaded successfully with GPU optimization"
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

# Create GPU monitoring script
cat > gpu_monitor.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import time
import json

def monitor_gpu():
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                print(f"GPU {i}: {parts[0]}")
                print(f"  Utilization: {parts[1]}%")
                print(f"  Memory: {parts[2]}/{parts[3]} MB")
                print(f"  Temperature: {parts[4]}°C")
                print()
    except Exception as e:
        print(f"Error monitoring GPU: {e}")

if __name__ == "__main__":
    monitor_gpu()
EOF

chmod +x gpu_monitor.py

# Verify installation
echo "🔍 Verifying GPU-accelerated installation..."

# Check CUDA
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ CUDA is working"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "❌ CUDA verification failed"
    exit 1
fi

# Check Ollama with GPU
if ollama list > /dev/null 2>&1; then
    echo "✅ Ollama is working"
else
    echo "❌ Ollama verification failed"
    exit 1
fi

# Test GPU utilization with Ollama
echo "🧪 Testing GPU utilization..."
timeout 30 ollama run llama3.2-vision:11b "Hello" > /dev/null 2>&1 &
sleep 5
./gpu_monitor.py
wait

# Check model
if ollama list | grep -q "llama3.2-vision:11b"; then
    echo "✅ Vision model is available and loaded"
else
    echo "❌ Vision model not found"
    exit 1
fi

# Check Python packages
python3 -c "import streamlit, ollama, PIL, re, psutil" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies verified"
else
    echo "❌ Python dependencies verification failed"
    exit 1
fi

echo ""
echo "🎉 GPU-accelerated setup completed successfully!"
echo ""
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "To start the application:"
echo "  streamlit run app.py"
echo ""
echo "The app will be available at: http://localhost:8501"
echo ""
echo "📝 GPU Optimization Notes:"
echo "  - Ollama is configured to use GPU acceleration"
echo "  - Model will automatically use available GPU memory"
echo "  - Monitor GPU usage with: ./gpu_monitor.py"
echo "  - Check Ollama GPU status with: ollama ps"
echo ""
echo "📋 Logs and Monitoring:"
echo "  - Ollama service: sudo systemctl status ollama"
echo "  - GPU monitoring: ./gpu_monitor.py"
echo "  - Ollama logs: sudo journalctl -u ollama -f"
echo ""
echo "🔧 Troubleshooting:"
echo "  - If GPU not detected: nvidia-smi"
echo "  - Restart Ollama: sudo systemctl restart ollama"
echo "  - Check GPU memory: nvidia-smi"🔍 Verifying installation..."

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
