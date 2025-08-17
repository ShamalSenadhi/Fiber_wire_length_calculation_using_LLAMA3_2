import streamlit as st
import ollama
import re
from PIL import Image
import io
import time
import subprocess
import json
import psutil

# Configure Streamlit page
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📏 Handwritten Fiber Length Analyzer")
st.markdown("""
This application uses Ollama's llama3.2-vision:11b model to extract handwritten fiber lengths from images 
and calculate the difference between two measurements.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["llama3.2-vision:11b", "llava:latest", "bakllava:latest"],
    index=0
)

# Check if Ollama is running and GPU status
@st.cache_data
def check_ollama_connection():
    try:
        response = ollama.list()
        return True, "Connected to Ollama successfully"
    except Exception as e:
        return False, f"Failed to connect to Ollama: {str(e)}"

@st.cache_data
def check_gpu_status():
    """Check GPU availability and CUDA status"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_memory': [],
        'driver_version': 'Not available'
    }
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info['cuda_available'] = True
            lines = result.stdout.strip().split('\n')
            gpu_info['gpu_count'] = len(lines)
            
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_info['gpu_memory'].append({
                        'name': parts[0],
                        'total': int(parts[1]),
                        'used': int(parts[2]),
                        'free': int(parts[3])
                    })
        
        # Get driver version
        driver_result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                     capture_output=True, text=True, timeout=5)
        if driver_result.returncode == 0:
            gpu_info['driver_version'] = driver_result.stdout.strip()
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return gpu_info

def get_ollama_gpu_config():
    """Get Ollama GPU configuration"""
    try:
        # Check if Ollama is using GPU by examining running processes
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
        return "No models currently loaded"
    except Exception as e:
        return f"Error checking Ollama status: {str(e)}"

# Function to extract number from image with GPU acceleration
def extract_number_from_image(image_bytes, model=model_name):
    """Extract handwritten number from image using Ollama vision model with GPU acceleration"""
    try:
        with st.spinner("🔥 Analyzing image with GPU acceleration..."):
            start_time = time.time()
            
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': 'Extract the handwritten number in meters from this image. Only return the numerical value.',
                    'images': [image_bytes]
                }],
                options={
                    'num_gpu': -1,  # Use all available GPUs
                    'num_thread': 8,  # Optimize CPU threads
                    'temperature': 0.1,  # Lower temperature for more consistent results
                }
            )
            
            processing_time = time.time() - start_time
            
        content = response['message']['content']
        st.write(f"**Raw model output:** {content}")
        st.write(f"**⚡ Processing time:** {processing_time:.2f} seconds")
        
        # Use regex to find numerical value
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m(?:eters?)?)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning("No numerical value found in the response")
            return None
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Check Ollama connection status
connection_status, connection_msg = check_ollama_connection()
if connection_status:
    st.sidebar.success(connection_msg)
else:
    st.sidebar.error(connection_msg)
    st.error("Please ensure Ollama is running and the model is available.")
    st.stop()

# GPU Status Section
st.sidebar.header("🎮 GPU Status")
gpu_info = check_gpu_status()

if gpu_info['cuda_available']:
    st.sidebar.success("✅ CUDA GPU Available")
    st.sidebar.write(f"**GPU Count:** {gpu_info['gpu_count']}")
    st.sidebar.write(f"**Driver Version:** {gpu_info['driver_version']}")
    
    # Show GPU memory usage
    for i, gpu in enumerate(gpu_info['gpu_memory']):
        usage_percent = (gpu['used'] / gpu['total']) * 100 if gpu['total'] > 0 else 0
        st.sidebar.write(f"**GPU {i}:** {gpu['name']}")
        st.sidebar.progress(usage_percent / 100)
        st.sidebar.write(f"Memory: {gpu['used']}/{gpu['total']} MB ({usage_percent:.1f}%)")
        
    # Show Ollama GPU status
    ollama_status = get_ollama_gpu_config()
    with st.sidebar.expander("🔧 Ollama GPU Config"):
        st.code(ollama_status, language="text")
        
else:
    st.sidebar.warning("⚠️ No CUDA GPU detected")
    st.sidebar.write("Running on CPU mode")
    
# Performance Settings
st.sidebar.header("⚡ Performance")
use_gpu_acceleration = st.sidebar.checkbox("Use GPU Acceleration", value=gpu_info['cuda_available'])
batch_processing = st.sidebar.checkbox("Enable Batch Processing", value=False)

if not gpu_info['cuda_available'] and use_gpu_acceleration:
    st.sidebar.error("GPU acceleration requested but no CUDA GPU available")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Image 1")
    uploaded_file1 = st.file_uploader(
        "Upload first fiber length image",
        type=['png', 'jpg', 'jpeg'],
        key="image1"
    )
    
    if uploaded_file1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="First Image", use_column_width=True)

with col2:
    st.header("📸 Image 2")
    uploaded_file2 = st.file_uploader(
        "Upload second fiber length image",
        type=['png', 'jpg', 'jpeg'],
        key="image2"
    )
    
    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="Second Image", use_column_width=True)

# Process images when both are uploaded
if uploaded_file1 and uploaded_file2:
    if st.button("🔍 Analyze Images with GPU", type="primary"):
        st.header("📊 Analysis Results")
        
        # Show GPU utilization info
        if gpu_info['cuda_available'] and use_gpu_acceleration:
            st.info("🔥 Using GPU acceleration for faster processing!")
        else:
            st.info("💻 Using CPU processing")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start total processing timer
        total_start_time = time.time()
        
        # Process first image
        status_text.text("🔥 Processing first image with GPU...")
        progress_bar.progress(25)
        image1_bytes = uploaded_file1.getvalue()
        num1 = extract_number_from_image(image1_bytes, model_name)
        
        progress_bar.progress(50)
        
        # Process second image
        status_text.text("🔥 Processing second image with GPU...")
        progress_bar.progress(75)
        image2_bytes = uploaded_file2.getvalue()
        num2 = extract_number_from_image(image2_bytes, model_name)
        
        # Complete processing
        progress_bar.progress(100)
        total_time = time.time() - total_start_time
        status_text.text(f"✅ Analysis complete! Total time: {total_time:.2f}s")
        
        # Display performance metrics
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("⏱️ Total Processing Time", f"{total_time:.2f}s")
        with col_perf2:
            st.metric("🚀 Processing Mode", "GPU" if (gpu_info['cuda_available'] and use_gpu_acceleration) else "CPU")
        with col_perf3:
            avg_time = total_time / 2
            st.metric("📊 Avg Time/Image", f"{avg_time:.2f}s")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if num1 is not None:
                st.metric("Image 1 Length", f"{num1} m")
            else:
                st.error("Could not extract value from Image 1")
        
        with col2:
            if num2 is not None:
                st.metric("Image 2 Length", f"{num2} m")
            else:
                st.error("Could not extract value from Image 2")
        
        with col3:
            if num1 is not None and num2 is not None:
                difference = abs(num1 - num2)
                st.metric("Difference", f"{difference} m")
                
                # Additional analysis
                st.subheader("📈 Analysis Summary")
                larger_value = max(num1, num2)
                smaller_value = min(num1, num2)
                percentage_diff = (difference / larger_value) * 100 if larger_value > 0 else 0
                
                st.write(f"**Larger value:** {larger_value} m")
                st.write(f"**Smaller value:** {smaller_value} m")
                st.write(f"**Percentage difference:** {percentage_diff:.2f}%")
                
                if percentage_diff < 5:
                    st.success("✅ Values are very close (< 5% difference)")
                elif percentage_diff < 15:
                    st.warning("⚠️ Moderate difference (5-15%)")
                else:
                    st.error("❌ Significant difference (> 15%)")
            else:
                st.error("Cannot calculate difference due to missing values")

# Instructions section
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. **GPU Setup**: Ensure CUDA drivers are installed
2. **Start Ollama**: Ensure Ollama service is running with GPU support
3. **Model Ready**: Verify llama3.2-vision:11b is available
4. **Upload Images**: Upload two images with handwritten fiber lengths
5. **GPU Acceleration**: Enable GPU acceleration if available
6. **Analyze**: Click the analyze button to process with GPU
7. **View Results**: See extracted values and performance metrics

**Supported formats:** PNG, JPG, JPEG

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- CUDA drivers installed
- Sufficient GPU memory (>4GB recommended)
""")

# Performance Tips
with st.sidebar.expander("💡 Performance Tips"):
    st.markdown("""
    - **GPU Memory**: Ensure sufficient GPU memory
    - **Image Size**: Smaller images process faster
    - **Model Choice**: llama3.2-vision:11b is optimized for GPU
    - **Concurrent Processing**: Avoid running multiple analyses simultaneously
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit and Ollama | "
    f"Model: {model_name} | "
    f"Mode: {'🔥 GPU Accelerated' if gpu_info['cuda_available'] else '💻 CPU'}"
)
