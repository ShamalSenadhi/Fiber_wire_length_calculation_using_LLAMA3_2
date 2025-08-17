import streamlit as st
import ollama
import re
import io
import os
from PIL import Image
import subprocess
import time
import threading
import requests

# Page configuration
st.set_page_config(
    page_title="Fiber Wire Length Calculator",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'ollama_running' not in st.session_state:
    st.session_state.ollama_running = False
if 'model_pulled' not in st.session_state:
    st.session_state.model_pulled = False

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Start Ollama service in background"""
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Wait for service to start
        return True
    except Exception as e:
        st.error(f"Failed to start Ollama: {e}")
        return False

def pull_model():
    """Pull the LLAMA3.2 vision model"""
    try:
        result = subprocess.run(
            ["ollama", "pull", "llama3.2-vision:11b"], 
            capture_output=True, 
            text=True, 
            timeout=600
        )
        return result.returncode == 0
    except Exception as e:
        st.error(f"Failed to pull model: {e}")
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract a number from image bytes using Ollama and LLAMA3.2 vision model"""
    try:
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Extract the handwritten number in meters from this image. Only return the numerical value.',
                'images': [image_bytes]
            }]
        )
        
        content = response['message']['content']
        st.write(f"**Raw model output for {image_name}:**")
        st.write(content)
        
        # Use regular expression to find a numerical value
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m|meters)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {e}")
        return None

def main():
    st.title("📏 Fiber Wire Length Calculator")
    st.markdown("### Using LLAMA3.2 Vision Model with GPU Support")
    
    # Sidebar for model management
    with st.sidebar:
        st.header("🔧 Model Setup")
        
        # Check Ollama status
        if st.button("Check Ollama Status"):
            if check_ollama_running():
                st.success("✅ Ollama is running")
                st.session_state.ollama_running = True
            else:
                st.error("❌ Ollama is not running")
                st.session_state.ollama_running = False
        
        # Start Ollama
        if st.button("Start Ollama Service"):
            with st.spinner("Starting Ollama..."):
                if start_ollama():
                    st.success("✅ Ollama started successfully")
                    st.session_state.ollama_running = True
                else:
                    st.error("❌ Failed to start Ollama")
        
        # Pull model
        if st.button("Pull LLAMA3.2 Vision Model"):
            if not st.session_state.ollama_running:
                st.error("Please start Ollama service first")
            else:
                with st.spinner("Pulling model... This may take several minutes"):
                    if pull_model():
                        st.success("✅ Model pulled successfully")
                        st.session_state.model_pulled = True
                    else:
                        st.error("❌ Failed to pull model")
        
        # Status indicators
        st.markdown("---")
        st.markdown("**Status:**")
        if st.session_state.ollama_running:
            st.success("🟢 Ollama Running")
        else:
            st.error("🔴 Ollama Stopped")
            
        if st.session_state.model_pulled:
            st.success("🟢 Model Ready")
        else:
            st.warning("🟡 Model Not Loaded")
    
    # Main application
    if not st.session_state.ollama_running:
        st.warning("⚠️ Please start Ollama service from the sidebar first")
        st.stop()
    
    if not st.session_state.model_pulled:
        st.warning("⚠️ Please pull the LLAMA3.2 vision model from the sidebar first")
        st.stop()
    
    # File upload section
    st.markdown("## 📤 Upload Fiber Length Images")
    
    tab1, tab2 = st.tabs(["First Pair", "Second Pair"])
    
    with tab1:
        st.markdown("### Upload First Two Images")
        uploaded_files_1 = st.file_uploader(
            "Choose first two images", 
            accept_multiple_files=True, 
            type=['png', 'jpg', 'jpeg'],
            key="first_pair"
        )
        
        if uploaded_files_1 and len(uploaded_files_1) == 2:
            col1, col2 = st.columns(2)
            
            extracted_lengths_1 = {}
            
            for i, uploaded_file in enumerate(uploaded_files_1):
                with col1 if i == 0 else col2:
                    st.image(uploaded_file, caption=uploaded_file.name, width=300)
                    
                    if st.button(f"Process {uploaded_file.name}", key=f"process1_{i}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            image_bytes = uploaded_file.read()
                            length = extract_number_from_image_bytes(image_bytes, uploaded_file.name)
                            if length is not None:
                                extracted_lengths_1[uploaded_file.name] = length
                                st.success(f"Extracted length: {length} meters")
            
            # Calculate difference for first pair
            if len(extracted_lengths_1) == 2:
                lengths = list(extracted_lengths_1.values())
                difference = abs(lengths[0] - lengths[1])
                st.markdown("---")
                st.success(f"**Fiber length difference (First Pair): {difference} meters**")
        
        elif uploaded_files_1 and len(uploaded_files_1) != 2:
            st.error("Please upload exactly 2 images")
    
    with tab2:
        st.markdown("### Upload Second Two Images")
        uploaded_files_2 = st.file_uploader(
            "Choose second two images", 
            accept_multiple_files=True, 
            type=['png', 'jpg', 'jpeg'],
            key="second_pair"
        )
        
        if uploaded_files_2 and len(uploaded_files_2) == 2:
            col1, col2 = st.columns(2)
            
            extracted_lengths_2 = {}
            
            for i, uploaded_file in enumerate(uploaded_files_2):
                with col1 if i == 0 else col2:
                    st.image(uploaded_file, caption=uploaded_file.name, width=300)
                    
                    if st.button(f"Process {uploaded_file.name}", key=f"process2_{i}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            image_bytes = uploaded_file.read()
                            length = extract_number_from_image_bytes(image_bytes, uploaded_file.name)
                            if length is not None:
                                extracted_lengths_2[uploaded_file.name] = length
                                st.success(f"Extracted length: {length} meters")
            
            # Calculate difference for second pair
            if len(extracted_lengths_2) == 2:
                lengths = list(extracted_lengths_2.values())
                difference = abs(lengths[0] - lengths[1])
                st.markdown("---")
                st.success(f"**Fiber length difference (Second Pair): {difference} meters**")
        
        elif uploaded_files_2 and len(uploaded_files_2) != 2:
            st.error("Please upload exactly 2 images")
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        1. **Setup**: First, start Ollama service and pull the LLAMA3.2 vision model from the sidebar
        2. **Upload**: Upload exactly 2 images per tab containing handwritten fiber lengths
        3. **Process**: Click the process button for each image to extract the length
        4. **Results**: The app will calculate and display the difference between the two lengths
        
        **Requirements:**
        - Images should contain handwritten numbers representing fiber lengths in meters
        - Supported formats: PNG, JPG, JPEG
        - GPU acceleration will be used automatically if available
        """)

if __name__ == "__main__":
    main()
