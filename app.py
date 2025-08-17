import streamlit as st
import requests
import json
import base64
import re
from PIL import Image
import io
import subprocess
import os
import time

# Page config
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide"
)

def install_ollama():
    """Install Ollama if not already installed"""
    try:
        # Check if ollama is already installed
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    try:
        # Install Ollama
        st.info("Installing Ollama... This may take a few minutes.")
        install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        subprocess.run(install_cmd, shell=True, check=True)
        
        # Start Ollama service
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for service to start
        
        return True
    except Exception as e:
        st.error(f"Failed to install Ollama: {e}")
        return False

def pull_model():
    """Pull the vision model if not already available"""
    try:
        # Check if model exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'llama3.2-vision:11b' in result.stdout:
            return True
        
        st.info("Downloading vision model... This may take several minutes.")
        subprocess.run(['ollama', 'pull', 'llama3.2-vision:11b'], check=True)
        return True
    except Exception as e:
        st.error(f"Failed to pull model: {e}")
        return False

def extract_number_from_image(image_bytes, image_name='uploaded_image'):
    """Extract number from image using Ollama API"""
    try:
        # Convert image bytes to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare the request
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3.2-vision:11b",
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the handwritten number in meters from this image. Return only the numerical value.",
                    "images": [image_b64]
                }
            ],
            "stream": False
        }
        
        # Make the request
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        content = result['message']['content']
        
        st.write(f"**Raw model output for {image_name}:**")
        st.write(content)
        
        # Extract number using regex
        match = re.search(r'(\d+(?:\.\d+)?)', content)
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {e}")
        return None

# Main app
def main():
    st.title("📏 Fiber Length Analyzer")
    st.markdown("Upload two images of handwritten fiber lengths to calculate the difference.")
    
    # Initialize Ollama
    if 'ollama_ready' not in st.session_state:
        with st.spinner("Setting up Ollama..."):
            if install_ollama() and pull_model():
                st.session_state.ollama_ready = True
                st.success("✅ Ollama is ready!")
            else:
                st.error("❌ Failed to setup Ollama")
                st.stop()
    
    # File upload section
    st.header("Upload Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image 1")
        uploaded_file1 = st.file_uploader("Choose first image", type=['png', 'jpg', 'jpeg'], key="file1")
        if uploaded_file1:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Image 1", width=300)
    
    with col2:
        st.subheader("Image 2")
        uploaded_file2 = st.file_uploader("Choose second image", type=['png', 'jpg', 'jpeg'], key="file2")
        if uploaded_file2:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Image 2", width=300)
    
    # Process button
    if st.button("🔍 Analyze Images", type="primary"):
        if uploaded_file1 and uploaded_file2:
            st.header("Analysis Results")
            
            with st.spinner("Extracting numbers from images..."):
                # Reset file pointers
                uploaded_file1.seek(0)
                uploaded_file2.seek(0)
                
                # Extract numbers
                num1 = extract_number_from_image(uploaded_file1.read(), uploaded_file1.name)
                num2 = extract_number_from_image(uploaded_file2.read(), uploaded_file2.name)
                
                if num1 is not None and num2 is not None:
                    diff = abs(num1 - num2)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Image 1 Value", f"{num1} meters")
                    with col2:
                        st.metric("Image 2 Value", f"{num2} meters")
                    with col3:
                        st.metric("Difference", f"{diff} meters")
                    
                    # Success message
                    st.success(f"✅ Fiber length difference: **{diff} meters**")
                else:
                    st.error("❌ Could not extract numbers from one or both images")
        else:
            st.warning("⚠️ Please upload both images before analyzing")

if __name__ == "__main__":
    main()
