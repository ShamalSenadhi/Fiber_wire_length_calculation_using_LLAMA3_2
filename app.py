import streamlit as st
import ollama
import re
from PIL import Image
import io
import subprocess
import os
import time

def setup_ollama():
    """Setup Ollama service and model"""
    try:
        # Install Ollama if not present using direct curl | sh method
        if not os.path.exists('/usr/local/bin/ollama') and not os.path.exists('/usr/bin/ollama'):
            st.info("Installing Ollama... This may take a few minutes.")
            # Use shell=True to properly execute the curl | sh command
            result = subprocess.run('curl -fsSL https://ollama.ai/install.sh | sh', 
                                  shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Failed to install Ollama: {result.stderr}")
                return False
            st.success("Ollama installed successfully!")
        
        # Check if ollama is already running
        result = subprocess.run(['pgrep', '-f', 'ollama serve'], capture_output=True)
        if result.returncode != 0:
            st.info("Starting Ollama service...")
            # Start ollama serve in background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(15)  # Wait longer for service to start
        
        # Test if ollama is responsive
        for i in range(5):
            try:
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    break
                time.sleep(3)
            except subprocess.TimeoutExpired:
                continue
        else:
            st.warning("Ollama service may not be fully ready, but continuing...")
        
        # Check if model exists, if not pull it
        model_exists = False
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if 'llama3.2-vision:11b' in result.stdout:
                model_exists = True
        except:
            pass
            
        if not model_exists:
            st.info("Downloading vision model... This may take several minutes.")
            result = subprocess.run(['ollama', 'pull', 'llama3.2-vision:11b'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Failed to download model: {result.stderr}")
                return False
            st.success("Model downloaded successfully!")
        
        return True
    except Exception as e:
        st.error(f"Error setting up Ollama: {e}")
        st.info("You may need to install Ollama manually: https://ollama.ai/download")
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract number from image using Ollama vision model"""
    try:
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Extract the handwritten number in meters from this image. Just return the numerical value.',
                'images': [image_bytes]
            }]
        )
        
        content = response['message']['content']
        st.write(f"Model output for {image_name}: {content}")
        
        # Extract number using regex
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
    st.title("Fiber Length Difference Calculator")
    st.write("Upload two images with handwritten fiber lengths to calculate the difference")
    
    # Try to setup Ollama
    if 'ollama_ready' not in st.session_state:
        with st.spinner("Setting up Ollama..."):
            st.session_state.ollama_ready = setup_ollama()
    
    if not st.session_state.ollama_ready:
        st.error("⚠️ Ollama setup failed. This app requires Ollama to be installed.")
        st.markdown("""
        ### Manual Setup Instructions:
        1. Install Ollama from https://ollama.ai/download
        2. Run: `ollama serve`
        3. Run: `ollama pull llama3.2-vision:11b`
        4. Restart this app
        
        **Note:** This app currently requires a local Ollama installation and may not work on cloud platforms like Streamlit Cloud.
        """)
        
        # Show a demo interface anyway
        st.subheader("Demo Interface (Non-functional)")
        uploaded_files = st.file_uploader(
            "Upload exactly 2 images with handwritten fiber lengths",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            disabled=True
        )
        return
    
    st.success("✅ Ollama is ready!")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload exactly 2 images with handwritten fiber lengths",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        st.success("2 images uploaded successfully!")
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image 1")
            image1 = Image.open(uploaded_files[0])
            st.image(image1, width=300)
        
        with col2:
            st.subheader("Image 2")
            image2 = Image.open(uploaded_files[1])
            st.image(image2, width=300)
        
        if st.button("Calculate Difference"):
            with st.spinner("Processing images..."):
                # Convert images to bytes
                img1_bytes = uploaded_files[0].getvalue()
                img2_bytes = uploaded_files[1].getvalue()
                
                # Extract numbers
                num1 = extract_number_from_image_bytes(img1_bytes, uploaded_files[0].name)
                num2 = extract_number_from_image_bytes(img2_bytes, uploaded_files[1].name)
                
                if num1 is not None and num2 is not None:
                    diff = abs(num1 - num2)
                    st.success(f"🎯 Fiber length difference: **{diff} meters**")
                    
                    # Display results in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Image 1", f"{num1} m")
                    with col2:
                        st.metric("Image 2", f"{num2} m")
                    with col3:
                        st.metric("Difference", f"{diff} m")
                else:
                    st.error("Could not extract numbers from one or both images.")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning(f"Please upload exactly 2 images. You uploaded {len(uploaded_files)} image(s).")
    
    else:
        st.info("Please upload 2 images to get started.")

if __name__ == "__main__":
    main()
