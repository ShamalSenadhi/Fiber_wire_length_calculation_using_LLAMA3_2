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
        # Install Ollama if not present
        if not os.path.exists('/usr/local/bin/ollama'):
            st.info("Installing Ollama...")
            subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], 
                         stdout=subprocess.PIPE, check=True)
            install_script = subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], 
                                          capture_output=True, text=True, check=True)
            subprocess.run(['sh', '-c', install_script.stdout], check=True)
        
        # Check if ollama is already running
        result = subprocess.run(['pgrep', 'ollama'], capture_output=True)
        if result.returncode != 0:
            # Start ollama serve in background
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(10)  # Wait longer for service to start
        
        # Check if model exists, if not pull it
        try:
            ollama.show('llama3.2-vision:11b')
        except:
            st.info("Downloading vision model... This may take several minutes.")
            result = subprocess.run(['ollama', 'pull', 'llama3.2-vision:11b'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Failed to download model: {result.stderr}")
                return False
        
        return True
    except Exception as e:
        st.error(f"Error setting up Ollama: {e}")
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
    
    # Setup Ollama
    if 'ollama_ready' not in st.session_state:
        with st.spinner("Setting up Ollama..."):
            st.session_state.ollama_ready = setup_ollama()
    
    if not st.session_state.ollama_ready:
        st.error("Failed to setup Ollama. Please check installation.")
        return
    
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
