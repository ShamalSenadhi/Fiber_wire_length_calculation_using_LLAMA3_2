import streamlit as st
import ollama
import re
from PIL import Image
import io
import os
import subprocess
import time

# Page configuration
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide"
)

def setup_ollama():
    """Initialize Ollama service and pull the model"""
    try:
        # Check if Ollama is running
        result = subprocess.run(['pgrep', 'ollama'], capture_output=True)
        if result.returncode != 0:
            st.info("Starting Ollama service...")
            # Start Ollama in background
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)  # Wait for service to start
        
        # Pull the model if not exists
        try:
            models = ollama.list()
            model_names = [model['name'] for model in models['models']]
            if 'llama3.2-vision:11b' not in model_names:
                st.info("Downloading llama3.2-vision:11b model... This may take a while.")
                with st.spinner("Pulling model..."):
                    ollama.pull('llama3.2-vision:11b')
                st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error checking/pulling model: {e}")
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
                'content': 'Extract the handwritten number in meters from this image. Return only the numerical value.',
                'images': [image_bytes]
            }]
        )
        
        content = response['message']['content']
        st.write(f"**Raw model output for {image_name}:**")
        st.write(content)
        
        # Use regex to find numerical value
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m| meters)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing image {image_name}: {e}")
        return None

def main():
    st.title("🔍 Fiber Length Analyzer")
    st.markdown("Upload two images with handwritten fiber lengths to calculate the difference.")
    
    # Initialize Ollama
    if 'ollama_ready' not in st.session_state:
        with st.spinner("Setting up Ollama service..."):
            st.session_state.ollama_ready = setup_ollama()
    
    if not st.session_state.ollama_ready:
        st.error("Failed to setup Ollama service. Please check your installation.")
        return
    
    # File upload section
    st.subheader("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose exactly 2 image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        max_files=2
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        st.success(f"✅ Uploaded {len(uploaded_files)} files")
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image 1")
            image1 = Image.open(uploaded_files[0])
            st.image(image1, caption=uploaded_files[0].name, use_column_width=True)
        
        with col2:
            st.subheader("Image 2")
            image2 = Image.open(uploaded_files[1])
            st.image(image2, caption=uploaded_files[1].name, use_column_width=True)
        
        # Process button
        if st.button("🚀 Analyze Images", type="primary"):
            with st.spinner("Processing images with AI model..."):
                # Convert images to bytes
                img1_bytes = uploaded_files[0].getvalue()
                img2_bytes = uploaded_files[1].getvalue()
                
                # Extract numbers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Analysis Results - Image 1")
                    num1 = extract_number_from_image_bytes(img1_bytes, uploaded_files[0].name)
                
                with col2:
                    st.subheader("📊 Analysis Results - Image 2")
                    num2 = extract_number_from_image_bytes(img2_bytes, uploaded_files[1].name)
                
                # Calculate difference
                if num1 is not None and num2 is not None:
                    diff = abs(num1 - num2)
                    
                    st.divider()
                    st.subheader("🎯 Final Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Image 1", f"{num1} meters")
                    with col2:
                        st.metric("Image 2", f"{num2} meters")
                    with col3:
                        st.metric("Difference", f"{diff} meters", delta=f"±{diff}")
                    
                    st.success(f"✅ **Fiber length difference: {diff} meters**")
                else:
                    st.error("❌ Could not calculate difference due to missing number(s).")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning(f"⚠️ Please upload exactly 2 images. Currently uploaded: {len(uploaded_files)}")
    else:
        st.info("👆 Please upload exactly 2 image files to get started.")
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        1. **Upload Images**: Select exactly 2 image files containing handwritten fiber lengths
        2. **Supported Formats**: PNG, JPG, JPEG
        3. **Image Content**: Make sure the handwritten numbers are clearly visible
        4. **Processing**: Click 'Analyze Images' to extract numbers and calculate difference
        5. **Results**: View the extracted values and their difference in meters
        
        **Note**: This app uses the llama3.2-vision:11b model for optical character recognition of handwritten numbers.
        """)

if __name__ == "__main__":
    main()
