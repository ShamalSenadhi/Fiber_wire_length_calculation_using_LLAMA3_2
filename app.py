import streamlit as st
import ollama
import re
from PIL import Image
import io
import subprocess
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Fiber Length Difference Calculator",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'ollama_setup' not in st.session_state:
    st.session_state.ollama_setup = False
if 'model_downloaded' not in st.session_state:
    st.session_state.model_downloaded = False

def setup_ollama():
    """Setup Ollama service and download the model"""
    try:
        # Check if Ollama is already running
        try:
            ollama.list()
            st.session_state.ollama_setup = True
            return True
        except:
            pass
        
        # Install Ollama if not present
        with st.spinner("Installing Ollama... This may take a few minutes."):
            result = subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                subprocess.run(['sh'], input=result.stdout, text=True, timeout=300)
        
        # Start Ollama service
        with st.spinner("Starting Ollama service..."):
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(5)  # Wait for service to start
        
        st.session_state.ollama_setup = True
        return True
        
    except Exception as e:
        st.error(f"Failed to setup Ollama: {str(e)}")
        return False

def download_model():
    """Download the vision model"""
    try:
        with st.spinner("Downloading llama3.2-vision:11b model... This may take several minutes."):
            result = subprocess.run(['ollama', 'pull', 'llama3.2-vision:11b'], 
                                  capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                st.session_state.model_downloaded = True
                return True
            else:
                st.error(f"Failed to download model: {result.stderr}")
                return False
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract a number from image bytes using Ollama vision model"""
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
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m|meters)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {str(e)}")
        return None

def main():
    st.title("📏 Fiber Length Difference Calculator")
    st.markdown("Upload two images with handwritten fiber lengths to calculate the difference.")
    
    # Setup section
    st.header("🔧 Setup")
    
    if not st.session_state.ollama_setup:
        if st.button("Setup Ollama Service", type="primary"):
            if setup_ollama():
                st.success("✅ Ollama setup complete!")
                st.rerun()
    else:
        st.success("✅ Ollama service is ready")
        
        if not st.session_state.model_downloaded:
            if st.button("Download Vision Model", type="primary"):
                if download_model():
                    st.success("✅ Model downloaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Vision model is ready")
    
    # Only show upload section if everything is set up
    if st.session_state.ollama_setup and st.session_state.model_downloaded:
        st.header("📁 Upload Images")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose exactly 2 image files",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if len(uploaded_files) != 2:
                st.warning("⚠️ Please upload exactly 2 image files.")
            else:
                st.success(f"✅ Uploaded {len(uploaded_files)} images")
                
                # Display images side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Image 1")
                    image1 = Image.open(uploaded_files[0])
                    st.image(image1, caption=uploaded_files[0].name, use_column_width=True)
                
                with col2:
                    st.subheader("Image 2")
                    image2 = Image.open(uploaded_files[1])
                    st.image(image2, caption=uploaded_files[1].name, use_column_width=True)
                
                # Process images button
                if st.button("🔍 Extract Numbers and Calculate Difference", type="primary"):
                    st.header("📊 Results")
                    
                    with st.spinner("Processing images..."):
                        # Convert images to bytes
                        img1_bytes = uploaded_files[0].getvalue()
                        img2_bytes = uploaded_files[1].getvalue()
                        
                        # Extract numbers
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"Processing {uploaded_files[0].name}")
                            num1 = extract_number_from_image_bytes(img1_bytes, uploaded_files[0].name)
                        
                        with col2:
                            st.subheader(f"Processing {uploaded_files[1].name}")
                            num2 = extract_number_from_image_bytes(img2_bytes, uploaded_files[1].name)
                        
                        # Calculate difference
                        if num1 is not None and num2 is not None:
                            diff = abs(num1 - num2)
                            
                            st.header("📏 Final Results")
                            
                            # Create results table
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Image 1 Value", f"{num1} m")
                            
                            with col2:
                                st.metric("Image 2 Value", f"{num2} m")
                            
                            with col3:
                                st.metric("Difference", f"{diff} m")
                            
                            # Success message
                            st.success(f"✅ Fiber length difference calculated: **{diff} meters**")
                            
                        else:
                            st.error("❌ Could not calculate difference due to missing number(s).")
    
    # Instructions
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Setup**: Click the setup buttons to install Ollama and download the vision model
        2. **Upload**: Choose exactly 2 images with handwritten fiber lengths
        3. **Process**: Click the process button to extract numbers and calculate difference
        4. **Results**: View the extracted values and calculated difference
        
        **Supported formats**: PNG, JPG, JPEG, GIF, BMP
        
        **Note**: The first setup may take several minutes to complete.
        """)

if __name__ == "__main__":
    main()
