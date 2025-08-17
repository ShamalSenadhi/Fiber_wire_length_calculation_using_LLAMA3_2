import streamlit as st
import ollama
import re
from PIL import Image
import io
import subprocess
import time
import os

# Set page config
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide"
)

@st.cache_resource
def setup_ollama():
    """Initialize Ollama service and pull the model"""
    try:
        # Set environment variable for CUDA
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia'
        
        # Start Ollama service
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for service to start
        
        # Pull the model
        st.info("Loading AI model... This may take a few minutes on first run.")
        result = subprocess.run(['ollama', 'pull', 'llama3.2-vision:11b'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("AI model loaded successfully!")
            return True
        else:
            st.error(f"Failed to load model: {result.stderr}")
            return False
            
    except Exception as e:
        st.error(f"Error setting up Ollama: {str(e)}")
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract handwritten number from image using Ollama vision model"""
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
        st.write(f"**AI Response for {image_name}:** {content}")
        
        # Extract number using regex
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m| meters)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {str(e)}")
        return None

def main():
    st.title("🔬 Fiber Length Analyzer")
    st.markdown("Upload two images with handwritten fiber lengths to calculate the difference.")
    
    # Initialize Ollama
    if 'ollama_ready' not in st.session_state:
        st.session_state.ollama_ready = setup_ollama()
    
    if not st.session_state.ollama_ready:
        st.error("Failed to initialize AI model. Please refresh the page.")
        return
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose exactly 2 image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images containing handwritten fiber lengths"
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        col1, col2 = st.columns(2)
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            with col1 if i == 0 else col2:
                st.subheader(f"Image {i+1}: {uploaded_file.name}")
                
                # Display image
                image = Image.open(uploaded_file)
                image.thumbnail((300, 300))
                st.image(image, caption=f"Uploaded: {uploaded_file.name}")
                
                # Process image
                image_bytes = uploaded_file.getvalue()
                
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    number = extract_number_from_image_bytes(image_bytes, uploaded_file.name)
                    results.append(number)
                
                if number is not None:
                    st.success(f"✅ Extracted: {number} meters")
                else:
                    st.error("❌ Could not extract number")
        
        # Calculate difference
        if all(result is not None for result in results):
            difference = abs(results[0] - results[1])
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image 1 Length", f"{results[0]} m")
            with col2:
                st.metric("Image 2 Length", f"{results[1]} m")
            with col3:
                st.metric("Difference", f"{difference} m", 
                         delta=f"{difference:.2f}" if difference > 0 else "0")
            
            if difference > 0:
                st.info(f"The fiber length difference is **{difference} meters**")
            else:
                st.success("The fiber lengths are identical!")
        
        else:
            st.warning("Could not calculate difference - please ensure both images contain clear handwritten numbers")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning(f"Please upload exactly 2 images. You uploaded {len(uploaded_files)} files.")
    
    else:
        st.info("👆 Upload two images to get started")

if __name__ == "__main__":
    main()
