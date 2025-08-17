import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import subprocess
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fiber Length Recognition System",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FiberLengthExtractor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2-vision:11b"
        
    def check_ollama_service(self):
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama_service(self):
        """Start Ollama service in background"""
        try:
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(3)
            return True
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    def pull_model(self):
        """Pull the required model if not available"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def extract_number_from_image_bytes(self, image_bytes):
        """Extract fiber length from image using LLAMA3.2-Vision model"""
        try:
            # Convert image bytes to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": """You are an expert at reading handwritten measurements on fiber cable tags and telecommunications documentation. 
                
                Please carefully examine this image and extract any handwritten fiber length measurements you can see. 
                Look for:
                - Numbers followed by units like 'm', 'meter', 'meters', 'ft', 'feet'
                - Handwritten measurements on cable tags or labels
                - Length specifications in metric or imperial units
                
                Return only the numerical value of the fiber length in meters. If you see feet, convert to meters (1 foot = 0.3048 meters).
                If no clear measurement is visible, return 'No measurement found'.
                
                Respond with just the number (e.g., '15.5' for 15.5 meters) or 'No measurement found'.""",
                "images": [image_base64],
                "stream": False
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get('response', '').strip()
                
                # Try to extract numerical value
                try:
                    # Look for numbers in the response
                    import re
                    numbers = re.findall(r'\d+\.?\d*', extracted_text)
                    if numbers:
                        return float(numbers[0])
                    else:
                        return None
                except:
                    return None
            else:
                logger.error(f"API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting number from image: {e}")
            return None

def main():
    st.title("🔌 Fiber Cable Length Recognition System")
    st.markdown("### LLAMA3.2-Vision Integration for Handwritten Fiber Length Recognition")
    
    # Initialize the extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = FiberLengthExtractor()
    
    extractor = st.session_state.extractor
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("System Status")
        
        # Check Ollama service status
        service_status = extractor.check_ollama_service()
        if service_status:
            st.success("✅ Ollama Service: Running")
        else:
            st.error("❌ Ollama Service: Not Running")
            if st.button("Start Ollama Service"):
                with st.spinner("Starting Ollama service..."):
                    if extractor.start_ollama_service():
                        st.success("Service started successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to start service")
        
        # Model management
        st.header("Model Management")
        if st.button("Pull/Update Model"):
            with st.spinner("Pulling LLAMA3.2-Vision model..."):
                if extractor.pull_model():
                    st.success("Model ready!")
                else:
                    st.error("Failed to pull model")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses **LLAMA3.2-Vision** model to automatically 
        extract handwritten fiber cable length measurements from images.
        
        **Features:**
        - Handwritten text recognition
        - Automatic unit conversion
        - Comparative analysis
        - Quality assurance validation
        """)
    
    # Main application area
    if not service_status:
        st.warning("⚠️ Please start the Ollama service first using the sidebar controls.")
        return
    
    # File upload section
    st.header("📤 Image Upload")
    st.markdown("Upload exactly 2 images of fiber cable tags or documentation for comparative analysis.")
    
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload images containing handwritten fiber length measurements"
    )
    
    if uploaded_files:
        if len(uploaded_files) != 2:
            st.error(f"Please upload exactly 2 images. You uploaded {len(uploaded_files)} image(s).")
            return
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image 1")
            image1 = Image.open(uploaded_files[0])
            st.image(image1, caption="First fiber cable image", use_column_width=True)
        
        with col2:
            st.subheader("Image 2")
            image2 = Image.open(uploaded_files[1])
            st.image(image2, caption="Second fiber cable image", use_column_width=True)
        
        # Process images button
        if st.button("🔍 Extract Fiber Lengths", type="primary"):
            st.header("🔄 Processing Results")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            
            # Process first image
            status_text.text("Processing Image 1...")
            progress_bar.progress(25)
            
            img1_bytes = uploaded_files[0].getvalue()
            length1 = extractor.extract_number_from_image_bytes(img1_bytes)
            results['image1'] = {
                'length': length1,
                'filename': uploaded_files[0].name
            }
            
            progress_bar.progress(50)
            
            # Process second image
            status_text.text("Processing Image 2...")
            progress_bar.progress(75)
            
            img2_bytes = uploaded_files[1].getvalue()
            length2 = extractor.extract_number_from_image_bytes(img2_bytes)
            results['image2'] = {
                'length': length2,
                'filename': uploaded_files[1].name
            }
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Display results
            st.header("📊 Extraction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Image 1 Length",
                    value=f"{length1}m" if length1 is not None else "Not detected",
                    help=f"From: {results['image1']['filename']}"
                )
            
            with col2:
                st.metric(
                    label="Image 2 Length", 
                    value=f"{length2}m" if length2 is not None else "Not detected",
                    help=f"From: {results['image2']['filename']}"
                )
            
            with col3:
                if length1 is not None and length2 is not None:
                    difference = abs(length1 - length2)
                    st.metric(
                        label="Difference",
                        value=f"{difference:.2f}m",
                        delta=f"±{difference:.2f}m"
                    )
                else:
                    st.metric(
                        label="Difference",
                        value="Cannot calculate",
                        help="One or more measurements not detected"
                    )
            
            # Detailed analysis
            st.header("📈 Comparative Analysis")
            
            if length1 is not None and length2 is not None:
                difference = abs(length1 - length2)
                percentage_diff = (difference / max(length1, length2)) * 100
                
                if difference < 0.5:
                    st.success(f"✅ **Measurements are consistent** (difference: {difference:.2f}m, {percentage_diff:.1f}%)")
                elif difference < 2.0:
                    st.warning(f"⚠️ **Moderate difference detected** (difference: {difference:.2f}m, {percentage_diff:.1f}%)")
                else:
                    st.error(f"❌ **Significant difference detected** (difference: {difference:.2f}m, {percentage_diff:.1f}%)")
                
                # Create comparison table
                comparison_data = {
                    'Metric': ['Length (m)', 'Source File', 'Status'],
                    'Image 1': [f"{length1:.2f}", results['image1']['filename'], '✅ Detected'],
                    'Image 2': [f"{length2:.2f}", results['image2']['filename'], '✅ Detected']
                }
                
                st.table(comparison_data)
                
            else:
                error_details = []
                if length1 is None:
                    error_details.append(f"❌ Image 1 ({results['image1']['filename']}): No measurement detected")
                if length2 is None:
                    error_details.append(f"❌ Image 2 ({results['image2']['filename']}): No measurement detected")
                
                st.error("**Measurement extraction failed for one or more images:**")
                for error in error_details:
                    st.write(error)
                
                st.info("""
                **Possible reasons for detection failure:**
                - Image quality is too low
                - Handwriting is unclear or ambiguous
                - No measurement values visible in the image
                - Measurement format not recognized
                
                **Suggestions:**
                - Ensure images are clear and well-lit
                - Make sure measurements are clearly visible
                - Try different angles or closer shots
                """)
    
    else:
        st.info("👆 Please upload 2 images to begin fiber length extraction and analysis.")

if __name__ == "__main__":
    main()
