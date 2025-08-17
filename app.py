import streamlit as st
import ollama
import re
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Fiber Length Extractor",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.sub-header {
    font-size: 1.5rem;
    color: #3730a3;
    margin: 1.5rem 0;
}

.result-box {
    background-color: #f0f9ff;
    border: 2px solid #0ea5e9;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background-color: #fef2f2;
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}

.success-box {
    background-color: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extracted_numbers' not in st.session_state:
    st.session_state.extracted_numbers = []
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

def check_ollama_connection():
    """Check if Ollama service is running and the model is available."""
    try:
        # Try to list available models
        models = ollama.list()
        model_names = [model['name'] for model in models.get('models', [])]
        return 'llama3.2-vision:11b' in model_names
    except Exception as e:
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract a number from image bytes using Ollama vision model."""
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
        st.code(content, language="text")
        
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

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔢 Fiber Length Extractor</div>', unsafe_allow_html=True)
    st.markdown("Upload images with handwritten fiber lengths to extract numerical values and calculate differences.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check Ollama connection
        if st.button("🔍 Check Ollama Connection"):
            with st.spinner("Checking Ollama service..."):
                if check_ollama_connection():
                    st.success("✅ Ollama is running and llama3.2-vision:11b is available!")
                else:
                    st.error("❌ Ollama service not found or model not available. Please ensure Ollama is running and the model is installed.")
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload exactly 2 images** with handwritten fiber lengths
        2. **Click 'Extract Numbers'** to process the images
        3. **View the results** and calculated difference
        """)
        
        st.markdown("### 🔧 Requirements")
        st.markdown("""
        - Ollama service running
        - llama3.2-vision:11b model installed
        - Images with clear handwritten numbers
        """)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Images</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload exactly 2 images containing handwritten fiber lengths"
        )
        
        if uploaded_files:
            if len(uploaded_files) != 2:
                st.markdown('<div class="error-box">⚠️ Please upload exactly 2 images.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ {len(uploaded_files)} images uploaded successfully!</div>', unsafe_allow_html=True)
                
                # Display uploaded images
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f"**Image {i+1}: {uploaded_file.name}**")
                    image = Image.open(uploaded_file)
                    # Resize for display
                    image.thumbnail((300, 300))
                    st.image(image, caption=f"Image {i+1}", use_column_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🔢 Extraction Results</div>', unsafe_allow_html=True)
        
        if uploaded_files and len(uploaded_files) == 2:
            if st.button("🚀 Extract Numbers", type="primary", use_container_width=True):
                with st.spinner("Processing images with AI model..."):
                    extracted_numbers = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        st.markdown(f"### Processing Image {i+1}: {uploaded_file.name}")
                        
                        # Convert to bytes
                        image_bytes = uploaded_file.getvalue()
                        
                        # Extract number
                        number = extract_number_from_image_bytes(image_bytes, uploaded_file.name)
                        extracted_numbers.append(number)
                        
                        if number is not None:
                            st.markdown(f'<div class="result-box">📊 Extracted value: **{number} meters**</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">❌ Could not extract number from this image</div>', unsafe_allow_html=True)
                    
                    # Calculate difference if both numbers extracted
                    if all(num is not None for num in extracted_numbers):
                        difference = abs(extracted_numbers[0] - extracted_numbers[1])
                        
                        st.markdown("---")
                        st.markdown("### 📏 Final Results")
                        
                        results_col1, results_col2, results_col3 = st.columns(3)
                        
                        with results_col1:
                            st.metric(
                                label="Image 1",
                                value=f"{extracted_numbers[0]} m"
                            )
                        
                        with results_col2:
                            st.metric(
                                label="Image 2", 
                                value=f"{extracted_numbers[1]} m"
                            )
                        
                        with results_col3:
                            st.metric(
                                label="Difference",
                                value=f"{difference} m",
                                delta=f"{difference:.2f}"
                            )
                        
                        st.markdown(f'<div class="success-box">🎯 **Fiber Length Difference: {difference} meters**</div>', unsafe_allow_html=True)
                        
                        # Store in session state
                        st.session_state.extracted_numbers = extracted_numbers
                        
                    else:
                        st.markdown('<div class="error-box">❌ Could not calculate difference due to missing numbers.</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Upload 2 images above to start extraction")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🤖 Powered by Ollama & llama3.2-vision:11b | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
