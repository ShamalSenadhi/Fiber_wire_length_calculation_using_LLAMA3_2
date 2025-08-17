import streamlit as st
import ollama
import re
import io
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Fiber Wire Length Calculator",
    page_icon="📏",
    layout="wide"
)

st.title("🔬 Fiber Wire Length Calculator using LLaMA3 Vision")
st.markdown("Upload handwritten fiber length images to extract measurements and calculate differences")

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """
    Extract a number from image bytes using Ollama LLaMA3 vision model.
    """
    try:
        # Convert PIL image to base64 for Ollama
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

def display_image_with_resize(image, max_width=300, max_height=300):
    """
    Display image with resizing
    """
    image_copy = image.copy()
    image_copy.thumbnail((max_width, max_height))
    return image_copy

def main():
    # Initialize session state
    if 'processed_first_pair' not in st.session_state:
        st.session_state.processed_first_pair = False
    if 'first_pair_results' not in st.session_state:
        st.session_state.first_pair_results = {}

    # Sidebar for model configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Check if Ollama is running
    try:
        models = ollama.list()
        available_models = [model['name'] for model in models['models'] if 'vision' in model['name']]
        
        if not available_models:
            st.sidebar.error("No vision models found. Please install llama3.2-vision:11b")
            st.stop()
        
        selected_model = st.sidebar.selectbox(
            "Select Vision Model:",
            available_models,
            index=0 if 'llama3.2-vision:11b' in available_models else 0
        )
        
    except Exception as e:
        st.sidebar.error(f"Ollama not running or not accessible: {str(e)}")
        st.sidebar.info("Please make sure Ollama is installed and running with the llama3.2-vision:11b model")
        st.stop()

    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 First Image Pair")
        uploaded_files_1 = st.file_uploader(
            "Upload first two images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="first_pair"
        )
        
        if uploaded_files_1 and len(uploaded_files_1) == 2:
            if st.button("Process First Pair", key="process_first"):
                with st.spinner("Processing first image pair..."):
                    results = {}
                    
                    for uploaded_file in uploaded_files_1:
                        # Display image
                        image = Image.open(uploaded_file)
                        st.image(display_image_with_resize(image), 
                                caption=uploaded_file.name, use_column_width=True)
                        
                        # Convert to bytes for Ollama
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        # Extract number
                        length = extract_number_from_image_bytes(img_bytes, uploaded_file.name)
                        if length is not None:
                            results[uploaded_file.name] = length
                    
                    # Store results and calculate difference
                    st.session_state.first_pair_results = results
                    st.session_state.processed_first_pair = True
                    
                    if len(results) == 2:
                        lengths = list(results.values())
                        difference = abs(lengths[0] - lengths[1])
                        
                        st.success("✅ First Pair Results:")
                        for name, length in results.items():
                            st.write(f"**{name}:** {length} meters")
                        st.write(f"**Difference:** {difference} meters")
                    else:
                        st.error("Could not extract numbers from both images")
                        
        elif uploaded_files_1 and len(uploaded_files_1) != 2:
            st.warning("Please upload exactly 2 images for the first pair")

    with col2:
        st.header("📤 Second Image Pair")
        uploaded_files_2 = st.file_uploader(
            "Upload second two images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="second_pair"
        )
        
        if uploaded_files_2 and len(uploaded_files_2) == 2:
            if st.button("Process Second Pair", key="process_second"):
                with st.spinner("Processing second image pair..."):
                    results = {}
                    
                    for uploaded_file in uploaded_files_2:
                        # Display image
                        image = Image.open(uploaded_file)
                        st.image(display_image_with_resize(image), 
                                caption=uploaded_file.name, use_column_width=True)
                        
                        # Convert to bytes for Ollama
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        # Extract number
                        length = extract_number_from_image_bytes(img_bytes, uploaded_file.name)
                        if length is not None:
                            results[uploaded_file.name] = length
                    
                    if len(results) == 2:
                        lengths = list(results.values())
                        difference = abs(lengths[0] - lengths[1])
                        
                        st.success("✅ Second Pair Results:")
                        for name, length in results.items():
                            st.write(f"**{name}:** {length} meters")
                        st.write(f"**Difference:** {difference} meters")
                    else:
                        st.error("Could not extract numbers from both images")
                        
        elif uploaded_files_2 and len(uploaded_files_2) != 2:
            st.warning("Please upload exactly 2 images for the second pair")

    # Summary section
    if st.session_state.processed_first_pair:
        st.header("📊 Summary")
        st.write("**First Pair Results:**")
        for name, length in st.session_state.first_pair_results.items():
            st.write(f"- {name}: {length} meters")
        
        if len(st.session_state.first_pair_results) == 2:
            lengths = list(st.session_state.first_pair_results.values())
            difference = abs(lengths[0] - lengths[1])
            st.write(f"- **Difference:** {difference} meters")

    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions:
    1. **Install Requirements**: Make sure you have all required packages installed
    2. **Start Ollama**: Ensure Ollama is running with the LLaMA3 vision model
    3. **Upload Images**: Upload exactly 2 images for each pair
    4. **Process**: Click the process button to extract fiber lengths
    5. **Review Results**: Check the extracted measurements and differences
    
    ### 🔧 Model Requirements:
    - LLaMA3.2 Vision model (11B parameters)
    - Ollama server running locally
    - GPU support for faster processing
    """)

if __name__ == "__main__":
    main()
