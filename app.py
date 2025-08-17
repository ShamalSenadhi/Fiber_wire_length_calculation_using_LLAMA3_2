import streamlit as st
import ollama
import re
from PIL import Image
import io
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📏 Handwritten Fiber Length Analyzer")
st.markdown("""
This application uses Ollama's llama3.2-vision:11b model to extract handwritten fiber lengths from images 
and calculate the difference between two measurements.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["llama3.2-vision:11b", "llava:latest", "bakllava:latest"],
    index=0
)

# Check if Ollama is running
@st.cache_data
def check_ollama_connection():
    try:
        response = ollama.list()
        return True, "Connected to Ollama successfully"
    except Exception as e:
        return False, f"Failed to connect to Ollama: {str(e)}"

# Function to extract number from image
def extract_number_from_image(image_bytes, model=model_name):
    """Extract handwritten number from image using Ollama vision model"""
    try:
        with st.spinner("Analyzing image..."):
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': 'Extract the handwritten number in meters from this image. Only return the numerical value.',
                    'images': [image_bytes]
                }]
            )
        
        content = response['message']['content']
        st.write(f"**Raw model output:** {content}")
        
        # Use regex to find numerical value
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m(?:eters?)?)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning("No numerical value found in the response")
            return None
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Check Ollama connection status
connection_status, connection_msg = check_ollama_connection()
if connection_status:
    st.sidebar.success(connection_msg)
else:
    st.sidebar.error(connection_msg)
    st.error("Please ensure Ollama is running and the model is available.")
    st.stop()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Image 1")
    uploaded_file1 = st.file_uploader(
        "Upload first fiber length image",
        type=['png', 'jpg', 'jpeg'],
        key="image1"
    )
    
    if uploaded_file1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="First Image", use_column_width=True)

with col2:
    st.header("📸 Image 2")
    uploaded_file2 = st.file_uploader(
        "Upload second fiber length image",
        type=['png', 'jpg', 'jpeg'],
        key="image2"
    )
    
    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="Second Image", use_column_width=True)

# Process images when both are uploaded
if uploaded_file1 and uploaded_file2:
    if st.button("🔍 Analyze Images", type="primary"):
        st.header("📊 Analysis Results")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process first image
        status_text.text("Processing first image...")
        progress_bar.progress(25)
        image1_bytes = uploaded_file1.getvalue()
        num1 = extract_number_from_image(image1_bytes, model_name)
        
        # Process second image
        status_text.text("Processing second image...")
        progress_bar.progress(75)
        image2_bytes = uploaded_file2.getvalue()
        num2 = extract_number_from_image(image2_bytes, model_name)
        
        # Complete processing
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if num1 is not None:
                st.metric("Image 1 Length", f"{num1} m")
            else:
                st.error("Could not extract value from Image 1")
        
        with col2:
            if num2 is not None:
                st.metric("Image 2 Length", f"{num2} m")
            else:
                st.error("Could not extract value from Image 2")
        
        with col3:
            if num1 is not None and num2 is not None:
                difference = abs(num1 - num2)
                st.metric("Difference", f"{difference} m")
                
                # Additional analysis
                st.subheader("📈 Analysis Summary")
                larger_value = max(num1, num2)
                smaller_value = min(num1, num2)
                percentage_diff = (difference / larger_value) * 100 if larger_value > 0 else 0
                
                st.write(f"**Larger value:** {larger_value} m")
                st.write(f"**Smaller value:** {smaller_value} m")
                st.write(f"**Percentage difference:** {percentage_diff:.2f}%")
                
                if percentage_diff < 5:
                    st.success("✅ Values are very close (< 5% difference)")
                elif percentage_diff < 15:
                    st.warning("⚠️ Moderate difference (5-15%)")
                else:
                    st.error("❌ Significant difference (> 15%)")
            else:
                st.error("Cannot calculate difference due to missing values")

# Instructions section
st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. **Start Ollama**: Ensure Ollama service is running
2. **Model Ready**: Verify llama3.2-vision:11b is available
3. **Upload Images**: Upload two images with handwritten fiber lengths
4. **Analyze**: Click the analyze button to process both images
5. **View Results**: See extracted values and calculated difference

**Supported formats:** PNG, JPG, JPEG
""")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit and Ollama | "
    "Model: llama3.2-vision:11b"
)
