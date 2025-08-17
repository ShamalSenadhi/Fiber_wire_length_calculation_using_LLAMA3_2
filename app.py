import streamlit as st
import ollama
import re
from PIL import Image
import io
import os

# Set page config
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide"
)

st.title("📏 Handwritten Fiber Length Analyzer")
st.write("Upload two images with handwritten fiber lengths to calculate the difference.")

# Check if Ollama is available
@st.cache_resource
def check_ollama_connection():
    try:
        # Try to list available models
        models = ollama.list()
        return True
    except Exception as e:
        st.error(f"Ollama connection failed: {str(e)}")
        st.info("Make sure Ollama is running and the llama3.2-vision:11b model is installed.")
        return False

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """Extract a number from image bytes using Ollama vision model."""
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
        
        # Use regular expression to find a numerical value
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
    # Check Ollama connection
    if not check_ollama_connection():
        st.stop()
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Image 1")
        uploaded_file1 = st.file_uploader(
            "Choose first image file", 
            type=['png', 'jpg', 'jpeg'],
            key="file1"
        )
        
    with col2:
        st.subheader("📸 Image 2")
        uploaded_file2 = st.file_uploader(
            "Choose second image file", 
            type=['png', 'jpg', 'jpeg'],
            key="file2"
        )
    
    # Display uploaded images
    if uploaded_file1 is not None:
        with col1:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Image 1", use_column_width=True)
    
    if uploaded_file2 is not None:
        with col2:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Image 2", use_column_width=True)
    
    # Process images when both are uploaded
    if uploaded_file1 is not None and uploaded_file2 is not None:
        if st.button("🔍 Analyze Images", type="primary"):
            with st.spinner("Analyzing images..."):
                # Convert uploaded files to bytes
                image1_bytes = uploaded_file1.read()
                uploaded_file1.seek(0)  # Reset file pointer
                image2_bytes = uploaded_file2.read()
                
                # Extract numbers from both images
                col1_results, col2_results = st.columns(2)
                
                with col1_results:
                    st.subheader("Analysis Results - Image 1")
                    num1 = extract_number_from_image_bytes(image1_bytes, uploaded_file1.name)
                
                with col2_results:
                    st.subheader("Analysis Results - Image 2")
                    num2 = extract_number_from_image_bytes(image2_bytes, uploaded_file2.name)
                
                # Calculate difference if both numbers were extracted
                if num1 is not None and num2 is not None:
                    diff = abs(num1 - num2)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display results in a nice format
                    st.markdown("---")
                    st.subheader("📊 Results Summary")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Image 1 Value", f"{num1} m")
                    
                    with result_col2:
                        st.metric("Image 2 Value", f"{num2} m")
                    
                    with result_col3:
                        st.metric("Difference", f"{diff} m", delta=None)
                    
                else:
                    st.error("❌ Could not calculate difference due to missing number(s).")
    
    else:
        st.info("👆 Please upload both images to begin analysis.")

if __name__ == "__main__":
    main()
