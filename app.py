import streamlit as st
import ollama
import re
import io
from PIL import Image
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Fiber Wire Length Calculator",
    page_icon="📏",
    layout="wide"
)

# Title and description
st.title("📏 Fiber Wire Length Calculator")
st.markdown("**Using LLAMA3 Vision Model to extract handwritten fiber lengths from images**")

# Function to extract number from image using Ollama
def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """
    Extract handwritten number from image using LLAMA3 vision model
    """
    try:
        # Convert image bytes to base64 for Ollama
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Send chat request to Ollama model
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Extract the handwritten number in meters from this image. Only return the numerical value.',
                'images': [image_b64]
            }]
        )
        
        # Extract content from response
        content = response['message']['content']
        st.write(f"**Raw model output for {image_name}:**")
        st.code(content)
        
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

# Function to display image with resize
def display_resized_image(image_bytes, image_name, max_width=300, max_height=300):
    """
    Display resized image in Streamlit
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((max_width, max_height))
        return image
    except Exception as e:
        st.error(f"Could not display image {image_name}: {e}")
        return None

# Main application logic
def main():
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload exactly two images containing handwritten fiber lengths
    2. The LLAMA3 vision model will extract the numerical values
    3. The application will calculate the difference between the two lengths
    4. Repeat for additional image pairs
    """)
    
    # Check if Ollama is available
    try:
        models = ollama.list()
        # Handle different response formats
        if isinstance(models, dict) and 'models' in models:
            model_names = [model['name'] for model in models['models']]
        else:
            model_names = [model.get('name', '') for model in models] if isinstance(models, list) else []
        
        if 'llama3.2-vision:11b' not in model_names:
            st.warning("⚠️ LLAMA3.2 Vision model not found in available models:")
            st.code(model_names if model_names else "No models found")
            st.info("Please install it using: `ollama pull llama3.2-vision:11b`")
            st.info("Or try a different model that supports vision, such as 'llava' or 'llama3.2-vision'")
            # Don't return here - let user proceed anyway
        else:
            st.success("✅ LLAMA3.2 Vision model is available")
    except Exception as e:
        st.error(f"❌ Ollama connection failed: {str(e)}")
        st.info("Make sure Ollama is running locally. Start it with: `ollama serve`")
        st.info("If you're on Streamlit Cloud, you'll need to host Ollama separately.")
        # Don't return here - show the interface anyway for debugging
    
    # First set of images
    st.header("First Image Pair")
    uploaded_files_1 = st.file_uploader(
        "Upload first two images containing handwritten fiber lengths",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="first_pair"
    )
    
    if uploaded_files_1:
        if len(uploaded_files_1) != 2:
            st.warning("Please upload exactly two image files.")
        else:
            st.success(f"Uploaded {len(uploaded_files_1)} files")
            
            # Display uploaded images
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Image 1: {uploaded_files_1[0].name}")
                image1 = display_resized_image(uploaded_files_1[0].getvalue(), uploaded_files_1[0].name)
                if image1:
                    st.image(image1)
            
            with col2:
                st.subheader(f"Image 2: {uploaded_files_1[1].name}")
                image2 = display_resized_image(uploaded_files_1[1].getvalue(), uploaded_files_1[1].name)
                if image2:
                    st.image(image2)
            
            # Process images button
            if st.button("Process First Image Pair", key="process_1"):
                with st.spinner("Processing images with LLAMA3 model..."):
                    # Extract numbers from both images
                    num1 = extract_number_from_image_bytes(
                        uploaded_files_1[0].getvalue(), 
                        uploaded_files_1[0].name
                    )
                    num2 = extract_number_from_image_bytes(
                        uploaded_files_1[1].getvalue(), 
                        uploaded_files_1[1].name
                    )
                    
                    # Calculate difference
                    if num1 is not None and num2 is not None:
                        diff = abs(num1 - num2)
                        
                        st.success("✅ Processing completed!")
                        
                        # Display results
                        results_col1, results_col2, results_col3 = st.columns(3)
                        
                        with results_col1:
                            st.metric("Length 1", f"{num1} meters")
                        
                        with results_col2:
                            st.metric("Length 2", f"{num2} meters")
                        
                        with results_col3:
                            st.metric("Difference", f"{diff} meters", delta=f"±{diff}")
                        
                        st.balloons()
                    else:
                        st.error("❌ Could not calculate difference due to missing number(s).")
    
    st.divider()
    
    # Second set of images
    st.header("Second Image Pair")
    uploaded_files_2 = st.file_uploader(
        "Upload next two images containing handwritten fiber lengths",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="second_pair"
    )
    
    if uploaded_files_2:
        if len(uploaded_files_2) != 2:
            st.warning("Please upload exactly two image files.")
        else:
            st.success(f"Uploaded {len(uploaded_files_2)} files")
            
            # Display uploaded images
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Image 3: {uploaded_files_2[0].name}")
                image3 = display_resized_image(uploaded_files_2[0].getvalue(), uploaded_files_2[0].name)
                if image3:
                    st.image(image3)
            
            with col2:
                st.subheader(f"Image 4: {uploaded_files_2[1].name}")
                image4 = display_resized_image(uploaded_files_2[1].getvalue(), uploaded_files_2[1].name)
                if image4:
                    st.image(image4)
            
            # Process images button
            if st.button("Process Second Image Pair", key="process_2"):
                with st.spinner("Processing images with LLAMA3 model..."):
                    # Extract numbers from both images
                    num3 = extract_number_from_image_bytes(
                        uploaded_files_2[0].getvalue(), 
                        uploaded_files_2[0].name
                    )
                    num4 = extract_number_from_image_bytes(
                        uploaded_files_2[1].getvalue(), 
                        uploaded_files_2[1].name
                    )
                    
                    # Calculate difference
                    if num3 is not None and num4 is not None:
                        diff2 = abs(num3 - num4)
                        
                        st.success("✅ Processing completed!")
                        
                        # Display results
                        results_col1, results_col2, results_col3 = st.columns(3)
                        
                        with results_col1:
                            st.metric("Length 3", f"{num3} meters")
                        
                        with results_col2:
                            st.metric("Length 4", f"{num4} meters")
                        
                        with results_col3:
                            st.metric("Difference", f"{diff2} meters", delta=f"±{diff2}")
                        
                        st.balloons()
                    else:
                        st.error("❌ Could not calculate difference due to missing number(s).")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        "**Fiber Wire Length Calculator** | Powered by LLAMA3 Vision Model & Streamlit",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
