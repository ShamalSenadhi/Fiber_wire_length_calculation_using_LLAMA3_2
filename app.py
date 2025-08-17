import streamlit as st
import ollama
import re
from PIL import Image
import io
import base64

ollama serve  # Run in separate terminal
ollama pull llama3.2-vision:11b


# Page configuration
st.set_page_config(
    page_title="Fiber Length Analyzer",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #BDC3C7;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #F8F9FA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-card {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_number_from_image_bytes(image_bytes, image_name='uploaded_image'):
    """
    Extract a number from image bytes using Ollama and a specified model.
    """
    try:
        # Convert image bytes to base64 for Ollama
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Send a chat request to the Ollama model
        response = ollama.chat(
            model='llama3.2-vision:11b',
            messages=[{
                'role': 'user',
                'content': 'Extract the handwritten number in meters from this image.',
                'images': [image_base64]
            }]
        )
        
        # Extract the content of the model's response
        content = response['message']['content']
        st.write(f"**Raw model output for {image_name}:**")
        st.write(content)
        
        # Use regular expression to find a numerical value
        match = re.search(r'(\d+(?:\.\d+)?)(?:\s*m(?:eters?)?)?', content.lower())
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📏 Fiber Length Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses Ollama's LLaMA 3.2 Vision model to:
        - Extract handwritten numbers from images
        - Calculate the difference between fiber lengths
        - Display results with visualizations
        """)
        
        st.header("📋 Instructions")
        st.write("""
        1. Upload exactly 2 images containing handwritten fiber lengths
        2. Click 'Analyze Images' to process
        3. View extracted numbers and calculated difference
        """)
        
        st.header("⚙️ Requirements")
        st.write("""
        - Ollama server running locally
        - llama3.2-vision:11b model installed
        - Clear handwritten numbers in images
        """)

    # Check Ollama connection
    try:
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        if 'llama3.2-vision:11b' not in model_names:
            st.error("❌ llama3.2-vision:11b model not found. Please install it using: `ollama pull llama3.2-vision:11b`")
            return
        else:
            st.success("✅ Ollama connection established and LLaMA 3.2 Vision model ready")
    except Exception as e:
        st.error(f"❌ Cannot connect to Ollama server: {str(e)}")
        st.info("Please ensure Ollama is running locally with: `ollama serve`")
        return

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose exactly 2 images with handwritten fiber lengths",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload images containing handwritten numbers representing fiber lengths in meters"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        if len(uploaded_files) != 2:
            st.warning(f"⚠️ Please upload exactly 2 images. You uploaded {len(uploaded_files)} image(s).")
        else:
            # Display uploaded images
            st.subheader("📸 Uploaded Images")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Image 1:** {uploaded_files[0].name}")
                image1 = Image.open(uploaded_files[0])
                st.image(image1, caption=uploaded_files[0].name, use_column_width=True)
                
            with col2:
                st.write(f"**Image 2:** {uploaded_files[1].name}")
                image2 = Image.open(uploaded_files[1])
                st.image(image2, caption=uploaded_files[1].name, use_column_width=True)

            # Analysis button
            if st.button("🔍 Analyze Images", type="primary", use_container_width=True):
                with st.spinner("🤖 Processing images with LLaMA 3.2 Vision..."):
                    # Process images
                    image1_bytes = uploaded_files[0].getvalue()
                    image2_bytes = uploaded_files[1].getvalue()
                    
                    # Extract numbers
                    st.subheader("🧠 LLaMA 3.2 Vision Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Processing Image 1...**")
                        num1 = extract_number_from_image_bytes(image1_bytes, uploaded_files[0].name)
                        
                    with col2:
                        st.write("**Processing Image 2...**")
                        num2 = extract_number_from_image_bytes(image2_bytes, uploaded_files[1].name)
                    
                    # Calculate and display results
                    st.subheader("📊 Results")
                    
                    if num1 is not None and num2 is not None:
                        diff = abs(num1 - num2)
                        
                        # Results display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Image 1 Length", f"{num1} m")
                            
                        with col2:
                            st.metric("Image 2 Length", f"{num2} m")
                            
                        with col3:
                            st.metric("Difference", f"{diff} m", delta=f"{diff:.2f}")
                        
                        # Success message
                        st.markdown(f'''
                        <div class="result-card">
                            <h4>✅ Analysis Complete!</h4>
                            <p><strong>Fiber Length 1:</strong> {num1} meters</p>
                            <p><strong>Fiber Length 2:</strong> {num2} meters</p>
                            <p><strong>Absolute Difference:</strong> {diff} meters</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Display uploaded images (resized)
                        st.subheader("📸 Processed Images")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**{uploaded_files[0].name}** - {num1}m")
                            image1_resized = image1.copy()
                            image1_resized.thumbnail((300, 300))
                            st.image(image1_resized)
                            
                        with col2:
                            st.write(f"**{uploaded_files[1].name}** - {num2}m")
                            image2_resized = image2.copy()
                            image2_resized.thumbnail((300, 300))
                            st.image(image2_resized)
                        
                    else:
                        st.markdown('''
                        <div class="error-card">
                            <h4>❌ Analysis Failed</h4>
                            <p>Could not extract numbers from one or both images. Please ensure:</p>
                            <ul>
                                <li>Numbers are clearly handwritten and legible</li>
                                <li>Images have good contrast and lighting</li>
                                <li>Numbers are visible and not obscured</li>
                                <li>Ollama server is running properly</li>
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7F8C8D;'>Powered by Ollama LLaMA 3.2 Vision | Built with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
