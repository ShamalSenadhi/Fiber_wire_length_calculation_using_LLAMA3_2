import streamlit as st
import re
from PIL import Image
import io
import base64
import openai
from openai import OpenAI

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
    .info-card {
        background-color: #EBF3FD;
        border-left: 5px solid #3498DB;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def encode_image(image_bytes):
    """Convert image bytes to base64 string for OpenAI API"""
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_number_from_image_bytes(image_bytes, image_name, api_key):
    """
    Extract a number from image bytes using OpenAI's GPT-4 Vision API.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Convert image to base64
        base64_image = encode_image(image_bytes)
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the handwritten number from this image that represents a fiber length measurement. The number should be in meters. Only return the numerical value, nothing else."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        # Extract the content
        content = response.choices[0].message.content.strip()
        st.write(f"**AI Response for {image_name}:** {content}")
        
        # Use regular expression to find a numerical value
        match = re.search(r'(\d+(?:\.\d+)?)', content)
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
    
    # Sidebar with information and API key input
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use GPT-4 Vision for image analysis"
        )
        
        if api_key:
            st.success("✅ API Key provided")
        else:
            st.info("🔑 Please enter your OpenAI API key to continue")
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.write("""
        This app uses OpenAI's GPT-4 Vision to:
        - Extract handwritten numbers from images
        - Calculate the difference between fiber lengths
        - Display results with visualizations
        """)
        
        st.header("📋 Instructions")
        st.write("""
        1. Enter your OpenAI API key in the sidebar
        2. Upload exactly 2 images with handwritten fiber lengths
        3. Click 'Analyze Images' to process
        4. View extracted numbers and calculated difference
        """)
        
        st.header("💡 Tips")
        st.write("""
        - Ensure handwritten numbers are clear and legible
        - Good lighting and contrast improve accuracy
        - Numbers should be clearly visible in the image
        """)

    # API Key check
    if not api_key:
        st.markdown('''
        <div class="info-card">
            <h4>🔑 API Key Required</h4>
            <p>Please enter your OpenAI API key in the sidebar to use this application.</p>
            <p><strong>How to get an API key:</strong></p>
            <ol>
                <li>Go to <a href="https://platform.openai.com/" target="_blank">OpenAI Platform</a></li>
                <li>Sign up or log in to your account</li>
                <li>Navigate to the API keys section</li>
                <li>Create a new API key</li>
                <li>Copy and paste it in the sidebar</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)
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
                with st.spinner("🤖 Processing images with GPT-4 Vision..."):
                    # Process images
                    image1_bytes = uploaded_files[0].getvalue()
                    image2_bytes = uploaded_files[1].getvalue()
                    
                    # Extract numbers
                    st.subheader("🧠 AI Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Processing Image 1...**")
                        num1 = extract_number_from_image_bytes(image1_bytes, uploaded_files[0].name, api_key)
                        
                    with col2:
                        st.write("**Processing Image 2...**")
                        num2 = extract_number_from_image_bytes(image2_bytes, uploaded_files[1].name, api_key)
                    
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
                        
                    else:
                        st.markdown('''
                        <div class="error-card">
                            <h4>❌ Analysis Failed</h4>
                            <p>Could not extract numbers from one or both images. Please ensure:</p>
                            <ul>
                                <li>Numbers are clearly handwritten and legible</li>
                                <li>Images have good contrast and lighting</li>
                                <li>Numbers are visible and not obscured</li>
                                <li>Your OpenAI API key is valid and has sufficient credits</li>
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7F8C8D;'>Powered by OpenAI GPT-4 Vision | Built with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
