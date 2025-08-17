import streamlit as st
import re
from PIL import Image
import io
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

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

@st.cache_resource
def load_model():
    """Load the LLaVA model and processor"""
    try:
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        return processor, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def extract_number_from_image(image, image_name, processor, model):
    """
    Extract a number from image using LLaVA model.
    """
    try:
        # Prepare the prompt
        prompt = "USER: <image>\nExtract the handwritten number from this image that represents a fiber length measurement in meters. Only return the numerical value.\nASSISTANT:"
        
        # Process the image and text
        inputs = processor(prompt, image, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        # Decode the response
        response = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        assistant_response = response.split("ASSISTANT:")[-1].strip()
        
        st.write(f"**AI Response for {image_name}:** {assistant_response}")
        
        # Use regular expression to find a numerical value
        match = re.search(r'(\d+(?:\.\d+)?)', assistant_response)
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
        This app uses LLaVA (Large Language and Vision Assistant) to:
        - Extract handwritten numbers from images
        - Calculate the difference between fiber lengths
        - Display results with visualizations
        """)
        
        st.header("📋 Instructions")
        st.write("""
        1. Wait for the model to load (first time may take a few minutes)
        2. Upload exactly 2 images containing handwritten fiber lengths
        3. Click 'Analyze Images' to process
        4. View extracted numbers and calculated difference
        """)
        
        st.header("🤖 Model Info")
        st.write("""
        - **Model**: LLaVA-v1.6-Mistral-7B
        - **Type**: Vision-Language Model
        - **Provider**: Hugging Face Transformers
        """)

    # Load model
    with st.spinner("🤖 Loading AI model... (This may take a few minutes on first run)"):
        processor, model = load_model()
    
    if processor is None or model is None:
        st.error("❌ Failed to load the AI model. Please refresh the page and try again.")
        return
    else:
        st.success("✅ LLaVA model loaded successfully!")

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
                image1 = Image.open(uploaded_files[0]).convert("RGB")
                st.image(image1, caption=uploaded_files[0].name, use_column_width=True)
                
            with col2:
                st.write(f"**Image 2:** {uploaded_files[1].name}")
                image2 = Image.open(uploaded_files[1]).convert("RGB")
                st.image(image2, caption=uploaded_files[1].name, use_column_width=True)

            # Analysis button
            if st.button("🔍 Analyze Images", type="primary", use_container_width=True):
                with st.spinner("🤖 Processing images with LLaVA..."):
                    
                    # Extract numbers
                    st.subheader("🧠 LLaVA Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Processing Image 1...**")
                        num1 = extract_number_from_image(image1, uploaded_files[0].name, processor, model)
                        
                    with col2:
                        st.write("**Processing Image 2...**")
                        num2 = extract_number_from_image(image2, uploaded_files[1].name, processor, model)
                    
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
                            </ul>
                        </div>
                        ''', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7F8C8D;'>Powered by LLaVA-v1.6-Mistral-7B | Built with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
