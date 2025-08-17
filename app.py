import streamlit as st
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import re
import gc

@st.cache_resource
def load_model():
    """Load the Llama 3.2 Vision model with GPU support"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        
        # Load model with GPU support
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        st.success(f"✅ Llama 3.2 Vision model loaded successfully on {device}!")
        return model, processor, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("You may need to request access to the Llama models on Hugging Face")
        return None, None, None

def extract_number_from_image(image, image_name, model, processor, device):
    """Extract number from image using Llama 3.2 Vision"""
    try:
        # Prepare the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract the handwritten number in meters from this image. Return only the numerical value."}
                ]
            }
        ]
        
        # Process the input
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if device == "cuda":
            inputs = inputs.to(device)
        
        # Generate response
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode the response
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove the input prompt)
        prompt_length = len(input_text)
        response = generated_text[prompt_length:].strip()
        
        st.write(f"Model response for {image_name}: {response}")
        
        # Extract number using regex
        match = re.search(r'(\d+(?:\.\d+)?)', response)
        if match:
            return float(match.group(1))
        else:
            st.warning(f"No number found in {image_name}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {image_name}: {str(e)}")
        return None
    finally:
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

def main():
    st.title("🔢 Fiber Length Difference Calculator")
    st.markdown("*Powered by Llama 3.2 Vision with GPU acceleration*")
    
    # Load model
    if 'model_loaded' not in st.session_state:
        with st.spinner("Loading Llama 3.2 Vision model... This may take a few minutes."):
            model, processor, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.device = device
                st.session_state.model_loaded = True
            else:
                st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        st.error("❌ Model failed to load. Please check your setup.")
        st.markdown("""
        ### Requirements:
        1. **GPU with sufficient VRAM** (recommended: 12GB+ for 11B model)
        2. **Hugging Face access** to Llama models
        3. **HF Token** in secrets: `HF_TOKEN`
        
        ### Setup Instructions:
        1. Request access to Llama models at: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
        2. Add your HF token to Streamlit secrets
        3. Deploy on a GPU-enabled platform
        """)
        return
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        st.success(f"🚀 Running on: {gpu_name} ({gpu_memory}GB VRAM)")
    
    st.markdown("---")
    st.write("Upload two images with handwritten fiber lengths to calculate the difference")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose exactly 2 images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images containing handwritten numbers in meters"
    )
    
    if uploaded_files and len(uploaded_files) == 2:
        st.success("✅ 2 images uploaded successfully!")
        
        # Display uploaded images
        col1, col2 = st.columns(2)
        
        images = []
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            images.append(image)
            
            with [col1, col2][i]:
                st.subheader(f"Image {i+1}")
                st.image(image, width=300)
        
        if st.button("🔍 Extract Numbers & Calculate Difference", type="primary"):
            with st.spinner("Processing images with Llama 3.2 Vision..."):
                progress_bar = st.progress(0)
                
                # Extract numbers from both images
                results = []
                for i, (image, uploaded_file) in enumerate(zip(images, uploaded_files)):
                    progress_bar.progress((i + 1) * 40)
                    num = extract_number_from_image(
                        image, 
                        uploaded_file.name, 
                        st.session_state.model,
                        st.session_state.processor,
                        st.session_state.device
                    )
                    results.append(num)
                
                progress_bar.progress(100)
                
                if all(num is not None for num in results):
                    num1, num2 = results
                    diff = abs(num1 - num2)
                    
                    st.success("🎯 **Results:**")
                    
                    # Display results in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Image 1", f"{num1} m")
                    with col2:
                        st.metric("📏 Image 2", f"{num2} m")
                    with col3:
                        st.metric("📐 Difference", f"{diff} m", delta=f"±{diff}")
                    
                    st.balloons()
                else:
                    st.error("❌ Could not extract numbers from one or both images.")
                    st.info("Make sure the numbers are clearly visible and handwritten.")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning(f"⚠️ Please upload exactly 2 images. You uploaded {len(uploaded_files)} image(s).")
    
    else:
        st.info("👆 Upload 2 images above to get started!")
        
        # Add example section
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            1. **Upload** two images containing handwritten numbers
            2. **AI Analysis** using Llama 3.2 Vision model
            3. **Extract** numerical values from both images  
            4. **Calculate** the absolute difference
            
            **Supported formats:** PNG, JPG, JPEG
            **Best results:** Clear, well-lit handwritten numbers
            """)

if __name__ == "__main__":
    main()
