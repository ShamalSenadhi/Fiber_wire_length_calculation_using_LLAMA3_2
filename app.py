import streamlit as st
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import re
import gc

@st.cache_resource
def load_model():
    """Load an open-source vision model with GPU support"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Use LLaVA model (open source, no gating)
        model_id = "llava-hf/llava-1.5-7b-hf"
        
        # Load model with GPU support
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        st.success(f"✅ LLaVA-1.5 Vision model loaded successfully on {device}!")
        return model, processor, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback to smaller model
        try:
            st.info("Trying smaller model...")
            model_id = "microsoft/kosmos-2-patch14-224"
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            processor = AutoProcessor.from_pretrained(model_id)
            st.success(f"✅ Kosmos-2 model loaded on {device}!")
            return model, processor, device
        except Exception as e2:
            st.error(f"Fallback model also failed: {str(e2)}")
            return None, None, None

def extract_number_from_image_llava(image, image_name, model, processor, device):
    """Extract number from image using LLaVA"""
    try:
        # Prepare the conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What handwritten number do you see in this image? Just return the number value only."}
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Move to GPU if available
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=False,
                temperature=0.1
            )
        
        # Decode response
        response = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
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

def extract_number_simple_ocr(image, image_name):
    """Fallback OCR method using easyocr"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Convert PIL to numpy array
        import numpy as np
        img_array = np.array(image)
        
        # Extract text
        results = reader.readtext(img_array)
        
        # Look for numbers
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Only high confidence results
                match = re.search(r'(\d+(?:\.\d+)?)', text)
                if match:
                    number = float(match.group(1))
                    st.write(f"OCR found in {image_name}: {text} (confidence: {confidence:.2f})")
                    return number
        
        st.warning(f"No numbers found via OCR in {image_name}")
        return None
        
    except ImportError:
        st.error("EasyOCR not available. Install with: pip install easyocr")
        return None
    except Exception as e:
        st.error(f"OCR error for {image_name}: {str(e)}")
        return None

def main():
    st.title("🔢 Fiber Length Difference Calculator")
    st.markdown("*Powered by Open-Source Vision AI*")
    
    # Model selection
    method = st.radio(
        "Choose processing method:",
        ["Vision AI (LLaVA)", "OCR (EasyOCR)"],
        help="Vision AI is more accurate but requires more resources. OCR is faster but may be less accurate."
    )
    
    # Load model only if using Vision AI
    if method == "Vision AI (LLaVA)":
        if 'model_loaded' not in st.session_state:
            with st.spinner("Loading LLaVA Vision model... This may take a few minutes."):
                model, processor, device = load_model()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                else:
                    st.session_state.model_loaded = False
        
        if not st.session_state.model_loaded:
            st.error("❌ Vision model failed to load.")
            method = "OCR (EasyOCR)"  # Fallback
            st.info("Falling back to OCR method...")
    
    # GPU info
    if torch.cuda.is_available() and method == "Vision AI (LLaVA)":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        st.success(f"🚀 Running on: {gpu_name} ({gpu_memory}GB VRAM)")
    elif method == "Vision AI (LLaVA)":
        st.info("🖥️ Running on CPU (slower but works)")
    
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
        
        if st.button(f"🔍 Extract Numbers using {method}", type="primary"):
            with st.spinner(f"Processing images with {method}..."):
                progress_bar = st.progress(0)
                
                # Extract numbers from both images
                results = []
                for i, (image, uploaded_file) in enumerate(zip(images, uploaded_files)):
                    progress_bar.progress((i + 1) * 40)
                    
                    if method == "Vision AI (LLaVA)" and st.session_state.get('model_loaded'):
                        num = extract_number_from_image_llava(
                            image, 
                            uploaded_file.name, 
                            st.session_state.model,
                            st.session_state.processor,
                            st.session_state.device
                        )
                    else:
                        num = extract_number_simple_ocr(image, uploaded_file.name)
                    
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
                    st.info("Try the other processing method or ensure numbers are clearly visible.")
    
    elif uploaded_files and len(uploaded_files) != 2:
        st.warning(f"⚠️ Please upload exactly 2 images. You uploaded {len(uploaded_files)} image(s).")
    
    else:
        st.info("👆 Upload 2 images above to get started!")
        
        # Add example section
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Vision AI Mode:**
            - Uses LLaVA-1.5 model (7B parameters)
            - Better understanding of context
            - Works with GPU acceleration
            
            **OCR Mode:**  
            - Uses EasyOCR for text detection
            - Faster processing
            - Good for clear text
            
            **Best results:** Clear, well-lit handwritten numbers
            """)

if __name__ == "__main__":
    main()
