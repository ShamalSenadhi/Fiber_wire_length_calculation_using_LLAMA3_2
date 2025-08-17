import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import re
import warnings
import os
import time
import gc
from functools import wraps
import subprocess
import json
from io import BytesIO
import tempfile
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔍 Dual Cable Measurement Extractor - Local LLaMA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240,147,251,0.3);
    }
    
    .results-section {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .measurement-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .no-measurement-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .comparison-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .llama-response {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    
    .setup-box {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79,172,254,0.4);
    }
    
    .loading-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def timeout_handler(timeout_seconds=60):
    """Decorator to add timeout handling to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Operation timed out or failed: {str(e)}")
                return None
        return wrapper
    return decorator

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        return True, result.stdout
    except FileNotFoundError:
        return False, "Ollama not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "Ollama command timed out"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def check_available_models():
    """Check which vision models are available locally"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    # Check for vision-capable models
                    if any(vision_model in model_name.lower() for vision_model in ['llava', 'bakllava', 'cogvlm', 'moondream']):
                        models.append(model_name)
            return models
        return []
    except Exception:
        return []

def image_to_base64(image):
    """Convert PIL image to base64 string for Ollama"""
    try:
        buffer = BytesIO()
        # Convert to RGB if needed and save as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return None

class LocalLlamaAnalyzer:
    def __init__(self, model_name="llava"):
        """Initialize Local LLaMA analyzer"""
        self.model_name = model_name
        
    def analyze_cable_image(self, image, method_name):
        """Analyze cable image using local Ollama LLaMA vision model"""
        try:
            # Convert image to base64
            img_base64 = image_to_base64(image)
            if not img_base64:
                return {"measurements": [], "analysis": "Failed to process image"}
            
            # Create focused prompt for cable measurement
            prompt = f"""Analyze this cable image (processed with {method_name} enhancement). 

TASK: Find and extract ONLY length measurements from cables, labels, or markings.

LOOK FOR:
- Numbers followed by: m, meter, meters, cm, mm, mtr, mtrs
- Text printed on cables or labels
- Measurement markings or tags
- Handwritten measurements

INSTRUCTIONS:
1. Scan the entire image carefully
2. Extract only numerical measurements with units
3. Convert everything to meters (m)
4. List each measurement you find
5. If no measurements visible, say "No measurements detected"

FORMAT YOUR RESPONSE:
- Measurements found: [list them as "123m" or "12.5m"]
- Image quality: [brief assessment]
- Confidence: [High/Medium/Low]

Analyze now:"""

            # Call local Ollama
            response = self._call_ollama_vision(prompt, img_base64)
            return self._parse_llama_response(response, method_name)
            
        except Exception as e:
            return {"measurements": [], "analysis": f"Error: {str(e)}"}
    
    def _call_ollama_vision(self, prompt, img_base64):
        """Call Ollama vision model locally"""
        try:
            # Prepare the request data for Ollama API
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "top_p": 0.9,
                    "num_predict": 200   # Limit response length
                }
            }
            
            # Use curl to call Ollama API (more reliable than requests for local)
            import json
            import subprocess
            
            # Save data to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f)
                temp_file = f.name
            
            try:
                # Call Ollama API using curl
                result = subprocess.run([
                    'curl', '-X', 'POST',
                    'http://localhost:11434/api/generate',
                    '-H', 'Content-Type: application/json',
                    '-d', f'@{temp_file}'
                ], capture_output=True, text=True, timeout=120)
                
                # Clean up temp file
                os.unlink(temp_file)
                
                if result.returncode == 0 and result.stdout:
                    try:
                        response_data = json.loads(result.stdout)
                        return response_data.get('response', 'No response from model')
                    except json.JSONDecodeError:
                        return result.stdout
                else:
                    return f"Ollama error: {result.stderr or 'Unknown error'}"
                    
            except subprocess.TimeoutExpired:
                os.unlink(temp_file)
                return "Ollama request timed out. Model may be too large or busy."
            except Exception as e:
                os.unlink(temp_file)
                return f"Error calling Ollama: {str(e)}"
                
        except Exception as e:
            return f"Local Ollama call failed: {str(e)}"
    
    def _parse_llama_response(self, response, method_name):
        """Parse LLaMA response to extract measurements"""
        try:
            measurements = []
            
            # Extract measurements from LLaMA response using multiple patterns
            patterns = [
                r'(\d+(?:\.\d+)?)\s*m(?:\s|$|[^a-zA-Z])',  # "645m", "12.5m"
                r'(\d+(?:\.\d+)?)\s*meter[s]?',            # "645 meters"
                r'(\d+(?:\.\d+)?)\s*mtr[s]?',              # "645 mtrs"
                r'Measurements?.*?(\d+(?:\.\d+)?)\s*m',     # "Measurements: 123m"
                r'found.*?(\d+(?:\.\d+)?)\s*m',            # "found 123m"
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    value = match.group(1)
                    try:
                        # Validate it's a reasonable measurement
                        num_value = float(value)
                        if 0.1 <= num_value <= 10000:  # Reasonable cable length range
                            measurements.append(f"{value}m")
                    except ValueError:
                        continue
            
            # Convert other units found in response
            # mm to meters
            mm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*mm', response, re.IGNORECASE)
            for match in mm_matches:
                try:
                    mm_value = float(match.group(1))
                    m_value = mm_value / 1000
                    if 0.1 <= m_value <= 10000:  # Reasonable range
                        measurements.append(f"{m_value:.3f}m")
                except ValueError:
                    continue
            
            # cm to meters
            cm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*cm', response, re.IGNORECASE)
            for match in cm_matches:
                try:
                    cm_value = float(match.group(1))
                    m_value = cm_value / 100
                    if 0.01 <= m_value <= 10000:  # Reasonable range
                        measurements.append(f"{m_value:.2f}m")
                except ValueError:
                    continue
            
            # Remove duplicates and sort
            measurements = list(set(measurements))
            measurements.sort(key=lambda x: float(x.replace('m', '')))
            
            # Clean up response for display
            clean_response = response.replace('\\n', '\n').strip()
            
            return {
                "measurements": measurements,
                "analysis": clean_response[:500] + "..." if len(clean_response) > 500 else clean_response,
                "raw_response": response,
                "method": method_name
            }
            
        except Exception as e:
            return {
                "measurements": [],
                "analysis": f"Parse error: {str(e)}\nRaw response: {response[:200]}...",
                "raw_response": response,
                "method": method_name
            }

@timeout_handler(30)
def enhance_for_measurement(img, method):
    """Apply image enhancement specifically for measurement extraction"""
    try:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Reduce image size for faster processing
        height, width = gray.shape
        if height > 800 or width > 800:
            scale = min(800/height, 800/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        if method == 'High Contrast':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
            
        elif method == 'Cable Optimized':
            enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=40)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
            enhanced = clahe.apply(enhanced)
            
        elif method == 'Denoised':
            enhanced = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
            
        elif method == 'Edge Enhanced':
            enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
        elif method == 'Handwriting Optimized':
            enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
        else:  # Original
            enhanced = gray
        
        return Image.fromarray(enhanced)
        
    except Exception as e:
        st.warning(f"Enhancement failed for {method}: {str(e)}")
        return img

@timeout_handler(240)  # 4 minutes for all 6 methods
def process_single_image(img, analyzer, image_name):
    """Process a single image with all 6 enhancement methods using local LLaMA"""
    methods = ['Original', 'High Contrast', 'Cable Optimized', 'Denoised', 'Edge Enhanced', 'Handwriting Optimized']
    
    results = {}
    all_measurements = set()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, method in enumerate(methods):
            progress = (idx + 1) / len(methods)
            progress_bar.progress(progress)
            status_text.text(f"🤖 {image_name} - {method} with {analyzer.model_name} ({idx+1}/{len(methods)})")
            
            # Apply enhancement
            enhanced_img = enhance_for_measurement(img, method)
            
            if enhanced_img is None:
                continue
            
            # Analyze with Local LLaMA
            try:
                llama_result = analyzer.analyze_cable_image(enhanced_img, method)
                
                measurements = llama_result.get("measurements", [])
                all_measurements.update(measurements)
                
                results[method] = {
                    'image': enhanced_img,
                    'measurements': measurements,
                    'llama_analysis': llama_result.get("analysis", ""),
                    'raw_response': llama_result.get("raw_response", "")
                }
                
            except Exception as e:
                st.warning(f"LLaMA analysis failed for {method}: {str(e)}")
                results[method] = {
                    'image': enhanced_img,
                    'measurements': [],
                    'llama_analysis': f"Error: {str(e)}",
                    'raw_response': ""
                }
            
            gc.collect()
            time.sleep(1)  # Give local model time between requests
        
        progress_bar.empty()
        status_text.empty()
        
        return results, all_measurements
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Processing failed for {image_name}: {str(e)}")
        return {}, set()

def calculate_length_difference(measurements1, measurements2):
    """Calculate the length difference between two sets of measurements"""
    try:
        def get_numeric_values(measurements):
            return [float(m.replace('m', '')) for m in measurements if m.replace('m', '').replace('.', '').isdigit()]
        
        nums1 = get_numeric_values(measurements1)
        nums2 = get_numeric_values(measurements2)
        
        if not nums1 and not nums2:
            return {'analysis_possible': False, 'reason': 'No valid measurements found in either image'}
        elif not nums1:
            return {'analysis_possible': False, 'reason': 'No valid measurements found in Image 1'}
        elif not nums2:
            return {'analysis_possible': False, 'reason': 'No valid measurements found in Image 2'}
        
        primary1 = max(nums1)
        primary2 = max(nums2)
        difference = primary1 - primary2
        abs_difference = abs(difference)
        
        primary1_str = f"{primary1}m"
        primary2_str = f"{primary2}m"
        
        if difference > 0:
            diff_display = f"+{abs_difference}m"
            comparison_text = "Image 1 is longer than Image 2"
        elif difference < 0:
            diff_display = f"-{abs_difference}m"
            comparison_text = "Image 2 is longer than Image 1"
        else:
            diff_display = "0m"
            comparison_text = "Both images have equal length"
        
        percentage_diff = None
        if primary2 != 0:
            percentage = abs((difference / primary2) * 100)
            if percentage >= 0.1:
                percentage_diff = f"{percentage:.1f}%"
        
        return {
            'analysis_possible': True,
            'image1_primary': primary1_str,
            'image2_primary': primary2_str,
            'difference_value': difference,
            'difference_display': diff_display,
            'comparison_text': comparison_text,
            'percentage_difference': percentage_diff
        }
        
    except Exception as e:
        return {'analysis_possible': False, 'reason': f'Error calculating difference: {str(e)}'}

def display_results_grid(results, image_name):
    """Display results in a grid format with all 6 methods"""
    st.markdown(f"""
    <div class="results-section">
        <h2>🖼️ {image_name} Analysis Results (Local LLaMA)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    methods = ['Original', 'High Contrast', 'Cable Optimized', 'Denoised', 'Edge Enhanced', 'Handwriting Optimized']
    
    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            method_idx = row * 3 + col_idx
            if method_idx < len(methods):
                method = methods[method_idx]
                result = results.get(method, {'image': None, 'measurements': [], 'llama_analysis': ''})
                
                with cols[col_idx]:
                    st.markdown(f"**🎨 {method}**")
                    if result['image'] is not None:
                        st.image(result['image'], use_column_width=True)
                    
                    if result['measurements']:
                        measurements_text = ', '.join(result['measurements'])
                        st.markdown(f"""
                        <div class="measurement-box">
                            📏 {measurements_text}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="no-measurement-box">
                            ❌ No measurements detected
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show LLaMA analysis
                    if result.get('llama_analysis'):
                        with st.expander(f"🤖 LLaMA Analysis - {method}"):
                            st.markdown(f"""
                            <div class="llama-response">
                                {result['llama_analysis']}
                            </div>
                            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Dual Cable Measurement Extractor</h1>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin: 10px 0;">
            🦙 Powered by Local LLaMA Vision - No API Keys Required
        </div>
        <p>Upload 2 cable images to extract and compare measurements using local AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check Ollama installation
    ollama_ok, ollama_msg = check_ollama_installation()
    
    if not ollama_ok:
        st.markdown("""
        <div class="setup-box">
            <h3>🛠️ Setup Required: Install Ollama</h3>
            <p>Ollama is not installed or not running. Please follow these steps:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Installation Steps:**
        
        1. **Install Ollama:**
           - **Linux/Mac**: `curl -fsSL https://ollama.ai/install.sh | sh`
           - **Windows**: Download from https://ollama.ai/download
        
        2. **Pull a vision model:**
           ```bash
           ollama pull llava
           # or
           ollama pull llava:7b
           ollama pull llava:13b
           ```
        
        3. **Start Ollama server:**
           ```bash
           ollama serve
           ```
        
        4. **Refresh this page**
        """)
        
        st.info("💡 LLaVA models range from 4GB (7b) to 26GB (34b). Choose based on your system capabilities.")
        st.stop()
    
    # Check available models
    available_models = check_available_models()
    
    if not available_models:
        st.warning("⚠️ No vision models found. Please pull a vision model:")
        st.code("ollama pull llava")
        st.info("Available vision models: llava, llava:7b, llava:13b, llava:34b, bakllava, moondream")
        st.stop()
    
    st.success(f"✅ Ollama running with {len(available_models)} vision model(s) available")
    
    # Model selection
    selected_model = st.selectbox(
        "🦙 Select Vision Model",
        available_models,
        help="Choose the local LLaMA vision model to use"
    )
    
    # Model info
    model_info = {
        "llava": "General-purpose vision model (4GB)",
        "llava:7b": "7B parameter model (4.5GB)",
        "llava:13b": "13B parameter model (8GB)",
        "llava:34b": "34B parameter model (20GB)", 
        "bakllava": "Optimized for text/document analysis",
        "moondream": "Lightweight vision model (1.6GB)"
    }
    
    st.info(f"🤖 Using: {selected_model} - {model_info.get(selected_model, 'Local vision model')}")
    
    # Initialize analyzer
    try:
        analyzer = LocalLlamaAnalyzer(selected_model)
        st.success("🦙 Local LLaMA Vision Analyzer ready!")
    except Exception as e:
        st.error(f"❌ Error initializing analyzer: {str(e)}")
        st.stop()
    
    # Sidebar with features
    with st.sidebar:
        st.markdown("### ✨ Local LLaMA Features")
        st.markdown("""
        - 🦙 **100% Local**: No internet required
        - 🔒 **Privacy First**: Images never leave your computer
        - 🤖 **AI Vision**: Advanced multimodal understanding
        - 📏 **Smart Detection**: Context-aware measurement extraction
        - 🎨 **12 Total Analyses**: 6 methods per image
        - 💬 **Natural Language**: AI explains what it sees
        - ⚡ **No API Costs**: Completely free to use
        """)
        
        st.markdown("### 🎯 Model Comparison")
        st.markdown("""
        - 🦙 **LLaVA 7B**: Best balance (4.5GB RAM)
        - 🧠 **LLaVA 13B**: More accurate (8GB RAM)
        - 🚀 **LLaVA 34B**: Highest quality (20GB RAM)
        - 🍰 **BakLLaVA**: Great for text/documents
        - 🌙 **Moondream**: Fastest, lightweight
        """)
        
        st.markdown("### ⚠️ Performance Tips")
        st.markdown("""
        - Larger models = better accuracy
        - Processing takes 30-60 seconds per method
        - Close other apps for better performance
        - GPU acceleration auto-detected
        - First run may take longer (model loading)
        """)
    
    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Image 1</h3>
            <p>First cable image for local AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file1 = st.file_uploader("Choose first cable image", type=['png', 'jpg', 'jpeg'], key="file1")
    
    with col2:
        st.markdown("""
        <div class="upload-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📁 Upload Image 2</h3>
            <p>Second cable image for comparison</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file2 = st.file_uploader("Choose second cable image", type=['png', 'jpg', 'jpeg'], key="file2")
    
    # Process button
    if uploaded_file1 is not None and uploaded_file2 is not None:
        if st.button("🦙 Analyze Both Images with Local LLaMA", key="analyze_btn"):
            try:
                # Load images
                image1 = Image.open(uploaded_file1)
                image2 = Image.open(uploaded_file2)
                
                st.markdown("---")
                st.markdown("## 🔄 Processing Images with Local LLaMA Vision...")
                st.info(f"⏳ Processing with {selected_model}. This takes 4-8 minutes total (30-60 seconds per method). Please be patient...")
                
                # Process both images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🖼️ Processing Image 1")
                    results1, measurements1 = process_single_image(image1, analyzer, "Image 1")
                    
                with col2:
                    st.markdown("### 🖼️ Processing Image 2") 
                    results2, measurements2 = process_single_image(image2, analyzer, "Image 2")
                
                if not results1 or not results2:
                    st.error("❌ Processing failed. Please check that Ollama is running and try again.")
                    return
                
                st.markdown("---")
                
                # Display results
                display_results_grid(results1, "Image 1")
                st.markdown("---")
                display_results_grid(results2, "Image 2")
                
                # Comparison Summary
                length_difference = calculate_length_difference(measurements1, measurements2)
                common_measurements = list(measurements1.intersection(measurements2))
                common_measurements.sort(key=lambda x: float(x.replace('m', '')))
                
                st.markdown("""
                <div class="comparison-summary">
                    <h3>📊 Local LLaMA Comparative Analysis Summary</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🖼️ Image 1 Summary")
                    st.write(f"**Methods analyzed:** {len(results1)}")
                    st.write(f"**Measurements found:** {len(measurements1)}")
                    if measurements1:
                        measurements_str = ', '.join(sorted(measurements1, key=lambda x: float(x.replace('m', ''))))
                        st.write(f"**AI-detected measurements:** {measurements_str}")
                    else:
                        st.write("**AI-detected measurements:** None detected")
                
                with col2:
                    st.markdown("#### 🖼️ Image 2 Summary")
                    st.write(f"**Methods analyzed:** {len(results2)}")
                    st.write(f"**Measurements found:** {len(measurements2)}")
                    if measurements2:
                        measurements_str = ', '.join(sorted(measurements2, key=lambda x: float(x.replace('m', ''))))
                        st.write(f"**AI-detected measurements:** {measurements_str}")
                    else:
                        st.write("**AI-detected measurements:** None detected")
                
                # Length Difference Analysis
                st.markdown("#### 📏 Length Difference Analysis")
                if length_difference['analysis_possible']:
                    st.write(f"**Image 1 Primary Length:** {length_difference['image1_primary']}")
                    st.write(f"**Image 2 Primary Length:** {length_difference['image2_primary']}")
                    st.write(f"**Difference:** {length_difference['difference_display']}")
                    st.write(f"**AI Analysis:** {length_difference['comparison_text']}")
                    if length_difference['percentage_difference']:
                        st.write(f"**Percentage Difference:** {length_difference['percentage_difference']}")
                else:
                    st.warning(f"⚠️ {length_difference['reason']}")
                
                # Overall Comparison
                st.markdown("#### 🔄 Overall Local AI Analysis")
                st.write(f"**Total methods processed:** {len(results1) + len(results2)} ({len(results1)} + {len(results2)})")
                st.write(f"**Combined unique measurements:** {len(measurements1.union(measurements2))}")
                st.write(f"**Model used:** {selected_model}")
                
                if common_measurements:
                    st.write(f"**Common measurements found:** {', '.join(common_measurements)}")
                else:
                    st.write("**Common measurements:** None found")
                
                # Performance stats
                st.markdown("#### 📊 Processing Statistics")
                total_images_processed = len(results1) + len(results2)
                successful_analyses = sum(1 for r in results1.values() if r.get('measurements')) + sum(1 for r in results2.values() if r.get('measurements'))
                
                st.write(f"**Total enhanced images processed:** {total_images_processed}")
                st.write(f"**Successful AI analyses:** {successful_analyses}/{total_images_processed}")
                st.write(f"**Success rate:** {(successful_analyses/total_images_processed)*100:.1f}%")
                
                st.success("✅ Local LLaMA analysis completed successfully!")
                st.info("🦙 All processing done locally on your machine - no data sent to external servers!")
                
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.info("💡 Try restarting Ollama (`ollama serve`) or selecting a different model.")
    
    elif uploaded_file1 is not None or uploaded_file2 is not None:
        st.info("⚠️ Please upload both images for comparative analysis")
    else:
        st.info("📋 Upload both cable images to start local AI measurement extraction and comparison")
    
    # Footer with helpful info
    with st.expander("🔧 Troubleshooting & Tips"):
        st.markdown("""
        **Common Issues:**
        
        1. **"Ollama not found"**: Install Ollama from https://ollama.ai
        2. **"No models available"**: Run `ollama pull llava`
        3. **Processing timeout**: Try a smaller model like `moondream`
        4. **Out of memory**: Close other applications or use `llava:7b`
        5. **Slow processing**: Normal for CPU - GPU acceleration is automatic if available
        
        **Model Recommendations:**
        - **Fast computer (16GB+ RAM)**: `llava:13b` or `llava:34b`
        - **Medium computer (8GB RAM)**: `llava:7b` or `llava`
        - **Low-end computer**: `moondream`
        - **Best for text**: `bakllava`
        
        **Performance Tips:**
        - First run downloads model (can take 10+ minutes)
        - Subsequent runs are much faster
        - GPU automatically used if available (NVIDIA/Apple Silicon)
        - Close browser tabs and other apps for better performance
        - Smaller images process faster (images auto-resized to 800px max)
        """)

if __name__ == "__main__":
    main()
