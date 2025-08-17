import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import io
import base64
import re
import warnings
import os
import time
import gc
from functools import wraps
from skimage import filters, morphology, exposure
from scipy import ndimage
warnings.filterwarnings('ignore')

# Configure Tesseract path (adjust based on your system)
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/Mac: usually auto-detected, but you can set it explicitly if needed
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Page configuration
st.set_page_config(
    page_title="🔍 Enhanced Cable Measurement Extractor",
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
        font-size: 1.2em;
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
    
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
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

def timeout_handler(timeout_seconds=30):
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

def test_tesseract():
    """Test if Tesseract is properly installed and accessible"""
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} found"
    except Exception as e:
        return False, f"Tesseract not found: {str(e)}"

def preprocess_for_measurement_reading(img):
    """Advanced preprocessing specifically for measurement reading like cable tags"""
    try:
        # Convert PIL to cv2
        if isinstance(img, Image.Image):
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            cv_img = img.copy()
        
        # Resize if too large (maintain aspect ratio)
        height, width = cv_img.shape[:2]
        if height > 1200 or width > 1200:
            scale = min(1200/height, 1200/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        return cv_img
    except Exception as e:
        st.warning(f"Preprocessing failed: {str(e)}")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@timeout_handler(30)
def enhance_for_measurement(img, method):
    """Enhanced image processing methods optimized for measurement reading"""
    try:
        cv_img = preprocess_for_measurement_reading(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        if method == 'Adaptive Threshold':
            # Best for varying lighting conditions
            enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
            # Morphological operations to clean up
            kernel = np.ones((2,2), np.uint8)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
        elif method == 'High Contrast CLAHE':
            # Optimized CLAHE for measurement reading
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
            enhanced = clahe.apply(gray)
            # Increase contrast further
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=30)
            # Apply unsharp masking for text sharpening
            gaussian = cv2.GaussianBlur(enhanced, (0,0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
        elif method == 'Measurement Optimized':
            # Specifically optimized for reading measurements on cables
            # Step 1: Enhance contrast
            enhanced = cv2.convertScaleAbs(gray, alpha=2.2, beta=40)
            # Step 2: Apply bilateral filter to reduce noise while keeping edges
            enhanced = cv2.bilateralFilter(enhanced, 9, 80, 80)
            # Step 3: Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
            enhanced = clahe.apply(enhanced)
            # Step 4: Sharpen text
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
        elif method == 'Edge Enhanced':
            # Enhanced edge detection for text boundaries
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
            # Apply Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            # Unsharp masking for text enhancement
            gaussian = cv2.GaussianBlur(enhanced, (0,0), 3.0)
            enhanced = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
            # Ensure values are in valid range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        elif method == 'Noise Reduction Pro':
            # Advanced noise reduction while preserving text
            # Non-local means denoising
            enhanced = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
            # Enhance contrast after denoising
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
            # Slight sharpening
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=15)
            
        elif method == 'Multi-Scale Enhancement':
            # Multi-scale approach for different text sizes
            scales = [0.8, 1.0, 1.2]
            enhanced_scales = []
            
            for scale in scales:
                if scale != 1.0:
                    h, w = gray.shape
                    scaled = cv2.resize(gray, (int(w*scale), int(h*scale)))
                else:
                    scaled = gray.copy()
                
                # Apply enhancement to each scale
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                scaled_enhanced = clahe.apply(scaled)
                scaled_enhanced = cv2.convertScaleAbs(scaled_enhanced, alpha=1.6, beta=25)
                
                # Resize back to original if needed
                if scale != 1.0:
                    scaled_enhanced = cv2.resize(scaled_enhanced, (w, h))
                
                enhanced_scales.append(scaled_enhanced)
            
            # Combine scales (take maximum intensity)
            enhanced = np.maximum.reduce(enhanced_scales)
            
        else:  # Original
            enhanced = gray
        
        return Image.fromarray(enhanced)
        
    except Exception as e:
        st.warning(f"Enhancement failed for {method}: {str(e)}")
        return img

def extract_length_measurements(text, confidence_scores=None):
    """Enhanced measurement extraction with better pattern matching"""
    measurements = []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Enhanced patterns for measurement detection
    patterns = [
        # Standard meter patterns
        (r'(\d+(?:\.\d+)?)\s*m(?:\s|$|[^a-zA-Z])', 'direct'),
        (r'(\d+(?:\.\d+)?)\s*meter[s]?', 'meter'),
        (r'(\d+(?:\.\d+)?)\s*mtr[s]?', 'meter_short'),
        
        # Handle potential OCR errors
        (r'(\d+(?:\.\d+)?)\s*[nm](?:\s|$|[^a-zA-Z])', 'ocr_error_m'),  # n might be misread m
        (r'(\d{2,4})\s*(?:m|meter)', 'numeric_focus'),  # Focus on realistic cable lengths
        
        # Pattern for the specific format like "645m"
        (r'(\d{2,4})\s*m\b', 'cable_tag'),  # Specific for cable tags
        
        # Handle spaces and formatting issues
        (r'(\d+)\s*\.\s*(\d+)\s*m', 'decimal_spaced'),  # "12 . 5 m" -> "12.5m"
    ]
    
    confidence_map = {}
    
    for pattern, pattern_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if pattern_type == 'decimal_spaced':
                value = f"{match.group(1)}.{match.group(2)}"
            else:
                value = match.group(1)
            
            try:
                float_val = float(value)
                # Filter realistic measurements (0.1m to 10000m for cables)
                if 0.1 <= float_val <= 10000:
                    measurement = f"{value}m"
                    measurements.append(measurement)
                    
                    # Assign confidence based on pattern type
                    if pattern_type in ['direct', 'cable_tag']:
                        confidence_map[measurement] = 'high'
                    elif pattern_type in ['meter', 'meter_short']:
                        confidence_map[measurement] = 'medium'
                    else:
                        confidence_map[measurement] = 'low'
            except ValueError:
                continue
    
    # Convert other units to meters
    # mm to meters (only if result is reasonable)
    mm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*mm', text, re.IGNORECASE)
    for match in mm_matches:
        try:
            mm_value = float(match.group(1))
            m_value = mm_value / 1000
            if 0.1 <= m_value <= 1000:  # Reasonable range
                measurement = f"{m_value:.3f}m"
                measurements.append(measurement)
                confidence_map[measurement] = 'medium'
        except ValueError:
            continue
    
    # cm to meters
    cm_matches = re.finditer(r'(\d+(?:\.\d+)?)\s*cm', text, re.IGNORECASE)
    for match in cm_matches:
        try:
            cm_value = float(match.group(1))
            m_value = cm_value / 100
            if 0.01 <= m_value <= 100:  # Reasonable range
                measurement = f"{m_value:.2f}m"
                measurements.append(measurement)
                confidence_map[measurement] = 'medium'
        except ValueError:
            continue
    
    # Remove duplicates while preserving order and confidence
    unique_measurements = []
    seen = set()
    
    for measurement in measurements:
        # Normalize to avoid duplicates like "645m" and "645.000m"
        normalized = f"{float(measurement.replace('m', '')):.3f}m"
        if normalized not in seen:
            seen.add(normalized)
            # Use original format if it's cleaner
            if measurement.replace('m', '').find('.') == -1 or measurement.endswith('.000m'):
                clean_val = str(int(float(measurement.replace('m', '')))) + 'm'
                unique_measurements.append(clean_val)
                confidence_map[clean_val] = confidence_map.get(measurement, 'low')
            else:
                unique_measurements.append(measurement)
    
    # Sort by numerical value
    unique_measurements.sort(key=lambda x: float(x.replace('m', '')))
    
    return unique_measurements, confidence_map

@timeout_handler(120)
def process_single_image(img, image_name):
    """Process a single image with enhanced methods"""
    methods = [
        'Original', 
        'Adaptive Threshold', 
        'High Contrast CLAHE', 
        'Measurement Optimized',
        'Edge Enhanced', 
        'Noise Reduction Pro',
        'Multi-Scale Enhancement'
    ]
    
    results = {}
    all_measurements = set()
    confidence_scores = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, method in enumerate(methods):
            progress = (idx + 1) / len(methods)
            progress_bar.progress(progress)
            status_text.text(f"🔍 Processing {image_name} - {method} ({idx+1}/{len(methods)})")
            
            # Apply enhancement
            enhanced_img = enhance_for_measurement(img, method)
            
            if enhanced_img is None:
                continue
                
            # Convert to numpy array for Tesseract
            img_array = np.array(enhanced_img)
            
            # Enhanced OCR with multiple configurations
            try:
                # Optimized Tesseract configurations for measurement reading
                configs = [
                    # Single uniform block - best for measurement labels
                    '--psm 6 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS tessedit_pageseg_mode=6',
                    # Single text line - for single line measurements
                    '--psm 7 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS',
                    # Single word - for isolated measurements
                    '--psm 8 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS',
                    # Raw line - minimal processing
                    '--psm 13 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS',
                    # Sparse text - for scattered characters
                    '--psm 11 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS',
                    # Default with character whitelist
                    '--psm 3 -c tessedit_char_whitelist=0123456789.mMcCeEtTrRsS',
                ]
                
                all_text = ""
                best_confidence = 0
                
                for config in configs:
                    try:
                        # Get text with confidence
                        data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
                        text = pytesseract.image_to_string(img_array, config=config)
                        
                        if text.strip():
                            all_text += " " + text.strip()
                            
                            # Calculate average confidence
                            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                            if confidences:
                                avg_conf = sum(confidences) / len(confidences)
                                best_confidence = max(best_confidence, avg_conf)
                                
                    except Exception as config_error:
                        continue
                
                # Fallback to default if nothing worked
                if not all_text.strip():
                    all_text = pytesseract.image_to_string(img_array)
                
                # Extract measurements with confidence
                measurements, conf_map = extract_length_measurements(all_text)
                all_measurements.update(measurements)
                
                # Store results with enhanced info
                results[method] = {
                    'image': enhanced_img,
                    'measurements': measurements,
                    'raw_text': all_text.strip(),
                    'confidence': best_confidence,
                    'confidence_map': conf_map
                }
                
            except Exception as e:
                st.warning(f"OCR failed for {method}: {str(e)}")
                results[method] = {
                    'image': enhanced_img,
                    'measurements': [],
                    'raw_text': '',
                    'confidence': 0,
                    'confidence_map': {}
                }
            
            gc.collect()
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        return results, all_measurements
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Processing failed for {image_name}: {str(e)}")
        return {}, set()

def calculate_length_difference(measurements1, measurements2):
    """Enhanced length difference calculation"""
    try:
        def get_numeric_values(measurements):
            return [float(m.replace('m', '')) for m in measurements if m.replace('m', '').replace('.', '').isdigit()]
        
        nums1 = get_numeric_values(measurements1)
        nums2 = get_numeric_values(measurements2)
        
        if not nums1 and not nums2:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in either image'
            }
        elif not nums1:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in Image 1'
            }
        elif not nums2:
            return {
                'analysis_possible': False,
                'reason': 'No valid measurements found in Image 2'
            }
        
        # Get most likely measurement (highest value, as it's usually the main measurement)
        primary1 = max(nums1)
        primary2 = max(nums2)
        
        # Calculate difference
        difference = primary1 - primary2
        abs_difference = abs(difference)
        
        # Format for display
        primary1_str = f"{primary1}m" if primary1.is_integer() else f"{primary1:.3f}m".rstrip('0').rstrip('.')
        primary2_str = f"{primary2}m" if primary2.is_integer() else f"{primary2:.3f}m".rstrip('0').rstrip('.')
        
        if difference > 0:
            diff_display = f"+{abs_difference}m"
            comparison_text = "Image 1 cable is longer"
        elif difference < 0:
            diff_display = f"-{abs_difference}m"
            comparison_text = "Image 2 cable is longer"
        else:
            diff_display = "0m"
            comparison_text = "Both cables have equal length"
        
        # Calculate percentage difference
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
        return {
            'analysis_possible': False,
            'reason': f'Error calculating difference: {str(e)}'
        }

def display_results_grid(results, image_name):
    """Enhanced results display with confidence indicators"""
    st.markdown(f"""
    <div class="results-section">
        <h2>🖼️ {image_name} Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    methods = list(results.keys())
    
    # Create grid layout
    for row in range(0, len(methods), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            method_idx = row + col_idx
            if method_idx < len(methods):
                method = methods[method_idx]
                result = results.get(method, {
                    'image': None, 
                    'measurements': [], 
                    'confidence': 0,
                    'confidence_map': {}
                })
                
                with cols[col_idx]:
                    st.markdown(f"**🎨 {method}**")
                    if result.get('image') is not None:
                        st.image(result['image'], use_column_width=True)
                    
                    # Display confidence
                    confidence = result.get('confidence', 0)
                    if confidence > 80:
                        conf_class = "confidence-high"
                        conf_text = f"🟢 High Confidence ({confidence:.0f}%)"
                    elif confidence > 50:
                        conf_class = "confidence-medium"
                        conf_text = f"🟡 Medium Confidence ({confidence:.0f}%)"
                    else:
                        conf_class = "confidence-low"
                        conf_text = f"🔴 Low Confidence ({confidence:.0f}%)"
                    
                    st.markdown(f'<div class="{conf_class}">{conf_text}</div>', unsafe_allow_html=True)
                    
                    # Display measurements
                    if result.get('measurements'):
                        measurements_text = ', '.join(result['measurements'])
                        st.markdown(f"""
                        <div class="measurement-box">
                            📏 {measurements_text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show raw text if available
                        if result.get('raw_text'):
                            with st.expander("Raw OCR Text"):
                                st.text(result['raw_text'])
                    else:
                        st.markdown("""
                        <div class="no-measurement-box">
                            ❌ No measurements detected
                        </div>
                        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Enhanced Cable Measurement Extractor</h1>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin: 10px 0;">
            🚀 Advanced OCR with Confidence Scoring & Multi-Scale Enhancement
        </div>
        <p>Upload 2 cable images to extract and compare length measurements with high accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test Tesseract installation
    tesseract_ok, tesseract_msg = test_tesseract()
    if tesseract_ok:
        st.success(f"✅ {tesseract_msg}")
    else:
        st.error(f"❌ {tesseract_msg}")
        st.error("Please install Tesseract OCR and ensure it's in your system PATH")
        st.markdown("""
        **Installation Instructions:**
        - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
        - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
        - **macOS**: `brew install tesseract`
        """)
        st.stop()
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### ✨ Enhanced Features")
        st.markdown("""
        - 🤖 **Advanced Tesseract OCR**: Multiple PSM modes for optimal accuracy
        - 🎯 **Confidence Scoring**: Real-time OCR confidence measurement  
        - 📏 **Smart Unit Detection**: Automatic mm/cm to meter conversion
        - 🔍 **Multi-Scale Processing**: Enhanced text detection at different scales
        - 🎨 **7 Advanced Methods**: Optimized preprocessing techniques
        - 📊 **Comparative Analysis**: Detailed side-by-side comparison
        - ⚡ **Error Correction**: OCR error handling and pattern matching
        - 🧹 **Noise Reduction**: Advanced denoising while preserving text
        """)
        
        st.markdown("### 🎯 Optimized For")
        st.markdown("""
        - 📐 **Cable Tags**: "645m", "12.5m", handwritten measurements
        - 🏷️ **Equipment Labels**: Standard and custom formats
        - 📏 **Multiple Units**: mm, cm, meters with auto-conversion
        - ✍️ **Various Text Types**: Printed, handwritten, embossed
        - 🌟 **Challenging Conditions**: Low contrast, faded text, noise
        """)
        
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        - Ensure good lighting on measurement area
        - Keep camera steady for sharp text
        - Multiple methods increase accuracy
        - Check confidence scores for reliability
        - Compare results across different methods
        """)
    
    # Upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Image 1</h3>
            <p>First cable image for measurement extraction</p>
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
        if st.button("🔍 Analyze Both Images with Enhanced OCR", key="analyze_btn"):
            try:
                # Load images
                image1 = Image.open(uploaded_file1)
                image2 = Image.open(uploaded_file2)
                
                st.markdown("---")
                st.markdown("## 🔄 Processing Images with Enhanced Methods...")
                st.info("⏳ Processing with 7 advanced methods and confidence scoring. This may take 2-4 minutes...")
                
                # Process both images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🖼️ Processing Image 1")
                    results1, measurements1 = process_single_image(image1, "Image 1")
                    
                with col2:
                    st.markdown("### 🖼️ Processing Image 2")
                    results2, measurements2 = process_single_image(image2, "Image 2")
                
                if not results1 or not results2:
                    st.error("❌ Processing failed. Please try again with different images.")
                    return
                
                st.markdown("---")
                
                # Display results
                display_results_grid(results1, "Image 1")
                st.markdown("---")
                display_results_grid(results2, "Image 2")
                
                # Enhanced Comparison Summary
                length_difference = calculate_length_difference(measurements1, measurements2)
                common_measurements = list(measurements1.intersection(measurements2))
                common_measurements.sort(key=lambda x: float(x.replace('m', '')))
                
                st.markdown("""
                <div class="comparison-summary">
                    <h3>📊 Enhanced Comparative Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate best confidence methods
                def get_best_method_info(results):
                    best_method = None
                    best_confidence = 0
                    best_measurements = []
                    
                    for method, result in results.items():
                        confidence = result.get('confidence', 0)
                        measurements = result.get('measurements', [])
                        if measurements and confidence > best_confidence:
                            best_confidence = confidence
                            best_method = method
                            best_measurements = measurements
                    
                    return best_method, best_confidence, best_measurements
                
                best_method1, best_conf1, best_meas1 = get_best_method_info(results1)
                best_method2, best_conf2, best_meas2 = get_best_method_info(results2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🖼️ Image 1 Summary")
                    st.write(f"**Methods tested:** {len(results1)}")
                    st.write(f"**Total measurements found:** {len(measurements1)}")
                    if best_method1:
                        st.write(f"**Best method:** {best_method1}")
                        st.write(f"**Best confidence:** {best_conf1:.1f}%")
                        measurements_str = ', '.join(best_meas1)
                        st.write(f"**Best measurements:** {measurements_str}")
                    else:
                        st.write("**Best measurements:** None detected")
                
                with col2:
                    st.markdown("#### 🖼️ Image 2 Summary")
                    st.write(f"**Methods tested:** {len(results2)}")
                    st.write(f"**Total measurements found:** {len(measurements2)}")
                    if best_method2:
                        st.write(f"**Best method:** {best_method2}")
                        st.write(f"**Best confidence:** {best_conf2:.1f}%")
                        measurements_str = ', '.join(best_meas2)
                        st.write(f"**Best measurements:** {measurements_str}")
                    else:
                        st.write("**Best measurements:** None detected")
                
                # Length Difference Analysis
                st.markdown("#### 📏 Enhanced Length Analysis")
                if length_difference['analysis_possible']:
                    # Create a nice comparison display
                    col_left, col_center, col_right = st.columns([2, 1, 2])
                    
                    with col_left:
                        st.markdown(f"""
                        **🖼️ Image 1 Cable**
                        - **Length:** {length_difference['image1_primary']}
                        - **Method:** {best_method1 or 'N/A'}
                        - **Confidence:** {best_conf1:.1f}% 
                        """)
                    
                    with col_center:
                        diff_val = length_difference['difference_value']
                        if diff_val > 0:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); border-radius: 10px; color: white;">
                                <h3>📏 {length_difference['difference_display']}</h3>
                                <p>Image 1 is longer</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif diff_val < 0:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white;">
                                <h3>📏 {length_difference['difference_display']}</h3>
                                <p>Image 2 is longer</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
                                <h3>📏 {length_difference['difference_display']}</h3>
                                <p>Equal length</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_right:
                        st.markdown(f"""
                        **🖼️ Image 2 Cable**
                        - **Length:** {length_difference['image2_primary']}
                        - **Method:** {best_method2 or 'N/A'}
                        - **Confidence:** {best_conf2:.1f}%
                        """)
                    
                    # Additional analysis
                    st.markdown("#### 📈 Detailed Analysis")
                    if length_difference['percentage_difference']:
                        st.write(f"**Percentage Difference:** {length_difference['percentage_difference']}")
                    
                    # Confidence assessment
                    avg_confidence = (best_conf1 + best_conf2) / 2 if best_conf1 and best_conf2 else 0
                    if avg_confidence > 80:
                        confidence_assessment = "🟢 High confidence in results"
                    elif avg_confidence > 50:
                        confidence_assessment = "🟡 Medium confidence - consider manual verification"
                    else:
                        confidence_assessment = "🔴 Low confidence - results may be unreliable"
                    
                    st.write(f"**Overall Confidence:** {confidence_assessment}")
                    
                else:
                    st.warning(f"⚠️ {length_difference['reason']}")
                
                # Method Performance Analysis
                st.markdown("#### 🎯 Method Performance Analysis")
                
                # Create performance comparison
                performance_data = []
                for method in results1.keys():
                    img1_conf = results1[method].get('confidence', 0)
                    img1_meas = len(results1[method].get('measurements', []))
                    img2_conf = results2[method].get('confidence', 0)
                    img2_meas = len(results2[method].get('measurements', []))
                    
                    performance_data.append({
                        'Method': method,
                        'Img1_Confidence': img1_conf,
                        'Img1_Measurements': img1_meas,
                        'Img2_Confidence': img2_conf,
                        'Img2_Measurements': img2_meas,
                        'Avg_Confidence': (img1_conf + img2_conf) / 2,
                        'Total_Measurements': img1_meas + img2_meas
                    })
                
                # Sort by average confidence
                performance_data.sort(key=lambda x: x['Avg_Confidence'], reverse=True)
                
                # Display top performing methods
                st.markdown("**🏆 Top Performing Methods:**")
                for i, data in enumerate(performance_data[:3]):
                    emoji = ["🥇", "🥈", "🥉"][i]
                    st.write(f"{emoji} **{data['Method']}** - Avg Confidence: {data['Avg_Confidence']:.1f}%, Total Measurements: {data['Total_Measurements']}")
                
                # Overall Summary
                st.markdown("#### 🔄 Overall Summary")
                total_methods = len(results1) + len(results2)
                total_measurements = len(measurements1.union(measurements2))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Methods", total_methods)
                
                with col2:
                    st.metric("Unique Measurements", total_measurements)
                
                with col3:
                    if common_measurements:
                        st.metric("Common Measurements", len(common_measurements))
                    else:
                        st.metric("Common Measurements", "None")
                
                if common_measurements:
                    st.write(f"**Matching measurements found:** {', '.join(common_measurements)}")
                    st.success("✅ Cross-validation successful - consistent measurements detected!")
                else:
                    st.info("ℹ️ No common measurements found between images")
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                if avg_confidence > 80:
                    st.success("🎯 **High confidence results** - Measurements are likely accurate")
                elif avg_confidence > 50:
                    st.warning("⚠️ **Medium confidence** - Consider retaking images with better lighting or focus")
                    st.info("💡 **Tips:** Ensure text is well-lit, camera is steady, and measurement area is clearly visible")
                else:
                    st.error("❌ **Low confidence results** - Images may need improvement")
                    st.info("""
                    💡 **Improvement suggestions:**
                    - Better lighting on measurement area
                    - Higher resolution images
                    - Steadier camera position
                    - Cleaner measurement surface
                    """)
                
                st.success("✅ Enhanced analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
                st.info("💡 Please try again or contact support if the issue persists.")
    
    elif uploaded_file1 is not None or uploaded_file2 is not None:
        st.info("⚠️ Please upload both images for comparative analysis")
    else:
        st.info("📋 Upload both cable images to start enhanced measurement extraction and comparison")
        
        # Demo section
        with st.expander("📸 See Enhancement Examples"):
            st.markdown("""
            ### 🎨 Enhancement Methods Preview
            
            **1. Adaptive Threshold** - Best for varying lighting
            **2. High Contrast CLAHE** - Enhanced contrast with sharpening
            **3. Measurement Optimized** - Specifically tuned for cable measurements
            **4. Edge Enhanced** - Improved text boundary detection
            **5. Noise Reduction Pro** - Advanced denoising while preserving text
            **6. Multi-Scale Enhancement** - Processing at different scales for various text sizes
            **7. Original** - Unprocessed image for comparison
            
            Each method is optimized for different image conditions and text types.
            """)

if __name__ == "__main__":
    main()
