import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import io
import base64
import re
import warnings
import os
import time
import gc
from functools import wraps
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 Advanced Cable Measurement Reader",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
    
    .measurement-found {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        font-size: 1.4em;
        box-shadow: 0 5px 15px rgba(17,153,142,0.3);
    }
    
    .measurement-failed {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .confidence-ultra {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    .method-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def test_tesseract():
    """Test if Tesseract is properly installed and accessible"""
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} found"
    except Exception as e:
        return False, f"Tesseract not found: {str(e)}"

def preprocess_image_for_cable_reading(image):
    """Advanced preprocessing specifically for cable tag reading"""
    try:
        # Convert PIL to OpenCV
        if isinstance(image, Image.Image):
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_img = image.copy()
        
        # Resize if too large but maintain quality
        height, width = cv_img.shape[:2]
        if height > 2000 or width > 2000:
            scale = min(2000/height, 2000/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif height < 200 or width < 200:
            # Upscale small images
            scale = max(400/height, 400/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cv_img
    except Exception as e:
        st.warning(f"Preprocessing failed: {str(e)}")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def create_advanced_enhancement(img, method_name):
    """Advanced image enhancement methods specifically for cable measurements"""
    try:
        cv_img = preprocess_image_for_cable_reading(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        if method_name == 'Cable Tag Specialized':
            # Specifically designed for white tags with black text
            # Step 1: Enhance contrast significantly
            enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=50)
            
            # Step 2: Apply strong bilateral filtering to smooth background while preserving text edges
            enhanced = cv2.bilateralFilter(enhanced, 15, 100, 100)
            
            # Step 3: Apply adaptive threshold to handle uneven lighting
            enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
            
            # Step 4: Morphological operations to clean up noise
            kernel = np.ones((2,2), np.uint8)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
        elif method_name == 'Ultra Contrast':
            # Extreme contrast enhancement
            enhanced = cv2.convertScaleAbs(gray, alpha=3.0, beta=60)
            
            # Apply CLAHE with strong parameters
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
            enhanced = clahe.apply(enhanced)
            
            # Strong unsharp masking
            gaussian = cv2.GaussianBlur(enhanced, (0,0), 1.5)
            enhanced = cv2.addWeighted(enhanced, 2.5, gaussian, -1.5, 0)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
        elif method_name == 'Text Isolation':
            # Focus on isolating text regions
            enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            
            # Apply strong Gaussian blur then subtract to enhance edges
            blurred = cv2.GaussianBlur(enhanced, (15, 15), 0)
            enhanced = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
            
            # Apply threshold to create binary image
            _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method_name == 'Handwriting Enhanced':
            # Optimized for handwritten numbers
            enhanced = cv2.convertScaleAbs(gray, alpha=2.2, beta=40)
            
            # Use different kernel for morphological operations
            kernel = np.ones((3,3), np.uint8)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
            
            # Apply threshold
            _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method_name == 'Multi-Resolution':
            # Process at multiple resolutions and combine
            resolutions = [0.8, 1.0, 1.3]
            enhanced_versions = []
            
            for res in resolutions:
                if res != 1.0:
                    h, w = gray.shape
                    temp_img = cv2.resize(gray, (int(w*res), int(h*res)))
                else:
                    temp_img = gray.copy()
                
                # Apply enhancement
                temp_enhanced = cv2.convertScaleAbs(temp_img, alpha=2.3, beta=45)
                clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
                temp_enhanced = clahe.apply(temp_enhanced)
                
                # Resize back if needed
                if res != 1.0:
                    temp_enhanced = cv2.resize(temp_enhanced, (gray.shape[1], gray.shape[0]))
                
                enhanced_versions.append(temp_enhanced)
            
            # Combine by taking maximum intensity
            enhanced = np.maximum.reduce(enhanced_versions)
            
        elif method_name == 'Edge Detection Plus':
            # Advanced edge detection combined with enhancement
            enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=25)
            
            # Apply different edge detection methods
            edges_sobel = cv2.Sobel(enhanced, cv2.CV_64F, 1, 1, ksize=3)
            edges_sobel = np.uint8(np.absolute(edges_sobel))
            
            edges_laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
            edges_laplacian = np.uint8(np.absolute(edges_laplacian))
            
            # Combine edge information
            edges_combined = cv2.addWeighted(edges_sobel, 0.5, edges_laplacian, 0.5, 0)
            
            # Add back to original
            enhanced = cv2.addWeighted(enhanced, 0.7, edges_combined, 0.3, 0)
            
        elif method_name == 'Noise Elimination':
            # Advanced noise reduction while preserving text
            enhanced = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
            
            # Enhance after denoising
            enhanced = cv2.convertScaleAbs(enhanced, alpha=2.4, beta=35)
            
            # Apply sharpening
            kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
            
        else:  # Original
            enhanced = gray
        
        return Image.fromarray(enhanced)
        
    except Exception as e:
        st.warning(f"Enhancement failed for {method_name}: {str(e)}")
        return img

def extract_cable_measurements(text, method_confidence=0):
    """Highly specialized measurement extraction for cable readings"""
    measurements = []
    confidence_scores = {}
    
    # Clean text first
    text = re.sub(r'[^\d\w\s\.]', ' ', text)  # Remove special chars except digits, letters, spaces, dots
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize spaces
    
    # Ultra-specific patterns for cable measurements
    patterns = [
        # Direct measurement patterns (highest priority)
        (r'\b(\d{2,4})\s*m\b', 'cable_direct', 95),  # "645m", "1234m"
        (r'\b(\d{1,4})\s*\.?\s*(\d+)?\s*m\b', 'cable_decimal', 90),  # "12.5m", "645.0m"
        (r'(\d{2,4})\s*meter[s]?', 'meter_word', 85),  # "645 meters"
        (r'(\d{2,4})\s*mtr[s]?', 'meter_abbrev', 85),  # "645 mtr"
        
        # Handle OCR errors and variations
        (r'\b(\d{2,4})\s*[mn]\b', 'ocr_error_mn', 70),  # 'm' read as 'n' or vice versa
        (r'\b(\d{2,4})\s*[mM]\b', 'case_variant', 80),  # Case variations
        
        # Realistic cable length patterns
        (r'\b([1-9]\d{1,3})\s*m', 'realistic_range', 85),  # 10-9999m range
        
        # Handle spaced digits (OCR artifacts)
        (r'(\d)\s+(\d)\s+(\d)\s*m', 'spaced_digits', 60),  # "6 4 5 m" -> "645m"
    ]
    
    for pattern, pattern_type, base_confidence in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if pattern_type == 'spaced_digits':
                    # Combine spaced digits
                    value = match.group(1) + match.group(2) + match.group(3)
                elif pattern_type == 'cable_decimal':
                    # Handle decimal format
                    if match.group(2):
                        value = match.group(1) + '.' + match.group(2)
                    else:
                        value = match.group(1)
                else:
                    value = match.group(1)
                
                # Convert to float and validate realistic cable lengths
                float_val = float(value)
                if 5 <= float_val <= 10000:  # Realistic cable length range
                    measurement = f"{int(float_val) if float_val.is_integer() else float_val}m"
                    measurements.append(measurement)
                    
                    # Calculate confidence score
                    final_confidence = min(100, base_confidence + method_confidence)
                    confidence_scores[measurement] = final_confidence
                    
            except (ValueError, AttributeError):
                continue
    
    # Unit conversions with validation
    # mm to meters
    mm_pattern = r'(\d+)\s*mm'
    for match in re.finditer(mm_pattern, text, re.IGNORECASE):
        try:
            mm_val = int(match.group(1))
            if 5000 <= mm_val <= 10000000:  # 5m to 10km in mm
                m_val = mm_val / 1000
                measurement = f"{int(m_val) if m_val.is_integer() else m_val:.1f}m"
                measurements.append(measurement)
                confidence_scores[measurement] = 75
        except ValueError:
            continue
    
    # cm to meters
    cm_pattern = r'(\d+)\s*cm'
    for match in re.finditer(cm_pattern, text, re.IGNORECASE):
        try:
            cm_val = int(match.group(1))
            if 500 <= cm_val <= 1000000:  # 5m to 10km in cm
                m_val = cm_val / 100
                measurement = f"{int(m_val) if m_val.is_integer() else m_val:.1f}m"
                measurements.append(measurement)
                confidence_scores[measurement] = 75
        except ValueError:
            continue
    
    # Remove duplicates and prioritize by confidence
    unique_measurements = {}
    for measurement in measurements:
        key = float(measurement.replace('m', ''))
        if key not in unique_measurements or confidence_scores[measurement] > confidence_scores.get(unique_measurements[key], 0):
            unique_measurements[key] = measurement
    
    # Sort by value and return
    final_measurements = sorted(unique_measurements.values(), key=lambda x: float(x.replace('m', '')))
    final_confidence_map = {m: confidence_scores.get(m, 50) for m in final_measurements}
    
    return final_measurements, final_confidence_map

def perform_advanced_ocr(image, method_name):
    """Advanced OCR with multiple configurations for cable reading"""
    try:
        img_array = np.array(image)
        
        # Specialized OCR configurations for cable measurements
        ocr_configs = [
            # Configuration 1: Focus on single line text (best for cable tags)
            '--psm 7 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
            
            # Configuration 2: Single uniform block
            '--psm 6 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
            
            # Configuration 3: Single word (for isolated measurements)
            '--psm 8 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
            
            # Configuration 4: Raw line with minimal processing
            '--psm 13 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
            
            # Configuration 5: Sparse text
            '--psm 11 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
            
            # Configuration 6: Default with whitelist
            '--psm 3 -c tessedit_char_whitelist=0123456789.mMeEtTrR ',
        ]
        
        all_texts = []
        best_confidence = 0
        
        for config in ocr_configs:
            try:
                # Get OCR result with confidence data
                ocr_data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(img_array, config=config)
                
                if text.strip():
                    all_texts.append(text.strip())
                    
                    # Calculate confidence
                    confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        best_confidence = max(best_confidence, avg_conf)
                        
            except Exception:
                continue
        
        # Combine all OCR results
        combined_text = ' '.join(all_texts)
        
        return combined_text, best_confidence
        
    except Exception as e:
        return "", 0

def process_image_with_advanced_methods(image, image_name):
    """Process image with advanced enhancement methods"""
    methods = [
        'Cable Tag Specialized',
        'Ultra Contrast', 
        'Text Isolation',
        'Handwriting Enhanced',
        'Multi-Resolution',
        'Edge Detection Plus',
        'Noise Elimination',
        'Original'
    ]
    
    results = {}
    all_measurements = set()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, method in enumerate(methods):
            progress = (idx + 1) / len(methods)
            progress_bar.progress(progress)
            status_text.text(f"🔍 {image_name} - {method} ({idx+1}/{len(methods)})")
            
            # Apply enhancement
            enhanced_img = create_advanced_enhancement(image, method)
            
            if enhanced_img is None:
                continue
            
            # Perform OCR
            ocr_text, ocr_confidence = perform_advanced_ocr(enhanced_img, method)
            
            # Extract measurements
            measurements, conf_map = extract_cable_measurements(ocr_text, int(ocr_confidence/10))
            all_measurements.update(measurements)
            
            # Store results
            results[method] = {
                'image': enhanced_img,
                'measurements': measurements,
                'raw_text': ocr_text,
                'ocr_confidence': ocr_confidence,
                'measurement_confidence': conf_map
            }
            
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        return results, all_measurements
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Processing failed: {str(e)}")
        return {}, set()

def display_advanced_results(results, image_name):
    """Display results with advanced confidence indicators"""
    st.markdown(f"### 🖼️ {image_name} - Advanced Analysis")
    
    # Find best result
    best_method = None
    best_score = 0
    best_measurement = None
    
    for method, result in results.items():
        measurements = result.get('measurements', [])
        if measurements:
            for measurement in measurements:
                confidence = result['measurement_confidence'].get(measurement, 0)
                if confidence > best_score:
                    best_score = confidence
                    best_method = method
                    best_measurement = measurement
    
    # Display best result prominently
    if best_measurement:
        confidence_class = "confidence-ultra" if best_score >= 90 else "confidence-high" if best_score >= 75 else "confidence-medium" if best_score >= 50 else "confidence-low"
        
        st.markdown(f"""
        <div class="{confidence_class}">
            🎯 BEST RESULT: {best_measurement} 
            <br>Method: {best_method} | Confidence: {best_score}%
        </div>
        """, unsafe_allow_html=True)
    
    # Display all method results in grid
    methods = list(results.keys())
    for row in range(0, len(methods), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            method_idx = row + col_idx
            if method_idx < len(methods):
                method = methods[method_idx]
                result = results[method]
                
                with cols[col_idx]:
                    with st.container():
                        st.markdown(f"**🎨 {method}**")
                        
                        if result['image'] is not None:
                            st.image(result['image'], use_column_width=True)
                        
                        # Display measurements
                        measurements = result.get('measurements', [])
                        if measurements:
                            for measurement in measurements:
                                conf = result['measurement_confidence'].get(measurement, 0)
                                if conf >= 75:
                                    st.markdown(f"""
                                    <div class="measurement-found">
                                        📏 {measurement} ({conf}% confidence)
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info(f"📏 {measurement} ({conf}% confidence)")
                        else:
                            st.markdown("""
                            <div class="measurement-failed">
                                ❌ No measurements found
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show OCR confidence and raw text in expander
                        ocr_conf = result.get('ocr_confidence', 0)
                        raw_text = result.get('raw_text', '')
                        
                        with st.expander(f"Details (OCR: {ocr_conf:.0f}%)"):
                            if raw_text:
                                st.text(f"Raw OCR: {raw_text}")
                            else:
                                st.text("No text detected")

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Advanced Cable Measurement Reader</h1>
        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin: 10px 0;">
            🚀 Specialized OCR for Cable Tags - Optimized for "645m" Type Readings
        </div>
        <p>Ultra-precise measurement extraction from cable identification tags</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test Tesseract
    tesseract_ok, tesseract_msg = test_tesseract()
    if tesseract_ok:
        st.success(f"✅ {tesseract_msg}")
    else:
        st.error(f"❌ {tesseract_msg}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Advanced Features")
        st.markdown("""
        - **Cable Tag Specialized**: Optimized for white tags with black text
        - **Ultra Contrast**: Extreme contrast enhancement for faded text
        - **Text Isolation**: Advanced text region detection
        - **Handwriting Enhanced**: Improved handwritten number recognition
        - **Multi-Resolution**: Process at multiple scales simultaneously
        - **Edge Detection Plus**: Advanced edge-based text enhancement
        - **Noise Elimination**: Superior noise reduction with text preservation
        - **Original**: Baseline comparison
        """)
        
        st.markdown("### 📏 Target Formats")
        st.markdown("""
        - ✅ **"645m"** - Primary cable tag format
        - ✅ **"12.5m"** - Decimal measurements
        - ✅ **"1234 meter"** - Word format
        - ✅ **"6450mm"** - Auto-converted to meters
        - ✅ **Handwritten numbers** on tags
        """)
    
    # Upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Upload Cable Image 1</h3>
            <p>Focus on the measurement tag area</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file1 = st.file_uploader("Choose first image", type=['png', 'jpg', 'jpeg'], key="img1")
    
    with col2:
        st.markdown("""
        <div class="upload-section" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📁 Upload Cable Image 2</h3>
            <p>For comparison analysis</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file2 = st.file_uploader("Choose second image", type=['png', 'jpg', 'jpeg'], key="img2")
    
    # Process images
    if uploaded_file1 is not None and uploaded_file2 is not None:
        if st.button("🎯 Advanced Analysis", key="process"):
            image1 = Image.open(uploaded_file1)
            image2 = Image.open(uploaded_file2)
            
            st.markdown("---")
            st.info("🔍 Processing with advanced cable-specific methods...")
            
            # Process both images
            col1, col2 = st.columns(2)
            
            with col1:
                results1, measurements1 = process_image_with_advanced_methods(image1, "Image 1")
                display_advanced_results(results1, "Image 1")
            
            with col2:
                results2, measurements2 = process_image_with_advanced_methods(image2, "Image 2")
                display_advanced_results(results2, "Image 2")
            
            # Summary comparison
            if measurements1 or measurements2:
                st.markdown("---")
                st.markdown("### 📊 Comparison Summary")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if measurements1:
                        st.success(f"🖼️ **Image 1**: {', '.join(sorted(measurements1, key=lambda x: float(x.replace('m', ''))))}")
                    else:
                        st.warning("🖼️ **Image 1**: No measurements detected")
                
                with col_right:
                    if measurements2:
                        st.success(f"🖼️ **Image 2**: {', '.join(sorted(measurements2, key=lambda x: float(x.replace('m', ''))))}")
                    else:
                        st.warning("🖼️ **Image 2**: No measurements detected")
                
                # Length difference calculation
                if measurements1 and measurements2:
                    try:
                        val1 = max([float(m.replace('m', '')) for m in measurements1])
                        val2 = max([float(m.replace('m', '')) for m in measurements2])
                        diff = abs(val1 - val2)
                        
                        if diff == 0:
                            st.success("✅ **Perfect Match**: Both cables have identical length!")
                        else:
                            longer = "Image 1" if val1 > val2 else "Image 2"
                            st.info(f"📏 **Length Difference**: {diff}m ({longer} is longer)")
                    except:
                        st.info("📊 Comparison calculation unavailable")
            
            st.success("🎯 Advanced analysis completed!")
    
    elif uploaded_file1 or uploaded_file2:
        st.info("📤 Please upload both images for comparison analysis")
    else:
        st.info("📋 Upload cable images to start advanced measurement extraction")

if __name__ == "__main__":
    main()
