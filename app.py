import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import re
from decimal import Decimal, InvalidOperation

# Set page config
st.set_page_config(
    page_title="📏 Wire Length Distance Calculator",
    page_icon="📏",
    layout="wide"
)

def advanced_number_enhancement(img, method='precision_numbers'):
    """Enhanced preprocessing specifically for precise number recognition"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    if method == 'precision_numbers':
        # Multi-stage enhancement for precise number recognition
        # Stage 1: Noise reduction with edge preservation
        denoised = cv2.bilateralFilter(gray, 15, 80, 80)
        
        # Stage 2: Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        
        # Stage 3: Sharpening for crisp edges
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Stage 4: Morphological operations for digit separation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
        
        # Stage 5: Final thresholding
        _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Stage 6: Dilation for better character connection
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        result = cv2.dilate(binary, kernel_dilate, iterations=1)
        
    elif method == 'technical_drawing':
        # Optimized for technical drawings and blueprints
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        
        # Contrast stretching
        p2, p98 = np.percentile(denoised, (2, 98))
        stretched = np.clip((denoised - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        enhanced = clahe.apply(stretched)
        
        # Edge-preserving smoothing
        smoothed = cv2.edgePreservingFilter(enhanced, flags=2, sigma_s=50, sigma_r=0.4)
        
        # Adaptive thresholding
        result = cv2.adaptiveThreshold(smoothed.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
    elif method == 'handwritten_digits':
        # Optimized for handwritten measurements
        # Gentle noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        
        # Gamma correction for better visibility
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        
        # Slight blur to connect broken strokes
        blurred = cv2.GaussianBlur(gamma_corrected, (2, 2), 0)
        
        # Adaptive threshold with larger block size for handwriting
        result = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
    else:  # default fallback
        # Advanced multi-stage processing combining best techniques
        # Stage 1: Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Stage 2: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Stage 3: Unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        unsharp = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
        
        # Stage 4: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(unsharp, cv2.MORPH_OPEN, kernel)
        
        # Stage 5: Final normalization
        result = cv2.normalize(opened, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)

def get_precision_ocr_config(language='eng'):
    """Precision OCR configuration optimized for meter readings"""
    
    # Character set for meters: digits, decimal point, and 'm'
    meter_chars = '0123456789.m'
    
    # Multiple configurations for different scenarios
    configs = [
        f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={meter_chars} -c classify_bln_numeric_mode=1',
        f'--oem 3 --psm 7 -l {language} -c tessedit_char_whitelist={meter_chars}',
        f'--oem 3 --psm 6 -l {language} -c tessedit_char_whitelist={meter_chars}',
        f'--oem 3 --psm 13 -l {language} -c tessedit_char_whitelist={meter_chars}',
    ]
    
    return configs

def extract_length_measurements(text):
    """Extract and validate length measurements from text with focus on meters"""
    if not text:
        return []
    
    measurements = []
    
    # Comprehensive regex patterns for different measurement formats
    patterns = [
        r'(\d+(?:[.,]\d+)?)\s*m\b',                      # 1484m, 12.5m
        r'(\d+(?:[.,]\d+)?)\s*(?:meter|metre)s?\b',      # 1484 meter
        r'\b(\d+(?:[.,]\d+)?)m\b',                       # Standalone format like 1484m
        r'(\d+(?:[.,]\d+)?)(?=\s*(?:m\b|meter|metre))', # Number before meter
        r'(?:^|\s)(\d+(?:[.,]\d+)?)\s*m(?:\s|$)',       # Number with m at word boundaries
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
        for match in matches:
            try:
                # Extract numeric value
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Only accept reasonable meter values (0.1m to 10000m)
                if 0.1 <= value <= 10000:
                    measurements.append({
                        'original': match.group(0),
                        'value': value,
                        'unit': 'm',
                        'meters': value,
                        'confidence': calculate_measurement_confidence(match.group(0), text, value)
                    })
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates and sort by confidence
    unique_measurements = []
    seen_values = set()
    
    for measurement in measurements:
        rounded_meters = round(measurement['meters'], 2)
        if rounded_meters not in seen_values:
            seen_values.add(rounded_meters)
            unique_measurements.append(measurement)
    
    return sorted(unique_measurements, key=lambda x: x['confidence'], reverse=True)

def calculate_measurement_confidence(match_text, full_text, value):
    """Calculate confidence score for a measurement"""
    confidence = 0.6  # Base confidence
    
    # Higher confidence for proper formatting
    if re.match(r'^\d+(?:[.,]\d+)?m$', match_text.strip()):
        confidence += 0.2
    
    # Higher confidence for reasonable measurement values
    if 100 <= value <= 5000:   # Most common wire length range
        confidence += 0.2
    elif 10 <= value <= 10000:  # Reasonable range
        confidence += 0.1
    
    # Context clues
    context_words = ['wire', 'cable', 'length', 'distance', 'measure', 'total', 'span', 'fiber']
    for word in context_words:
        if word in full_text.lower():
            confidence += 0.1
            break
    
    return min(confidence, 1.0)

def perform_precision_ocr_multiple(img, language='eng'):
    """Perform OCR with multiple configurations and return best results"""
    configs = get_precision_ocr_config(language)
    all_results = []
    
    for config in configs:
        try:
            # Get text with this configuration
            text = pytesseract.image_to_string(img, config=config).strip()
            if text:
                measurements = extract_length_measurements(text)
                all_results.extend(measurements)
        except Exception as e:
            continue
    
    # Consolidate and return best measurements
    return consolidate_measurements(all_results)

def consolidate_measurements(measurements):
    """Consolidate similar measurements and rank by confidence"""
    if not measurements:
        return []
    
    # Group similar measurements (within 1% tolerance)
    groups = []
    for measurement in measurements:
        added_to_group = False
        for group in groups:
            if abs(measurement['meters'] - group[0]['meters']) / max(measurement['meters'], group[0]['meters']) < 0.01:
                group.append(measurement)
                added_to_group = True
                break
        if not added_to_group:
            groups.append([measurement])
    
    # Calculate group confidence and select best measurement from each group
    consolidated = []
    for group in groups:
        # Sort by confidence
        group.sort(key=lambda x: x['confidence'], reverse=True)
        best = group[0].copy()
        
        # Boost confidence if multiple methods agree
        if len(group) > 1:
            best['confidence'] = min(1.0, best['confidence'] + 0.1 * (len(group) - 1))
            best['agreement_count'] = len(group)
        else:
            best['agreement_count'] = 1
            
        consolidated.append(best)
    
    return sorted(consolidated, key=lambda x: (x['confidence'], x['agreement_count']), reverse=True)

def process_image_for_measurements(image, enhancement_method='precision_numbers', language='eng'):
    """Process a single image to extract meter measurements"""
    try:
        # Enhance the image
        with st.spinner("🔧 Enhancing image for OCR..."):
            enhanced_image = advanced_number_enhancement(image, enhancement_method)
        
        # Perform OCR with multiple methods
        with st.spinner("🔍 Extracting measurements..."):
            measurements = perform_precision_ocr_multiple(enhanced_image, language)
        
        return enhanced_image, measurements
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, []

def calculate_distance(value1, value2):
    """Calculate the absolute difference between two measurements"""
    if value1 is not None and value2 is not None:
        return abs(value1 - value2)
    return None

# Initialize session state
if 'image1_processed' not in st.session_state:
    st.session_state.image1_processed = False
if 'image2_processed' not in st.session_state:
    st.session_state.image2_processed = False
if 'measurements1' not in st.session_state:
    st.session_state.measurements1 = []
if 'measurements2' not in st.session_state:
    st.session_state.measurements2 = []
if 'enhanced1' not in st.session_state:
    st.session_state.enhanced1 = None
if 'enhanced2' not in st.session_state:
    st.session_state.enhanced2 = None

# Main UI
st.title("📏 Wire Length Distance Calculator")
st.markdown("**Automatic detection of meter measurements in two images and distance calculation**")

# Enhanced sidebar
st.sidebar.header("⚙️ Settings")

# Enhancement method selection
enhancement_method = st.sidebar.selectbox(
    "Enhancement Method",
    [
        ('precision_numbers', '🎯 Precision Numbers (Recommended)'),
        ('technical_drawing', '📐 Technical Drawing & Blueprints'),
        ('handwritten_digits', '✍️ Handwritten Measurements'),
        ('advanced_multi_stage', '🚀 Advanced Multi-Stage'),
    ],
    format_func=lambda x: x[1],
    help="Select the enhancement method that best matches your image type"
)[0]

# Language selection
language = st.sidebar.selectbox(
    "OCR Language",
    [
        ('eng', '🇺🇸 English'),
        ('eng+deu', '🇺🇸🇩🇪 English + German'),
        ('eng+fra', '🇺🇸🇫🇷 English + French'),
    ],
    format_func=lambda x: x[1]
)[0]

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold", 
    0.3, 1.0, 0.6, 0.1, 
    help="Minimum confidence for accepting measurements"
)

# File upload section
st.header("📁 Upload Images")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image 1")
    uploaded_file1 = st.file_uploader(
        "Upload first image with meter measurement",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        key="file1"
    )

with col2:
    st.subheader("📷 Image 2")
    uploaded_file2 = st.file_uploader(
        "Upload second image with meter measurement",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        key="file2"
    )

# Process images when uploaded
if uploaded_file1 is not None:
    image1 = Image.open(uploaded_file1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Original Image 1", use_container_width=True)
    
    if st.button("🔍 Process Image 1", key="process1"):
        enhanced1, measurements1 = process_image_for_measurements(image1, enhancement_method, language)
        if enhanced1:
            st.session_state.enhanced1 = enhanced1
            st.session_state.measurements1 = measurements1
            st.session_state.image1_processed = True
    
    # Show processed results for image 1
    if st.session_state.image1_processed and st.session_state.enhanced1:
        with col2:
            st.image(st.session_state.enhanced1, caption="Enhanced Image 1", use_container_width=True)
        
        if st.session_state.measurements1:
            filtered_measurements1 = [m for m in st.session_state.measurements1 if m['confidence'] >= confidence_threshold]
            if filtered_measurements1:
                best1 = filtered_measurements1[0]
                st.success(f"✅ **Image 1 Detected: {best1['meters']:.3f}m** (Confidence: {best1['confidence']:.1%})")
                
                if len(filtered_measurements1) > 1:
                    with st.expander(f"📊 All measurements from Image 1 ({len(filtered_measurements1)} found)"):
                        for i, m in enumerate(filtered_measurements1, 1):
                            st.write(f"{i}. {m['meters']:.3f}m (Confidence: {m['confidence']:.1%}, Original: '{m['original']}')")
            else:
                st.warning("⚠️ No measurements detected in Image 1 with sufficient confidence")
        else:
            st.error("❌ No measurements detected in Image 1")

if uploaded_file2 is not None:
    image2 = Image.open(uploaded_file2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image2, caption="Original Image 2", use_container_width=True)
    
    if st.button("🔍 Process Image 2", key="process2"):
        enhanced2, measurements2 = process_image_for_measurements(image2, enhancement_method, language)
        if enhanced2:
            st.session_state.enhanced2 = enhanced2
            st.session_state.measurements2 = measurements2
            st.session_state.image2_processed = True
    
    # Show processed results for image 2
    if st.session_state.image2_processed and st.session_state.enhanced2:
        with col2:
            st.image(st.session_state.enhanced2, caption="Enhanced Image 2", use_container_width=True)
        
        if st.session_state.measurements2:
            filtered_measurements2 = [m for m in st.session_state.measurements2 if m['confidence'] >= confidence_threshold]
            if filtered_measurements2:
                best2 = filtered_measurements2[0]
                st.success(f"✅ **Image 2 Detected: {best2['meters']:.3f}m** (Confidence: {best2['confidence']:.1%})")
                
                if len(filtered_measurements2) > 1:
                    with st.expander(f"📊 All measurements from Image 2 ({len(filtered_measurements2)} found)"):
                        for i, m in enumerate(filtered_measurements2, 1):
                            st.write(f"{i}. {m['meters']:.3f}m (Confidence: {m['confidence']:.1%}, Original: '{m['original']}')")
            else:
                st.warning("⚠️ No measurements detected in Image 2 with sufficient confidence")
        else:
            st.error("❌ No measurements detected in Image 2")

# Calculate distance when both images are processed
if (st.session_state.image1_processed and st.session_state.image2_processed and 
    st.session_state.measurements1 and st.session_state.measurements2):
    
    st.header("📊 Distance Calculation Results")
    
    # Get best measurements from both images
    filtered_measurements1 = [m for m in st.session_state.measurements1 if m['confidence'] >= confidence_threshold]
    filtered_measurements2 = [m for m in st.session_state.measurements2 if m['confidence'] >= confidence_threshold]
    
    if filtered_measurements1 and filtered_measurements2:
        best1 = filtered_measurements1[0]
        best2 = filtered_measurements2[0]
        
        # Calculate distance
        distance = calculate_distance(best1['meters'], best2['meters'])
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📏 Image 1 Length",
                f"{best1['meters']:.3f}m",
                delta=f"Confidence: {best1['confidence']:.1%}"
            )
        
        with col2:
            st.metric(
                "📏 Image 2 Length", 
                f"{best2['meters']:.3f}m",
                delta=f"Confidence: {best2['confidence']:.1%}"
            )
        
        with col3:
            st.metric(
                "📐 Distance (Difference)",
                f"{distance:.3f}m",
                delta=f"Δ = |{best1['meters']:.3f} - {best2['meters']:.3f}|"
            )
        
        # Additional analysis
        st.subheader("📈 Analysis Summary")
        larger_value = max(best1['meters'], best2['meters'])
        smaller_value = min(best1['meters'], best2['meters'])
        percentage_diff = (distance / larger_value) * 100 if larger_value > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**🔺 Larger value:** {larger_value:.3f}m")
            st.write(f"**🔻 Smaller value:** {smaller_value:.3f}m")
            st.write(f"**📊 Percentage difference:** {percentage_diff:.2f}%")
            st.write(f"**🎯 Average confidence:** {(best1['confidence'] + best2['confidence'])/2:.1%}")
        
        with col2:
            if percentage_diff < 5:
                st.success("✅ **Very close measurements** (< 5% difference)")
                st.info("The two measurements are highly similar.")
            elif percentage_diff < 15:
                st.warning("⚠️ **Moderate difference** (5-15%)")
                st.info("There is a noticeable difference between measurements.")
            else:
                st.error("❌ **Significant difference** (> 15%)")
                st.info("The measurements show a substantial difference.")
        
        # Export results
        st.subheader("💾 Export Results")
        result_summary = f"""Wire Length Distance Calculation Results:

Image 1: {best1['meters']:.3f}m (Confidence: {best1['confidence']:.1%})
Image 2: {best2['meters']:.3f}m (Confidence: {best2['confidence']:.1%})

Distance (Difference): {distance:.3f}m
Percentage Difference: {percentage_diff:.2f}%

Analysis: {'Very close measurements' if percentage_diff < 5 else 'Moderate difference' if percentage_diff < 15 else 'Significant difference'}
"""
        
        st.text_area("📋 Results Summary", result_summary, height=200)
        
        if st.button("🎉 Generate New Analysis"):
            st.balloons()
            st.success("✅ Analysis complete! Results ready for use.")
    
    else:
        st.warning("⚠️ Cannot calculate distance - one or both images have no measurements with sufficient confidence.")

# Process both images button
if uploaded_file1 is not None and uploaded_file2 is not None:
    st.header("🚀 Quick Process Both Images")
    if st.button("⚡ Process Both Images Simultaneously", type="primary"):
        col1, col2 = st.columns(2)
        
        # Process image 1
        with col1:
            st.info("🔍 Processing Image 1...")
            image1 = Image.open(uploaded_file1)
            enhanced1, measurements1 = process_image_for_measurements(image1, enhancement_method, language)
            if enhanced1:
                st.session_state.enhanced1 = enhanced1
                st.session_state.measurements1 = measurements1
                st.session_state.image1_processed = True
                st.success("✅ Image 1 processed!")
        
        # Process image 2
        with col2:
            st.info("🔍 Processing Image 2...")
            image2 = Image.open(uploaded_file2)
            enhanced2, measurements2 = process_image_for_measurements(image2, enhancement_method, language)
            if enhanced2:
                st.session_state.enhanced2 = enhanced2
                st.session_state.measurements2 = measurements2
                st.session_state.image2_processed = True
                st.success("✅ Image 2 processed!")
        
        st.rerun()

# Help section
with st.expander("🎓 How to Use", expanded=False):
    st.markdown("""
    ### 📋 Step-by-Step Guide:
    
    1. **📁 Upload Images**: Upload two images containing meter measurements (e.g., "1484m", "12.5m")
    
    2. **⚙️ Configure Settings**:
       - Choose enhancement method based on your image type
       - Set confidence threshold (higher = more strict)
       - Select appropriate language for OCR
    
    3. **🔍 Process Images**: 
       - Click "Process Image 1" and "Process Image 2" individually
       - OR use "Process Both Images Simultaneously" for faster processing
    
    4. **📊 View Results**:
       - See detected measurements with confidence scores
       - Automatic distance calculation between the two measurements
       - Analysis of difference percentage and interpretation
    
    ### 📏 Supported Formats:
    - Direct meter readings: `1484m`, `12.5m`, `0.75m`
    - With spaces: `1484 m`, `12.5 meter`, `150 metres`
    
    ### 💡 Tips for Best Results:
    - Use clear, high-quality images
    - Ensure good contrast between text and background  
    - Choose the right enhancement method for your image type
    - Images with handwritten numbers work best with "Handwritten Digits" method
    """)

# Footer
st.markdown("---")
st.markdown("🎯 **Wire Length Distance Calculator** - Automatic meter measurement detection and distance calculation! 📏✨")
