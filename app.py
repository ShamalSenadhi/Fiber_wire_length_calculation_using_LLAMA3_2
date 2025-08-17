import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

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
        denoised = cv2.bilateralFilter(gray, 15, 80, 80)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
        _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        result = cv2.dilate(binary, kernel_dilate, iterations=1)
        
    elif method == 'technical_drawing':
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        p2, p98 = np.percentile(denoised, (2, 98))
        stretched = np.clip((denoised - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        enhanced = clahe.apply(stretched)
        smoothed = cv2.edgePreservingFilter(enhanced, flags=2, sigma_s=50, sigma_r=0.4)
        result = cv2.adaptiveThreshold(smoothed.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
    elif method == 'meter_scale':
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        alpha = 2.0
        beta = -50
        enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
        
    elif method == 'ultra_sharp':
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        result = cv2.adaptiveThreshold(normalized.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        
    elif method == 'blueprint_mode':
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        lab[:,:,0] = l_channel
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
        result = cv2.add(enhanced_gray, tophat)
        
    elif method == 'handwritten_digits':
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        blurred = cv2.GaussianBlur(gamma_corrected, (2, 2), 0)
        result = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
    elif method == 'high_dpi_scan':
        h, w = gray.shape
        if w > 2000 or h > 2000:
            scale = min(2000/w, 2000/h)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        def single_scale_retinex(img, sigma):
            retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
            return retinex
        
        gray_float = gray.astype(np.float64) + 1.0
        retinex1 = single_scale_retinex(gray_float, 15)
        retinex2 = single_scale_retinex(gray_float, 80)
        retinex3 = single_scale_retinex(gray_float, 250)
        retinex = (retinex1 + retinex2 + retinex3) / 3.0
        retinex = np.clip((retinex - retinex.min()) * 255 / (retinex.max() - retinex.min()), 0, 255)
        result = retinex.astype(np.uint8)
        
    elif method == 'low_contrast_boost':
        equalized = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
        clahe_applied = clahe.apply(gray)
        combined = cv2.addWeighted(equalized, 0.6, clahe_applied, 0.4, 0)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        result = cv2.filter2D(combined, -1, kernel)
        
    elif method == 'wire_diagram_special':
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        masked = cv2.bitwise_and(gray, gray, mask=dilated_edges)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_masked = clahe.apply(masked)
        result = cv2.addWeighted(gray, 0.7, enhanced_masked, 0.3, 0)
        
    elif method == 'measurement_tape':
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        morph_h = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        morph_v = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        combined = cv2.add(morph_h, morph_v)
        result = cv2.subtract(gray, combined)
        result = cv2.convertScaleAbs(result, alpha=2.0, beta=0)
        
    else:  # advanced_multi_stage
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        unsharp = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(unsharp, cv2.MORPH_OPEN, kernel)
        result = cv2.normalize(opened, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)

def extract_number_from_image(image, image_name='image'):
    """Extract numerical values in meters from image using OCR"""
    try:
        # Use Tesseract OCR to extract text
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,mMkKcC '
        text = pytesseract.image_to_string(image, config=config)
        print(f"\nRaw OCR output for {image_name}:\n{text}")
        
        # Use regex to find numerical values with optional 'm' or 'meters'
        patterns = [
            r'(\d+(?:[.,]\d+)?)\s*m(?:eter)?s?\b',           # 1484m, 12.5m, 1,234m
            r'(\d+(?:[.,]\d+)?)\s*(?:km|kilometer)s?\b',     # 12.5km, 1 kilometer  
            r'(\d+(?:[.,]\d+)?)\s*(?:cm|centimeter)s?\b',    # 150cm, 15.5 centimeters
            r'(\d+(?:[.,]\d+)?)\s*(?:mm|millimeter)s?\b',    # 1500mm, 15.5 millimeters
            r'\b(\d+(?:[.,]\d+)?)m\b',                       # Standalone format like 1484m
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Convert to meters based on unit detected
                full_match = match.group(0).lower()
                if 'km' in full_match or 'kilometer' in full_match:
                    meters = value * 1000
                elif 'cm' in full_match or 'centimeter' in full_match:
                    meters = value / 100
                elif 'mm' in full_match or 'millimeter' in full_match:
                    meters = value / 1000
                else:
                    meters = value
                
                print(f"Extracted number from {image_name}: {meters} meters")
                return meters
        
        print(f"No number found in {image_name}")
        return None
        
    except Exception as e:
        print(f"Error extracting number from {image_name}: {e}")
        return None

# Initialize session state
if 'measurements' not in st.session_state:
    st.session_state.measurements = []

# Main UI
st.title("📏 Wire Length Distance Calculator")
st.markdown("**Extract wire lengths from images and calculate the distance between them**")

# Settings sidebar
st.sidebar.header("⚙️ Settings")
enhancement_method = st.sidebar.selectbox(
    "Enhancement Method",
    [
        ('precision_numbers', '🎯 Precision Numbers'),
        ('technical_drawing', '📐 Technical Drawing'),
        ('meter_scale', '📏 Meter Scale'),
        ('ultra_sharp', '⚡ Ultra Sharp'),
        ('blueprint_mode', '🔵 Blueprint Mode'),
        ('handwritten_digits', '✍️ Handwritten'),
        ('high_dpi_scan', '📋 High DPI Scan'),
        ('low_contrast_boost', '🔆 Contrast Boost'),
        ('wire_diagram_special', '🔌 Wire Diagram'),
        ('measurement_tape', '📐 Measurement Tape'),
        ('advanced_multi_stage', '🚀 Multi-Stage'),
    ],
    format_func=lambda x: x[1]
)[0]

# File upload section
st.header("📁 Upload Images")
uploaded_files = st.file_uploader(
    "Choose images with wire length measurements",
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    accept_multiple_files=True,
    help="Upload images containing wire length measurements in meters"
)

if uploaded_files:
    st.header("🔄 Processing Images")
    
    if st.button("🚀 Extract All Wire Lengths", type="primary"):
        st.session_state.measurements = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"📷 Processing Image {i+1}: {uploaded_file.name}")
            
            # Load original image
            original_image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Image**")
                st.image(original_image, use_container_width=True)
            
            with st.spinner(f"Enhancing image {i+1}..."):
                # Apply enhancement
                enhanced_image = advanced_number_enhancement(original_image, enhancement_method)
                
                with col2:
                    st.write("**Enhanced Image**")
                    st.image(enhanced_image, use_container_width=True)
            
            with st.spinner(f"Extracting numbers from image {i+1}..."):
                # Extract number
                extracted_length = extract_number_from_image(enhanced_image, uploaded_file.name)
                
                if extracted_length is not None:
                    st.session_state.measurements.append({
                        'name': uploaded_file.name,
                        'length': extracted_length
                    })
                    st.success(f"✅ Extracted: **{extracted_length} meters** from {uploaded_file.name}")
                else:
                    st.error(f"❌ Could not extract length from {uploaded_file.name}")
            
            st.markdown("---")
    
    # Results section
    if st.session_state.measurements:
        st.header("📊 Extracted Measurements")
        
        # Display all measurements
        for i, measurement in enumerate(st.session_state.measurements, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{measurement['name']}**")
            with col2:
                st.metric("Length", f"{measurement['length']} m")
        
        # Calculate distances if we have multiple measurements
        if len(st.session_state.measurements) >= 2:
            st.header("📏 Distance Calculations")
            
            # Show all pairwise distances
            for i in range(len(st.session_state.measurements)):
                for j in range(i + 1, len(st.session_state.measurements)):
                    measurement1 = st.session_state.measurements[i]
                    measurement2 = st.session_state.measurements[j]
                    
                    distance = abs(measurement1['length'] - measurement2['length'])
                    
                    st.info(f"📏 **Distance between {measurement1['name']} and {measurement2['name']}:** {distance} meters")
            
            # Summary statistics if we have many measurements
            if len(st.session_state.measurements) > 2:
                lengths = [m['length'] for m in st.session_state.measurements]
                max_length = max(lengths)
                min_length = min(lengths)
                max_distance = max_length - min_length
                avg_length = sum(lengths) / len(lengths)
                
                st.header("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Maximum Distance", f"{max_distance:.2f} m")
                with col2:
                    st.metric("Maximum Length", f"{max_length:.2f} m")
                with col3:
                    st.metric("Minimum Length", f"{min_length:.2f} m")
                with col4:
                    st.metric("Average Length", f"{avg_length:.2f} m")
        
        # Export results
        st.header("💾 Export Results")
        if st.button("📋 Copy All Results"):
            results_text = "Wire Length Measurements:\n"
            for measurement in st.session_state.measurements:
                results_text += f"- {measurement['name']}: {measurement['length']} m\n"
            
            if len(st.session_state.measurements) >= 2:
                results_text += "\nDistances:\n"
                for i in range(len(st.session_state.measurements)):
                    for j in range(i + 1, len(st.session_state.measurements)):
                        measurement1 = st.session_state.measurements[i]
                        measurement2 = st.session_state.measurements[j]
                        distance = abs(measurement1['length'] - measurement2['length'])
                        results_text += f"- {measurement1['name']} ↔ {measurement2['name']}: {distance} m\n"
            
            st.code(results_text)
            st.success("✅ Results ready for copying!")

# Help section
with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload Images**: Select multiple images containing wire length measurements
    2. **Choose Enhancement**: Pick the method that works best for your image type
    3. **Extract Lengths**: Click "Extract All Wire Lengths" to process all images
    4. **View Distances**: The app automatically calculates distances between all wire lengths
    
    ### 📏 Supported Formats:
    - `1484m`, `12.5m` (meters)
    - `1.5km`, `2km` (kilometers → converted to meters)
    - `150cm`, `75.5cm` (centimeters → converted to meters)
    - `1500mm`, `125mm` (millimeters → converted to meters)
    
    ### 🎯 Tips for Best Results:
    - Use high-quality, clear images
    - Ensure measurements are clearly visible
    - Try different enhancement methods if extraction fails
    - Works best with printed/digital text, but also supports handwritten numbers
    """)

# Footer
st.markdown("---")
st.markdown("🎯 **Wire Length Distance Calculator** - Extract measurements and calculate distances automatically! 📏✨")
