import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="🔍 Enhanced Tesseract Wire Calculator",
    page_icon="🔍",
    layout="wide"
)

class TesseractWireExtractor:
    """Enhanced Tesseract-based wire length extractor with multiple OCR strategies"""
    
    def __init__(self):
        self.ocr_configs = {
            'default': '--oem 3 --psm 6',
            'digits_only': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,',
            'measurements': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,mMkKcCeEtTrRsS ',
            'single_line': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,mMkKcCeEtTrRsS ',
            'sparse_text': '--oem 3 --psm 11',
            'single_word': '--oem 3 --psm 8',
            'vertical_text': '--oem 3 --psm 5',
            'uniform_block': '--oem 3 --psm 6',
            'handwritten': '--oem 3 --psm 13',
            'legacy_engine': '--oem 0 --psm 6 -c tessedit_char_whitelist=0123456789.,mMkKcCeEtTrRsS '
        }
    
    def preprocess_image(self, image, method='adaptive'):
        """Advanced image preprocessing optimized for Tesseract OCR"""
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        if method == 'adaptive':
            # Adaptive preprocessing based on image characteristics
            height, width = gray.shape
            
            # Check if image is high resolution
            if width > 1500 or height > 1500:
                # High-res preprocessing
                denoised = cv2.fastNlMeansDenoising(gray, h=10)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                result = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                # Standard resolution preprocessing
                denoised = cv2.bilateralFilter(gray, 9, 75, 75)
                enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
                result = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        elif method == 'high_contrast':
            # High contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            result = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        elif method == 'edge_preserving':
            # Edge-preserving smoothing
            smoothed = cv2.edgePreservingFilter(gray, flags=2, sigma_s=50, sigma_r=0.4)
            result = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        
        elif method == 'morphological':
            # Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            result = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        elif method == 'unsharp_mask':
            # Unsharp masking for text clarity
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            result = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        elif method == 'gradient':
            # Gradient-based enhancement
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            result = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        else:  # 'simple'
            # Simple threshold
            result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return Image.fromarray(result)
    
    def extract_with_multiple_configs(self, image, image_name='image'):
        """Try multiple Tesseract configurations to find the best result"""
        results = {}
        
        for config_name, config in self.ocr_configs.items():
            try:
                text = pytesseract.image_to_string(image, config=config)
                extracted_number = self.parse_measurement_text(text, image_name, config_name)
                if extracted_number is not None:
                    results[config_name] = {
                        'value': extracted_number,
                        'raw_text': text.strip(),
                        'confidence': self.calculate_confidence(text, extracted_number)
                    }
            except Exception as e:
                st.warning(f"Config {config_name} failed: {str(e)}")
                continue
        
        return results
    
    def parse_measurement_text(self, text, image_name='image', config_name='unknown'):
        """Enhanced parsing of measurement text with better pattern matching"""
        if not text or not text.strip():
            return None
            
        # Clean the text
        cleaned_text = re.sub(r'[^\d.,mMkKcCeEtTrRsS\s]', ' ', text)
        
        # Multiple pattern matching strategies
        patterns = [
            # Standard meter patterns
            (r'(\d+(?:[.,]\d+)?)\s*m(?:eter)?s?\b', 1),
            (r'(\d+(?:[.,]\d+)?)\s*M(?:ETER)?S?\b', 1),
            
            # Kilometer patterns  
            (r'(\d+(?:[.,]\d+)?)\s*km\b', 1000),
            (r'(\d+(?:[.,]\d+)?)\s*KM\b', 1000),
            (r'(\d+(?:[.,]\d+)?)\s*k\s*m\b', 1000),
            
            # Centimeter patterns
            (r'(\d+(?:[.,]\d+)?)\s*cm\b', 0.01),
            (r'(\d+(?:[.,]\d+)?)\s*CM\b', 0.01),
            (r'(\d+(?:[.,]\d+)?)\s*c\s*m\b', 0.01),
            
            # Millimeter patterns
            (r'(\d+(?:[.,]\d+)?)\s*mm\b', 0.001),
            (r'(\d+(?:[.,]\d+)?)\s*MM\b', 0.001),
            (r'(\d+(?:[.,]\d+)?)\s*m\s*m\b', 0.001),
            
            # Standalone number patterns (assume meters)
            (r'\b(\d+(?:[.,]\d+)?)\s*m\b', 1),
            (r'\b(\d+(?:[.,]\d+)?)\s*M\b', 1),
            
            # Numbers followed by common OCR mistakes
            (r'(\d+(?:[.,]\d+)?)\s*[nm]\b', 1),  # 'n' misread as 'm'
            (r'(\d+(?:[.,]\d+)?)\s*[rn]n\b', 1), # 'rn' or 'nn' misread as 'm'
        ]
        
        for pattern, multiplier in patterns:
            matches = re.finditer(pattern, cleaned_text.lower())
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '.')
                    value = float(value_str)
                    meters = value * multiplier
                    
                    # Sanity check - wire lengths should be reasonable
                    if 0.001 <= meters <= 100000:  # 1mm to 100km
                        print(f"[{config_name}] Extracted from {image_name}: {meters}m (pattern: {pattern})")
                        return meters
                except (ValueError, IndexError):
                    continue
        
        # Try to extract any number if no unit found (assume meters)
        number_pattern = r'\b(\d+(?:[.,]\d+)?)\b'
        numbers = re.findall(number_pattern, cleaned_text)
        if numbers:
            try:
                value = float(numbers[0].replace(',', '.'))
                if 0.1 <= value <= 10000:  # Reasonable range for meters
                    print(f"[{config_name}] Extracted number without unit from {image_name}: {value}m")
                    return value
            except ValueError:
                pass
        
        return None
    
    def calculate_confidence(self, text, extracted_value):
        """Calculate confidence score for extracted value"""
        if not text or extracted_value is None:
            return 0
        
        score = 50  # Base score
        
        # Length of text (longer usually better)
        if len(text.strip()) > 5:
            score += 20
        
        # Contains unit indicators
        if re.search(r'[mM]', text):
            score += 20
        
        # Clean extraction (digits and units)
        clean_chars = len(re.findall(r'[\d.,mMkKcC]', text))
        total_chars = len(text.strip())
        if total_chars > 0:
            score += (clean_chars / total_chars) * 10
        
        return min(score, 100)

def main():
    st.title("🔍 Enhanced Tesseract Wire Length Calculator")
    st.markdown("**Advanced OCR with multiple Tesseract strategies for maximum accuracy**")
    
    # Initialize extractor
    extractor = TesseractWireExtractor()
    
    # Initialize session state
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []
    
    # Sidebar settings
    st.sidebar.header("⚙️ OCR Settings")
    
    preprocessing_method = st.sidebar.selectbox(
        "Image Preprocessing",
        ['adaptive', 'high_contrast', 'edge_preserving', 'morphological', 'unsharp_mask', 'gradient', 'simple'],
        help="Choose preprocessing method for better OCR accuracy"
    )
    
    show_all_configs = st.sidebar.checkbox("Show All OCR Attempts", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 50)
    
    # File upload
    st.header("📁 Upload Wire Length Images")
    uploaded_files = st.file_uploader(
        "Choose images with wire length measurements",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.header("🔄 Processing with Enhanced Tesseract OCR")
        
        if st.button("🚀 Extract All Measurements", type="primary"):
            st.session_state.all_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"📷 Processing: {uploaded_file.name}")
                
                # Load and display original
                original_image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Image**")
                    st.image(original_image, use_container_width=True)
                
                # Preprocess
                with st.spinner("Preprocessing image..."):
                    processed_image = extractor.preprocess_image(original_image, preprocessing_method)
                
                with col2:
                    st.write("**Processed Image**")
                    st.image(processed_image, use_container_width=True)
                
                # Extract with multiple configs
                with st.spinner("Running multiple OCR configurations..."):
                    results = extractor.extract_with_multiple_configs(processed_image, uploaded_file.name)
                
                # Display results
                if results:
                    # Find best result
                    best_config = max(results.keys(), key=lambda k: results[k]['confidence'])
                    best_result = results[best_config]
                    
                    if best_result['confidence'] >= confidence_threshold:
                        st.success(f"✅ **Best Result**: {best_result['value']:.3f} meters (Confidence: {best_result['confidence']:.1f}%)")
                        st.session_state.all_results.append({
                            'file': uploaded_file.name,
                            'length': best_result['value'],
                            'confidence': best_result['confidence'],
                            'method': best_config,
                            'raw_text': best_result['raw_text']
                        })
                    else:
                        st.warning(f"⚠️ Low confidence result: {best_result['value']:.3f} meters ({best_result['confidence']:.1f}%)")
                    
                    # Show all attempts if requested
                    if show_all_configs:
                        with st.expander(f"🔍 All OCR Attempts ({len(results)} configs tried)"):
                            for config, result in sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True):
                                st.write(f"**{config}**: {result['value']:.3f}m (Confidence: {result['confidence']:.1f}%)")
                                st.code(f"Raw text: '{result['raw_text']}'")
                else:
                    st.error("❌ No measurements detected with any OCR configuration")
                
                st.markdown("---")
        
        # Results section
        if st.session_state.all_results:
            st.header("📊 Extracted Measurements")
            
            # Create results dataframe
            df = pd.DataFrame(st.session_state.all_results)
            st.dataframe(df, use_container_width=True)
            
            # Calculate distances
            if len(st.session_state.all_results) >= 2:
                st.header("📏 Distance Matrix")
                
                # Create distance matrix
                n = len(st.session_state.all_results)
                distance_matrix = np.zeros((n, n))
                file_names = [r['file'] for r in st.session_state.all_results]
                
                for i in range(n):
                    for j in range(n):
                        distance_matrix[i][j] = abs(st.session_state.all_results[i]['length'] - 
                                                  st.session_state.all_results[j]['length'])
                
                distance_df = pd.DataFrame(distance_matrix, 
                                         index=file_names, 
                                         columns=file_names)
                distance_df = distance_df.round(3)
                
                st.dataframe(distance_df, use_container_width=True)
                
                # Summary statistics
                st.header("📈 Summary Statistics")
                lengths = [r['length'] for r in st.session_state.all_results]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", len(lengths))
                with col2:
                    st.metric("Max Length", f"{max(lengths):.3f} m")
                with col3:
                    st.metric("Min Length", f"{min(lengths):.3f} m")
                with col4:
                    st.metric("Average Length", f"{np.mean(lengths):.3f} m")
            
            # Export options
            st.header("💾 Export Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Generate Report"):
                    report = f"""# Wire Length Measurement Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Preprocessing Method: {preprocessing_method}
Confidence Threshold: {confidence_threshold}%

## Extracted Measurements:
"""
                    for result in st.session_state.all_results:
                        report += f"- {result['file']}: {result['length']:.3f}m (Confidence: {result['confidence']:.1f}%, Method: {result['method']})\n"
                    
                    if len(st.session_state.all_results) >= 2:
                        report += "\n## Pairwise Distances:\n"
                        for i in range(len(st.session_state.all_results)):
                            for j in range(i+1, len(st.session_state.all_results)):
                                dist = abs(st.session_state.all_results[i]['length'] - st.session_state.all_results[j]['length'])
                                report += f"- {st.session_state.all_results[i]['file']} ↔ {st.session_state.all_results[j]['file']}: {dist:.3f}m\n"
                    
                    st.code(report)
            
            with col2:
                if st.button("📊 Download CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Measurements CSV",
                        data=csv,
                        file_name=f"wire_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    # Help section
    with st.expander("ℹ️ How This Enhanced OCR Works", expanded=False):
        st.markdown("""
        ### 🎯 Multi-Strategy OCR Approach:
        
        **1. Image Preprocessing Options:**
        - **Adaptive**: Smart preprocessing based on image characteristics
        - **High Contrast**: Enhanced contrast and sharpening
        - **Edge Preserving**: Smooth noise while keeping text edges sharp
        - **Morphological**: Text-specific morphological operations
        - **Unsharp Mask**: Professional sharpening technique
        - **Gradient**: Edge-based enhancement
        
        **2. Multiple Tesseract Configurations:**
        - **Default**: Standard OCR for mixed content
        - **Digits Only**: Numbers and decimal points only
        - **Measurements**: Numbers with measurement units
        - **Single Line/Word**: For isolated measurements
        - **Sparse Text**: For scattered text elements
        - **Handwritten**: Specialized for handwritten numbers
        - **Legacy Engine**: Alternative OCR engine
        
        **3. Smart Pattern Recognition:**
        - Supports meters (m), kilometers (km), centimeters (cm), millimeters (mm)
        - Handles OCR errors (e.g., 'm' misread as 'n' or 'rn')
        - Confidence scoring for result validation
        - Automatic unit conversion to meters
        
        **4. Quality Assurance:**
        - Multiple attempts with different strategies
        - Confidence scoring for each result
        - Sanity checks for reasonable wire lengths
        - Detailed logging of all attempts
        """)

if __name__ == "__main__":
    main()
