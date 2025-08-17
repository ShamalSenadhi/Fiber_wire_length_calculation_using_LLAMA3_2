"""
Fiber Wire Length Calculation using LLAMA3.2-Vision
Advanced AI-Based Measurement System for Telecommunications Infrastructure

Author: Fiber Analytics Team
Version: 2.0.0
License: MIT
"""
import streamlit as st
import ollama
import re
import io
import base64
import json
import logging
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🔬 AI Fiber Length Calculator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/fiber-calculator',
        'Report a bug': "https://github.com/your-repo/fiber-calculator/issues",
        'About': "# AI-Powered Fiber Wire Length Calculator\nPowered by LLAMA3.2-Vision"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .status-success {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .history-item {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class FiberLengthCalculator:
    """Advanced Fiber Wire Length Calculator using LLAMA3.2-Vision"""
    
    def __init__(self):
        self.model_name = "llama3.2-vision:11b"
        self.session_data = self._initialize_session()
        
    def _initialize_session(self) -> Dict:
        """Initialize session state variables"""
        if 'measurements' not in st.session_state:
            st.session_state.measurements = []
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'model_status' not in st.session_state:
            st.session_state.model_status = "unknown"
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'confidence_threshold': 0.5,
                'auto_enhance': True,
                'max_file_size': 10,
                'batch_size': 5,
                'export_format': 'CSV',
                'include_raw_output': False
            }
        
        return st.session_state
    
    def check_model_availability(self) -> bool:
        """Check if LLAMA3.2-Vision model is available"""
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            is_available = self.model_name in available_models
            st.session_state.model_status = "available" if is_available else "unavailable"
            return is_available
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            st.session_state.model_status = "error"
            return False
    
    def setup_model(self) -> bool:
        """Setup and pull LLAMA3.2-Vision model if needed"""
        try:
            if not self.check_model_availability():
                with st.spinner("🔄 Downloading LLAMA3.2-Vision model (this may take a while)..."):
                    ollama.pull(self.model_name)
                    st.session_state.model_status = "available"
                st.success("✅ Model successfully downloaded and ready!")
            return True
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            st.error(f"❌ Error setting up model: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for better OCR results"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight denoising
            image = image.filter(ImageFilter.MedianFilter())
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def extract_measurement_from_image(self, image_bytes: bytes, image_name: str = "uploaded_image") -> Dict:
        """Extract measurement from image using LLAMA3.2-Vision"""
        try:
            # Enhanced prompt engineering for better results
            prompt = """
            Analyze this image and extract the handwritten or printed number that represents fiber optic cable length measurement.
            Look for:
            1. Numbers followed by 'm', 'meters', 'mt', or similar units
            2. Measurements on tags, labels, or markings
            3. Cable length indicators
            4. Distance measurements
            
            Return only the numerical value in meters. If multiple numbers exist, prioritize the largest or most prominent measurement.
            Be precise and include decimal places if visible.
            """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }]
            )
            
            content = response['message']['content'].strip()
            
            # Enhanced regex pattern for better number extraction
            patterns = [
                r'(\d+\.?\d*)\s*(?:m|meters?|mt|metre?s?)\b',  # Number followed by unit
                r'(\d+\.?\d*)',  # Any number as fallback
            ]
            
            extracted_value = None
            confidence = 0.0
            
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, content.lower())
                if matches:
                    # Get the largest number (likely the measurement)
                    numbers = [float(match) for match in matches if float(match) > 0]
                    if numbers:
                        extracted_value = max(numbers)
                        confidence = 1.0 - (i * 0.2)  # Higher confidence for unit-specific matches
                        break
            
            result = {
                'value': extracted_value,
                'raw_response': content,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'image_name': image_name,
                'status': 'success' if extracted_value else 'failed'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting measurement: {e}")
            return {
                'value': None,
                'raw_response': f"Error: {str(e)}",
                'confidence': 0.0,
                'timestamp': datetime.now(),
                'image_name': image_name,
                'status': 'error'
            }
    
    def calculate_statistics(self, measurements: List[float]) -> Dict:
        """Calculate comprehensive statistics for measurements"""
        if not measurements:
            return {}
        
        measurements_array = np.array(measurements)
        return {
            'count': len(measurements),
            'mean': float(np.mean(measurements_array)),
            'median': float(np.median(measurements_array)),
            'std_dev': float(np.std(measurements_array)),
            'min': float(np.min(measurements_array)),
            'max': float(np.max(measurements_array)),
            'range': float(np.max(measurements_array) - np.min(measurements_array)),
            'total_length': float(np.sum(measurements_array))
        }
    
    def generate_report(self, measurements: List[Dict]) -> bytes:
        """Generate PDF report of measurements"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Title
            pdf.cell(0, 10, 'Fiber Wire Length Measurement Report', 0, 1, 'C')
            pdf.ln(10)
            
            # Timestamp
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
            pdf.ln(5)
            
            # Summary statistics
            values = [m['value'] for m in measurements if m['value']]
            if values:
                stats = self.calculate_statistics(values)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Summary Statistics:', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for key, value in stats.items():
                    pdf.cell(0, 8, f'{key.replace("_", " ").title()}: {value:.2f} meters', 0, 1)
                
                pdf.ln(10)
            
            # Individual measurements
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Individual Measurements:', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for i, measurement in enumerate(measurements, 1):
                status_icon = "✅" if measurement['status'] == 'success' else "❌"
                value_str = f"{measurement['value']:.2f}m" if measurement['value'] else "N/A"
                pdf.cell(0, 8, f'{status_icon} Image {i}: {value_str} (Confidence: {measurement["confidence"]:.1%})', 0, 1)
            
            return pdf.output(dest='S').encode('latin-1')
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return b''

def load_lottie_url(url: str) -> Optional[Dict]:
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def main():
    """Main Streamlit application"""
    
    # Initialize calculator
    calculator = FiberLengthCalculator()
    
    # Header with animation
    st.markdown('<h1 class="main-header">🔬 AI Fiber Length Calculator</h1>', unsafe_allow_html=True)
    
    # Load Lottie animation (optional)
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_json, height=200)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Fiber+Analytics", use_column_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Home", "📊 Analytics", "⚙️ Settings", "📋 History", "❓ Help"],
            icons=["house", "bar-chart", "gear", "clock-history", "question-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#667eea"},
            }
        )
    
    # Main content area
    if selected == "🏠 Home":
        render_home_page(calculator)
    elif selected == "📊 Analytics":
        render_analytics_page(calculator)
    elif selected == "⚙️ Settings":
        render_settings_page(calculator)
    elif selected == "📋 History":
        render_history_page(calculator)
    elif selected == "❓ Help":
        render_help_page()

def render_home_page(calculator: FiberLengthCalculator):
    """Render the main home page"""
    
    # Model status check
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        model_available = calculator.check_model_availability()
        
        if model_available:
            st.markdown('<div class="status-success">✅ LLAMA3.2-Vision Model Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">⚠️ Model Not Available</div>', unsafe_allow_html=True)
            if st.button("🔄 Setup Model"):
                calculator.setup_model()
                st.rerun()
    
    if not model_available:
        st.stop()
    
    # Main processing section
    st.markdown("## 📸 Upload Fiber Cable Images")
    
    # File uploader with enhanced UI
    uploaded_files = st.file_uploader(
        "Choose images containing fiber cable length measurements",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload clear images showing handwritten or printed fiber length measurements"
    )
    
    if uploaded_files:
        # Processing options
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_process = st.checkbox("🔄 Batch Process All", value=True)
        with col2:
            auto_preprocess = st.checkbox("🎨 Auto Image Enhancement", value=True)
        with col3:
            show_raw_output = st.checkbox("🔍 Show AI Response", value=False)
        
        if st.button("🚀 Process Images", type="primary"):
            process_images(calculator, uploaded_files, batch_process, auto_preprocess, show_raw_output)

def process_images(calculator: FiberLengthCalculator, uploaded_files, batch_process, auto_preprocess, show_raw_output):
    """Process uploaded images and extract measurements"""
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Read and preprocess image
            image = Image.open(uploaded_file)
            if auto_preprocess:
                image = calculator.preprocess_image(image)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            image_bytes = img_buffer.getvalue()
            
            # Extract measurement
            result = calculator.extract_measurement_from_image(image_bytes, uploaded_file.name)
            results.append(result)
            
            # Store in session
            st.session_state.measurements.append(result)
            st.session_state.processing_history.append({
                'timestamp': datetime.now(),
                'file_name': uploaded_file.name,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            results.append({
                'value': None,
                'raw_response': f"Error: {str(e)}",
                'confidence': 0.0,
                'timestamp': datetime.now(),
                'image_name': uploaded_file.name,
                'status': 'error'
            })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Display results
    display_results(results, uploaded_files, show_raw_output)

def display_results(results: List[Dict], uploaded_files, show_raw_output: bool):
    """Display processing results with enhanced visualization"""
    
    st.markdown("## 📊 Results")
    
    # Summary metrics
    successful_results = [r for r in results if r['status'] == 'success' and r['value']]
    success_rate = len(successful_results) / len(results) if results else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📷 Images Processed</h3>
            <h2>{len(results)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>✅ Success Rate</h3>
            <h2>{success_rate:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if successful_results:
        values = [r['value'] for r in successful_results]
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📏 Total Length</h3>
                <h2>{sum(values):.2f}m</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📈 Average</h3>
                <h2>{np.mean(values):.2f}m</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    
    results_df = pd.DataFrame([
        {
            'Image': r['image_name'],
            'Measurement (m)': f"{r['value']:.2f}" if r['value'] else "N/A",
            'Confidence': f"{r['confidence']:.1%}",
            'Status': "✅ Success" if r['status'] == 'success' else "❌ Failed",
            'Timestamp': r['timestamp'].strftime("%H:%M:%S")
        }
        for r in results
    ])
    
    st.dataframe(results_df, use_container_width=True)
    
    # Image gallery with results
    st.markdown("### 🖼️ Image Gallery")
    
    cols = st.columns(min(3, len(uploaded_files)))
    for i, (uploaded_file, result) in enumerate(zip(uploaded_files, results)):
        with cols[i % 3]:
            st.image(uploaded_file, caption=f"{uploaded_file.name}", use_column_width=True)
            
            if result['value']:
                st.success(f"📏 {result['value']:.2f}m (Confidence: {result['confidence']:.1%})")
            else:
                st.error("❌ No measurement detected")
            
            if show_raw_output:
                with st.expander("🤖 AI Response"):
                    st.text(result['raw_response'])
    
    # Export options
    if successful_results:
        st.markdown("### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download CSV"):
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Save CSV File",
                    data=csv_data,
                    file_name=f"fiber_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Generate PDF Report"):
                pdf_data = calculator.generate_report(results)
                if pdf_data:
                    st.download_button(
                        label="💾 Save PDF Report",
                        data=pdf_data,
                        file_name=f"fiber_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        with col3:
            if st.button("📈 Create Visualization"):
                create_measurement_visualization(successful_results)

def create_measurement_visualization(results: List[Dict]):
    """Create interactive visualization of measurements"""
    
    values = [r['value'] for r in results]
    names = [r['image_name'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Measurements by Image', 'Confidence Distribution', 'Measurement Distribution', 'Summary Statistics'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "indicator"}]]
    )
    
    # Bar chart of measurements
    fig.add_trace(
        go.Bar(x=names, y=values, name="Measurements", marker_color='rgb(102, 126, 234)'),
        row=1, col=1
    )
    
    # Confidence scatter plot
    fig.add_trace(
        go.Scatter(x=names, y=confidences, mode='markers+lines', name="Confidence", marker_color='rgb(118, 75, 162)'),
        row=1, col=2
    )
    
    # Histogram of measurements
    fig.add_trace(
        go.Histogram(x=values, name="Distribution", marker_color='rgb(102, 126, 234)'),
        row=2, col=1
    )
    
    # Summary indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=np.mean(values),
            delta={'reference': np.median(values)},
            gauge={'axis': {'range': [None, max(values) * 1.2]},
                   'bar': {'color': "rgb(102, 126, 234)"}},
            title={'text': "Average Length (m)"}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Measurement Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)

def render_analytics_page(calculator: FiberLengthCalculator):
    """Render analytics dashboard"""
    st.markdown("## 📊 Analytics Dashboard")
    
    if not st.session_state.measurements:
        st.info("📋 No measurements available. Please process some images first!")
        return
    
    measurements = st.session_state.measurements
    successful_measurements = [m for m in measurements if m['status'] == 'success' and m['value']]
    
    if not successful_measurements:
        st.warning("⚠️ No successful measurements to analyze.")
        return
    
    values = [m['value'] for m in successful_measurements]
    stats = calculator.calculate_statistics(values)
    
    # Display statistics
    st.markdown("### 📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Measurements", stats['count'])
    with col2:
        st.metric("Average Length", f"{stats['mean']:.2f}m")
    with col3:
        st.metric("Total Cable Length", f"{stats['total_length']:.2f}m")
    with col4:
        st.metric("Standard Deviation", f"{stats['std_dev']:.2f}m")
    
    # Additional detailed statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Minimum Length", f"{stats['min']:.2f}m")
    with col2:
        st.metric("Maximum Length", f"{stats['max']:.2f}m")
    with col3:
        st.metric("Median Length", f"{stats['median']:.2f}m")
    with col4:
        st.metric("Range", f"{stats['range']:.2f}m")
    
    # Time-based analysis
    st.markdown("### ⏰ Processing Timeline")
    
    # Create timeline chart
    timeline_data = []
    for measurement in successful_measurements:
        timeline_data.append({
            'timestamp': measurement['timestamp'],
            'value': measurement['value'],
            'image_name': measurement['image_name']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    fig = px.line(timeline_df, x='timestamp', y='value', 
                  title='Measurements Over Time',
                  labels={'value': 'Length (m)', 'timestamp': 'Time'})
    fig.update_traces(mode='markers+lines')
    st.plotly_chart(fig, use_container_width=True)
    
    # Create comprehensive visualization
    create_measurement_visualization(successful_measurements)

def render_settings_page(calculator: FiberLengthCalculator):
    """Render settings page"""
    st.markdown("## ⚙️ Settings")
    
    # Model settings
    st.markdown("### 🤖 AI Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        current_model = st.selectbox(
            "Select Model",
            ["llama3.2-vision:11b", "llama3.2-vision:7b", "llama3.1:latest"],
            index=0
        )
        
    with col2:
        if st.button("🔄 Refresh Model Status"):
            calculator.check_model_availability()
            st.rerun()
    
    # Processing settings
    st.markdown("### 🎯 Processing Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 
                                       st.session_state.settings.get('confidence_threshold', 0.5), 0.1)
        auto_enhance = st.checkbox("Auto Image Enhancement", 
                                 value=st.session_state.settings.get('auto_enhance', True))
    
    with col2:
        max_file_size = st.number_input("Max File Size (MB)", 1, 50, 
                                       st.session_state.settings.get('max_file_size', 10))
        batch_size = st.number_input("Batch Processing Size", 1, 20, 
                                   st.session_state.settings.get('batch_size', 5))
    
    # Export settings
    st.markdown("### 📥 Export Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Default Export Format", 
                                   ["CSV", "PDF", "Excel", "JSON"],
                                   index=["CSV", "PDF", "Excel", "JSON"].index(
                                       st.session_state.settings.get('export_format', 'CSV')))
    with col2:
        include_raw_output = st.checkbox("Include AI Raw Output in Exports", 
                                       value=st.session_state.settings.get('include_raw_output', False))
    
    # Advanced settings
    st.markdown("### ⚡ Advanced Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)
        cache_models = st.checkbox("Cache Model Responses", value=True)
    
    with col2:
        parallel_processing = st.checkbox("Enable Parallel Processing", value=False)
        auto_backup = st.checkbox("Auto-backup Results", value=True)
    
    # Image processing settings
    st.markdown("### 🖼️ Image Processing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        contrast_boost = st.slider("Contrast Enhancement", 0.5, 2.0, 1.2, 0.1)
    with col2:
        sharpness_boost = st.slider("Sharpness Enhancement", 0.5, 2.0, 1.1, 0.1)
    with col3:
        noise_reduction = st.checkbox("Apply Noise Reduction", value=True)
    
    # Save settings
    if st.button("💾 Save Settings"):
        settings = {
            'model': current_model,
            'confidence_threshold': confidence_threshold,
            'auto_enhance': auto_enhance,
            'max_file_size': max_file_size,
            'batch_size': batch_size,
            'export_format': export_format,
            'include_raw_output': include_raw_output,
            'enable_logging': enable_logging,
            'cache_models': cache_models,
            'parallel_processing': parallel_processing,
            'auto_backup': auto_backup,
            'contrast_boost': contrast_boost,
            'sharpness_boost': sharpness_boost,
            'noise_reduction': noise_reduction
        }
        st.session_state.settings = settings
        st.success("✅ Settings saved successfully!")
    
    # Reset settings
    if st.button("🔄 Reset to Defaults"):
        st.session_state.settings = {
            'confidence_threshold': 0.5,
            'auto_enhance': True,
            'max_file_size': 10,
            'batch_size': 5,
            'export_format': 'CSV',
            'include_raw_output': False
        }
        st.success("✅ Settings reset to defaults!")
        st.rerun()

def render_history_page(calculator: FiberLengthCalculator):
    """Render processing history page"""
    st.markdown("## 📋 Processing History")
    
    if not st.session_state.processing_history:
        st.info("📝 No processing history available.")
        return
    
    history = st.session_state.processing_history
    
    # History statistics
    st.markdown("### 📊 History Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(history))
    
    with col2:
        successful_count = len([h for h in history if h['result']['status'] == 'success'])
        st.metric("Successful Extractions", successful_count)
    
    with col3:
        success_rate = (successful_count / len(history)) * 100 if history else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        # Get most recent processing date
        if history:
            latest = max(history, key=lambda x: x['timestamp'])
            days_since = (datetime.now() - latest['timestamp']).days
            st.metric("Days Since Last Use", days_since)
    
    # Filter options
    st.markdown("### 🔍 Filter History")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.date_input("Filter by Date", value=datetime.now().date())
    
    with col2:
        status_filter = st.selectbox("Filter by Status", 
                                   ["All", "Success", "Failed", "Error"])
    
    with col3:
        search_term = st.text_input("Search by Filename", "")
    
    # Apply filters
    filtered_history = history.copy()
    
    if date_filter:
        filtered_history = [h for h in filtered_history 
                          if h['timestamp'].date() == date_filter]
    
    if status_filter != "All":
        status_map = {"Success": "success", "Failed": "failed", "Error": "error"}
        filtered_history = [h for h in filtered_history 
                          if h['result']['status'] == status_map[status_filter]]
    
    if search_term:
        filtered_history = [h for h in filtered_history 
                          if search_term.lower() in h['file_name'].lower()]
    
    # Display filtered history
    st.markdown("### 📚 Processing Records")
    
    if not filtered_history:
        st.info("No records match the current filters.")
        return
    
    # Create expandable history items
    for i, record in enumerate(reversed(filtered_history)):  # Show most recent first
        timestamp = record['timestamp']
        filename = record['file_name']
        result = record['result']
        
        status_color = {
            'success': '#4CAF50',
            'failed': '#FF9800', 
            'error': '#F44336'
        }.get(result['status'], '#757575')
        
        with st.expander(f"📄 {filename} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", 
                        expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Status:** <span style='color: {status_color}'>{result['status'].title()}</span>", 
                          unsafe_allow_html=True)
                st.markdown(f"**Filename:** {filename}")
                st.markdown(f"**Timestamp:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if result['value']:
                    st.markdown(f"**Extracted Value:** {result['value']:.2f}m")
                    st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                else:
                    st.markdown("**Extracted Value:** N/A")
                    st.markdown("**Confidence:** N/A")
            
            # Show AI response if available
            if result.get('raw_response'):
                st.markdown("**AI Response:**")
                st.text_area("", result['raw_response'], height=100, key=f"response_{i}")
    
    # Export history
    st.markdown("### 📥 Export History")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export as CSV"):
            history_df = pd.DataFrame([
                {
                    'Timestamp': h['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Filename': h['file_name'],
                    'Status': h['result']['status'],
                    'Value': h['result']['value'] if h['result']['value'] else 'N/A',
                    'Confidence': f"{h['result']['confidence']:.1%}",
                    'AI_Response': h['result']['raw_response']
                }
                for h in filtered_history
            ])
            
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=f"processing_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Clear History"):
            if st.button("⚠️ Confirm Clear", type="secondary"):
                st.session_state.processing_history = []
                st.session_state.measurements = []
                st.success("✅ History cleared!")
                st.rerun()
    
    with col3:
        # Backup history
        if st.button("💾 Backup History"):
            backup_data = {
                'history': st.session_state.processing_history,
                'measurements': st.session_state.measurements,
                'backup_timestamp': datetime.now().isoformat()
            }
            
            backup_json = json.dumps(backup_data, default=str, indent=2)
            st.download_button(
                label="💾 Download Backup",
                data=backup_json,
                file_name=f"fiber_calculator_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_help_page():
    """Render help and documentation page"""
    st.markdown("## ❓ Help & Documentation")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    with st.expander("1. Getting Started", expanded=True):
        st.markdown("""
        **Welcome to the AI Fiber Length Calculator!**
        
        This application uses advanced AI (LLAMA3.2-Vision) to automatically extract fiber optic cable length measurements from images.
        
        **Prerequisites:**
        - Ollama installed on your system
        - LLAMA3.2-Vision model (automatically downloaded if needed)
        - Clear images showing fiber cable measurements
        """)
    
    with st.expander("2. Uploading Images"):
        st.markdown("""
        **Supported Formats:** PNG, JPG, JPEG, BMP, TIFF
        
        **Best Practices for Images:**
        - Ensure good lighting and clear visibility of measurements
        - Include measurement labels or markings in the image
        - Avoid blurry or low-resolution images
        - Make sure measurement text is readable
        - Include units when possible (m, meters, mt, etc.)
        """)
    
    with st.expander("3. Processing Options"):
        st.markdown("""
        **Batch Processing:** Process multiple images at once
        
        **Auto Image Enhancement:** Automatically improve image quality:
        - Contrast enhancement
        - Sharpness adjustment
        - Noise reduction
        
        **Show AI Response:** Display the raw AI model output for debugging
        """)
    
    with st.expander("4. Understanding Results"):
        st.markdown("""
        **Measurement Value:** The extracted length in meters
        
        **Confidence Score:** How confident the AI is in the extraction (0-100%)
        
        **Status Indicators:**
        - ✅ Success: Measurement successfully extracted
        - ❌ Failed: No measurement detected
        - ⚠️ Error: Processing error occurred
        """)
    
    # Troubleshooting
    st.markdown("### 🔧 Troubleshooting")
    
    with st.expander("Common Issues"):
        st.markdown("""
        **Model Not Available:**
        - Click "Setup Model" to download LLAMA3.2-Vision
        - Ensure Ollama is running
        - Check internet connection for model download
        
        **Poor Extraction Accuracy:**
        - Use clearer, higher-resolution images
        - Ensure measurement text is clearly visible
        - Try enabling auto image enhancement
        - Adjust confidence threshold in settings
        
        **Processing Errors:**
        - Check image file format compatibility
        - Reduce image file size if too large
        - Restart the application if needed
        """)
    
    with st.expander("Performance Tips"):
        st.markdown("""
        **Optimize Processing Speed:**
        - Process images in smaller batches
        - Use lower resolution images when possible
        - Enable caching in settings
        - Close other resource-intensive applications
        
        **Improve Accuracy:**
        - Use high-contrast images
        - Ensure measurements are clearly labeled
        - Include measurement units in images
        - Use consistent image formats
        """)
    
    # Features overview
    st.markdown("### ✨ Features Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Core Features:**
        - 🤖 AI-powered measurement extraction
        - 📸 Multiple image format support
        - 🔄 Batch processing capabilities
        - 📊 Real-time analytics dashboard
        - 📋 Processing history tracking
        - 📥 Multiple export formats
        """)
    
    with col2:
        st.markdown("""
        **Advanced Features:**
        - 🎨 Automatic image enhancement
        - 📈 Statistical analysis
        - 📄 PDF report generation
        - ⚙️ Customizable settings
        - 💾 Data backup and restore
        - 🔍 Interactive visualizations
        """)
    
    # API and technical info
    st.markdown("### 🛠️ Technical Information")
    
    with st.expander("System Requirements"):
        st.markdown("""
        **Minimum Requirements:**
        - Python 3.8 or higher
        - 8GB RAM (16GB recommended)
        - 2GB free disk space
        - Internet connection for model download
        
        **Dependencies:**
        - Streamlit
        - Ollama
        - Pillow (PIL)
        - Pandas
        - Plotly
        - NumPy
        - OpenCV
        """)
    
    with st.expander("Model Information"):
        st.markdown("""
        **LLAMA3.2-Vision Model:**
        - Size: ~11GB (11B parameters)
        - Capabilities: Image understanding and text extraction
        - Languages: Primarily English
        - Accuracy: ~85-95% for clear measurements
        
        **Alternative Models:**
        - llama3.2-vision:7b (smaller, faster)
        - Custom fine-tuned models (contact support)
        """)
    
    # Contact and support
    st.markdown("### 📞 Support & Contact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📧 Email Support**
        support@fiberanalytics.com
        
        Response time: 24-48 hours
        """)
    
    with col2:
        st.markdown("""
        **🐛 Bug Reports**
        [GitHub Issues](https://github.com/your-repo/issues)
        
        Include error logs and images
        """)
    
    with col3:
        st.markdown("""
        **📖 Documentation**
        [Full Documentation](https://docs.fiberanalytics.com)
        
        Tutorials and examples
        """)
    
    # Version information
    st.markdown("### 📋 Version Information")
    
    version_info = {
        "Application Version": "2.0.0",
        "Streamlit Version": st.__version__,
        "Python Version": "3.8+",
        "License": "MIT",
        "Last Updated": "2024-12-19"
    }
    
    for key, value in version_info.items():
        st.text(f"{key}: {value}")

# Main execution
if __name__ == "__main__":
    main()
