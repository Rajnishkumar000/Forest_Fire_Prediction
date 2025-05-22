
import streamlit as st
import requests
import json
import time
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import traceback

# Configure page
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

# Initialize session state
if 'training_tasks' not in st.session_state:
    st.session_state.training_tasks = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []

def safe_api_call(func, *args, **kwargs):
    """Safely make API calls with error handling"""
    try:
        return func(*args, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please ensure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_available_models():
    """Get available models from API"""
    def _get_models():
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()["models"]
        return []
    
    result = safe_api_call(_get_models)
    return result if result is not None else []

def start_training(config):
    """Start training via API"""
    def _start_training():
        response = requests.post(f"{API_BASE_URL}/train", json=config, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training failed: {response.text}")
            return None
    
    return safe_api_call(_start_training)

def get_training_status(task_id):
    """Get training status from API"""
    def _get_status():
        response = requests.get(f"{API_BASE_URL}/training-status/{task_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    
    return safe_api_call(_get_status)

def get_trained_models():
    """Get list of trained models"""
    def _get_models():
        response = requests.get(f"{API_BASE_URL}/trained-models", timeout=10)
        if response.status_code == 200:
            return response.json()["models"]
        return []
    
    result = safe_api_call(_get_models)
    return result if result is not None else []

def predict_image(model_id, rgb_image=None, ir_image=None):
    """Make prediction via API"""
    def _predict():
        files = {}
        if rgb_image:
            files['rgb_image'] = ('rgb.jpg', rgb_image, 'image/jpeg')
        if ir_image:
            files['ir_image'] = ('ir.jpg', ir_image, 'image/jpeg')
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            params={'model_id': model_id},
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    
    return safe_api_call(_predict)

# Connection test
def test_api_connection():
    """Test API connection"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

# Main App Layout
st.markdown('<h1 class="main-header">🔥 Fire Detection System</h1>', unsafe_allow_html=True)

# Connection status
if test_api_connection():
    st.sidebar.success("🟢 API Connected")
else:
    st.sidebar.error("🔴 API Disconnected")
    st.sidebar.info("Make sure FastAPI server is running on the specified URL")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🚀 Train Model", "🔍 Predict", "📊 Model Management"])

# Home Page
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Fire Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Multiple Models</h3>
            <p>Choose from various deep learning architectures</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🌈 Multi-Modal</h3>
            <p>Support for RGB, IR, and dual-stream inputs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time</h3>
            <p>Fast prediction and training monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Features
    - **Train Custom Models**: Select from multiple architectures and train on your data
    - **Real-time Monitoring**: Track training progress with live updates
    - **Multi-Modal Prediction**: Support for RGB, thermal (IR), and combined inputs
    - **Model Management**: Manage and compare trained models
    - **Easy Deployment**: RESTful API for integration
    
    ### 🚀 Get Started
    1. Navigate to **Train Model** to start training a new model
    2. Use **Predict** to test your trained models
    3. Manage your models in **Model Management**
    
    ### 🔧 System Status
    """)
    
    # System status
    col1, col2 = st.columns(2)
    with col1:
        if test_api_connection():
            st.success("✅ Backend API is running")
        else:
            st.error("❌ Backend API is not accessible")
    
    with col2:
        available_models = get_available_models()
        st.info(f"📊 {len(available_models)} model architectures available")

# Train Model Page
elif page == "🚀 Train Model":
    st.markdown('<h2 class="sub-header">Train New Model</h2>', unsafe_allow_html=True)
    
    # Check API connection first
    if not test_api_connection():
        st.error("❌ Cannot connect to API. Please ensure the FastAPI server is running.")
        st.stop()
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ Could not retrieve available models from API.")
        st.stop()
    
    # Training Configuration
    with st.form("training_form"):
        st.markdown("### 🎛️ Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox("Select Model", available_models, help="Choose the neural network architecture")
            mode = st.selectbox("Input Mode", ["rgb", "ir", "both"], help="Type of input images")
            batch_size = st.slider("Batch Size", 16, 128, 64, help="Number of samples per batch")
            learning_rate = st.select_slider("Learning Rate", 
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2], 
                value=1e-3, format_func=lambda x: f"{x:.0e}",
                help="Learning rate for optimization")
        
        with col2:
            epochs = st.slider("Epochs", 1, 100, 10, help="Number of training epochs")
            classes_num = st.number_input("Number of Classes", 2, 10, 3, help="Number of output classes")
            subset_rate = st.slider("Dataset Subset Rate", 0.01, 1.0, 0.01, 
                                   help="Fraction of dataset to use (for quick testing)")
            trainset_rate = st.slider("Training Set Rate", 0.5, 0.95, 0.8, 
                                     help="Fraction of data for training (rest for validation)")
        
        st.markdown("### 📁 Dataset Paths")
        col3, col4 = st.columns(2)
        with col3:
            path_rgb = st.text_input("RGB Images Path", value="E:/data/254pRGBImages", 
                                    help="Path to RGB images directory")
        with col4:
            path_ir = st.text_input("IR Images Path", value="E:/data/254pThermalImages",
                                   help="Path to thermal/IR images directory")
        
        # Show estimated training time
        estimated_time = epochs * subset_rate * 5  # Rough estimate
        st.info(f"⏱️ Estimated training time: ~{estimated_time:.1f} minutes")
        
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)
        
        if submitted:
            with st.spinner("Starting training..."):
                config = {
                    "model_name": model_name,
                    "mode": mode,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "classes_num": classes_num,
                    "subset_rate": subset_rate,
                    "trainset_rate": trainset_rate,
                    "path_rgb": path_rgb,
                    "path_ir": path_ir
                }
                
                result = start_training(config)
                if result:
                    st.session_state.training_tasks.append(result["task_id"])
                    st.success(f"✅ Training started! Task ID: {result['task_id']}")
                    time.sleep(1)
                    st.rerun()
    
    # Display active training tasks
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📊 Training Progress</h3>', unsafe_allow_html=True)
    
    if st.session_state.training_tasks:
        for task_id in st.session_state.training_tasks.copy():
            status = get_training_status(task_id)
            if status:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Model:** {status.get('model_name', 'N/A')} | **Mode:** {status.get('mode', 'N/A')}")
                        
                        if status['status'] == 'training':
                            progress = status.get('progress', 0) / 100
                            st.progress(progress, text=f"Progress: {status.get('progress', 0):.1f}%")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if 'epoch' in status:
                                    st.metric("Epoch", f"{status['epoch']}/{status.get('total_epochs', '?')}")
                            with col_b:
                                if 'current_loss' in status:
                                    st.metric("Current Loss", f"{status['current_loss']:.4f}")
                        
                        elif status['status'] == 'completed':
                            st.success("✅ Training Completed!")
                            if task_id in st.session_state.training_tasks:
                                st.session_state.training_tasks.remove(task_id)
                        
                        elif status['status'] == 'failed':
                            st.error(f"❌ Training Failed: {status.get('error', 'Unknown error')}")
                            if task_id in st.session_state.training_tasks:
                                st.session_state.training_tasks.remove(task_id)
                        
                        else:
                            st.info(f"Status: {status['status'].title()}")
                    
                    with col2:
                        st.code(f"ID: {task_id[:8]}...")
                        if st.button("🗑️", key=f"stop_{task_id}", help="Remove from tracking"):
                            st.session_state.training_tasks.remove(task_id)
                            st.rerun()
                
                st.markdown("---")
        
        # Auto-refresh for active training
        active_training = any(
            get_training_status(task_id) and 
            get_training_status(task_id)['status'] in ['training', 'queued', 'initializing', 'loading_data'] 
            for task_id in st.session_state.training_tasks
        )
        
        if active_training:
            with st.empty():
                st.info("🔄 Auto-refreshing in 3 seconds...")
                time.sleep(3)
                st.rerun()
    else:
        st.info("No active training tasks. Start a new training above.")

# Predict Page
elif page == "🔍 Predict":
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    # Check API connection
    if not test_api_connection():
        st.error("❌ Cannot connect to API. Please ensure the FastAPI server is running.")
        st.stop()
    
    # Get trained models
    trained_models = get_trained_models()
    
    if not trained_models:
        st.warning("⚠️ No trained models available. Please train a model first.")
        if st.button("🔄 Refresh Models"):
            st.rerun()
        st.stop()
    
    # Model selection
    model_options = {f"{model['model_name']} ({model['mode']})": model['id'] for model in trained_models}
    selected_model_name = st.selectbox("Select Trained Model", list(model_options.keys()))
    selected_model_id = model_options[selected_model_name]
    
    # Get model info
    selected_model_info = next(model for model in trained_models if model['id'] == selected_model_id)
    mode = selected_model_info['mode']
    
    st.info(f"📋 Selected model mode: **{mode.upper()}**")
    
    # Initialize file variables
    rgb_file = None
    ir_file = None
    rgb_image = None
    ir_image = None
    
    # Image upload based on mode
    col1, col2 = st.columns(2)
    
    if mode in ['rgb', 'both']:
        with col1:
            st.markdown("### 📸 RGB Image")
            rgb_file = st.file_uploader("Upload RGB Image", type=['png', 'jpg', 'jpeg'], key="rgb")
            if rgb_file is not None:
                try:
                    rgb_image = Image.open(rgb_file)
                    st.image(rgb_image, caption="RGB Image", use_column_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading RGB image: {str(e)}")
    
    if mode in ['ir', 'both']:
        with col2:
            st.markdown("### 🌡️ Thermal (IR) Image")
            ir_file = st.file_uploader("Upload IR Image", type=['png', 'jpg', 'jpeg'], key="ir")
            if ir_file is not None:
                try:
                    ir_image = Image.open(ir_file)
                    st.image(ir_image, caption="IR Image", use_column_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading IR image: {str(e)}")
    
    # Prediction button
    if st.button("🔮 Make Prediction", use_container_width=True):
        # Validate inputs
        validation_error = None
        
        if mode == 'rgb' and rgb_file is None:
            validation_error = "Please upload an RGB image for RGB mode."
        elif mode == 'ir' and ir_file is None:
            validation_error = "Please upload an IR image for IR mode."
        elif mode == 'both' and (rgb_file is None or ir_file is None):
            validation_error = "Please upload both RGB and IR images for both mode."
        
        if validation_error:
            st.error(f"❌ {validation_error}")
        else:
            with st.spinner("🔮 Making prediction..."):
                try:
                    # Prepare image data
                    rgb_data = None
                    ir_data = None
                    
                    if rgb_file is not None:
                        rgb_buffer = io.BytesIO()
                        rgb_image.save(rgb_buffer, format='JPEG')
                        rgb_data = rgb_buffer.getvalue()
                    
                    if ir_file is not None:
                        ir_buffer = io.BytesIO()
                        ir_image.save(ir_buffer, format='JPEG')
                        ir_data = ir_buffer.getvalue()
                    
                    # Make prediction
                    result = predict_image(selected_model_id, rgb_data, ir_data)
                    
                    if result:
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
                        
                        # Main prediction result
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Class", result['predicted_label'])
                        
                        with col2:
                            confidence_pct = result['confidence'] * 100
                            st.metric("Confidence", f"{confidence_pct:.2f}%")
                        
                        with col3:
                            # Color code based on prediction
                            if result['predicted_label'].lower() == 'fire':
                                status_color = "🔥 FIRE DETECTED"
                                status_style = "background-color: #ff4444; color: white; padding: 10px; border-radius: 5px; text-align: center;"
                            elif result['predicted_label'].lower() == 'smoke':
                                status_color = "💨 SMOKE DETECTED"
                                status_style = "background-color: #ff8800; color: white; padding: 10px; border-radius: 5px; text-align: center;"
                            else:
                                status_color = "✅ NO FIRE"
                                status_style = "background-color: #00aa00; color: white; padding: 10px; border-radius: 5px; text-align: center;"
                            
                            st.markdown(f'<div style="{status_style}">{status_color}</div>', unsafe_allow_html=True)
                        
                        # Probability visualization
                        st.markdown("### 📊 Class Probabilities")
                        
                        prob_data = pd.DataFrame(
                            list(result['probabilities'].items()),
                            columns=['Class', 'Probability']
                        )
                        
                        fig = px.bar(
                            prob_data, 
                            x='Class', 
                            y='Probability',
                            title="Prediction Probabilities",
                            color='Probability',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed probabilities
                        st.markdown("### 📋 Detailed Results")
                        prob_df = pd.DataFrame([result['probabilities']]).T
                        prob_df.columns = ['Probability']
                        prob_df['Percentage'] = prob_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(prob_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.error("Please check your images and try again.")

# Model Management Page
elif page == "📊 Model Management":
    st.markdown('<h2 class="sub-header">Model Management</h2>', unsafe_allow_html=True)
    
    # Check API connection
    if not test_api_connection():
        st.error("❌ Cannot connect to API. Please ensure the FastAPI server is running.")
        st.stop()
    
    # Get trained models
    trained_models = get_trained_models()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📈 Model Overview ({len(trained_models)} models)")
    with col2:
        if st.button("🔄 Refresh Models"):
            st.rerun()
    
    if not trained_models:
        st.info("📭 No trained models available yet. Train some models first!")
        st.markdown("[Go to Train Model page]()")
        st.stop()
    
    # Display models in cards
    for i, model in enumerate(trained_models):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{model['model_name']}**")
                st.write(f"Mode: {model['mode'].upper()}")
                st.write(f"Classes: {model['classes_num']}")
            
            with col2:
                st.write("**Model ID**")
                st.code(model['id'][:8] + "...")
            
            with col3:
                # Model architecture info
                if 'two_stream' in model['model_name'].lower():
                    st.write("🔄 **Two-Stream**")
                else:
                    st.write("➡️ **Single-Stream**")
            
            with col4:
                if st.button(f"🗑️ Delete", key=f"delete_{model['id']}", help="Delete this model"):
                    with st.spinner("Deleting model..."):
                        try:
                            response = requests.delete(f"{API_BASE_URL}/trained-models/{model['id']}", timeout=10)
                            if response.status_code == 200:
                                st.success("✅ Model deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete model")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
    
    # Model statistics
    if len(trained_models) > 0:
        st.markdown("### 📊 Model Statistics")
        
        try:
            model_df = pd.DataFrame(trained_models)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Model type distribution
                model_types = model_df['model_name'].value_counts()
                fig_types = px.pie(
                    values=model_types.values,
                    names=model_types.index,
                    title="Model Architecture Distribution"
                )
                st.plotly_chart(fig_types, use_container_width=True)
            
            with col2:
                # Mode distribution
                mode_counts = model_df['mode'].value_counts()
                fig_modes = px.bar(
                    x=mode_counts.index,
                    y=mode_counts.values,
                    title="Input Mode Distribution",
                    labels={'x': 'Mode', 'y': 'Count'}
                )
                st.plotly_chart(fig_modes, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error generating statistics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🔥 Fire Detection System | Built with Streamlit & FastAPI | 
    <a href="https://github.com" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)