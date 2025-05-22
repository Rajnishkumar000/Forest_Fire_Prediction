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
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from pathlib import Path
import glob

# Import your model classes (assuming they're in models.py)
try:
    from models import (
        Logistic_two_stream, Flame_one_stream, VGG16, Vgg_two_stream, 
        Logistic, Flame_two_stream, Mobilenetv2, Mobilenetv2_two_stream,
        LeNet5_one_stream, LeNet5_two_stream, Resnet18, Resnet18_two_stream
    )
    MODELS_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Model definitions not found. Only API mode will be available.")
    MODELS_AVAILABLE = False

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
    .offline-mode {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .online-mode {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .training-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Constants
TRAINED_MODELS_DIR = "trained_models"
MODEL_INFO_FILE = "model_info.json"
API_BASE_URL = "http://localhost:8000"

# Available model architectures
AVAILABLE_MODELS = [
    "Flame_one_stream", "Flame_two_stream", "VGG16", "Vgg_two_stream",
    "Logistic", "Logistic_two_stream", "Mobilenetv2", "Mobilenetv2_two_stream",
    "LeNet5_one_stream", "LeNet5_two_stream", "Resnet18", "Resnet18_two_stream"
]

MODEL_CUSTOM_LIST = [
    'Flame_one_stream', 'Flame_two_stream', 'Mobilenetv2_two_stream',
    'Vgg_two_stream', 'Logistic_two_stream', 'Resnet18_two_stream',
    'LeNet5_one_stream', 'LeNet5_two_stream'
]

# Initialize session state
if 'training_tasks' not in st.session_state:
    st.session_state.training_tasks = []
if 'offline_mode' not in st.session_state:
    st.session_state.offline_mode = False

# Utility Functions
def get_device():
    """Get available device"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_api_connection():
    """Test API connection"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_instance(model_name: str, classes_num: int = 3):
    """Get model instance and configuration"""
    if not MODELS_AVAILABLE:
        return None, None, None
    
    DEVICE = get_device()
    
    model_configs = {
        'Flame_one_stream': {'net': Flame_one_stream(), 'transform': False, 'size': 254},
        'Flame_two_stream': {'net': Flame_two_stream(), 'transform': False, 'size': 254},
        'VGG16': {'net': VGG16(classes_num), 'transform': True, 'size': 224},
        'Vgg_two_stream': {'net': Vgg_two_stream(), 'transform': True, 'size': 224},
        'Logistic': {'net': Logistic(classes_num), 'transform': False, 'size': 254},
        'Logistic_two_stream': {'net': Logistic_two_stream(classes_num), 'transform': False, 'size': 254},
        'Mobilenetv2': {'net': Mobilenetv2(classes_num), 'transform': True, 'size': 224},
        'Mobilenetv2_two_stream': {'net': Mobilenetv2_two_stream(), 'transform': True, 'size': 224},
        'Resnet18': {'net': Resnet18(classes_num), 'transform': True, 'size': 224},
        'Resnet18_two_stream': {'net': Resnet18_two_stream(), 'transform': True, 'size': 224},
        'LeNet5_one_stream': {'net': LeNet5_one_stream(), 'transform': False, 'size': 254},
        'LeNet5_two_stream': {'net': LeNet5_two_stream(), 'transform': False, 'size': 254}
    }
    
    if model_name not in model_configs:
        return None, None, None
    
    config = model_configs[model_name]
    net = config['net'].to(DEVICE)
    return net, config['transform'], config['size']

def load_local_model_info():
    """Load model information from local storage"""
    models = []
    if not os.path.exists(TRAINED_MODELS_DIR):
        return models
    
    # Look for .pth files
    model_files = glob.glob(os.path.join(TRAINED_MODELS_DIR, "*.pth"))
    
    for model_file in model_files:
        # Try to extract model info from filename or companion info file
        basename = os.path.basename(model_file).replace('.pth', '')
        info_file = os.path.join(TRAINED_MODELS_DIR, f"{basename}_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    model_info['model_path'] = model_file
                    models.append(model_info)
            except:
                # Fallback: extract info from filename
                models.append(extract_info_from_filename(basename, model_file))
        else:
            # Extract info from filename
            models.append(extract_info_from_filename(basename, model_file))
    
    return models

def extract_info_from_filename(filename, model_path):
    """Extract model info from filename"""
    parts = filename.split('_')
    
    # Try to identify model name and mode
    model_name = "Unknown"
    mode = "rgb"
    
    for model in AVAILABLE_MODELS:
        if model.lower() in filename.lower():
            model_name = model
            break
    
    if 'rgb' in filename.lower():
        mode = 'rgb'
    elif 'ir' in filename.lower():
        mode = 'ir'
    elif 'both' in filename.lower():
        mode = 'both'
    
    return {
        'id': filename,
        'model_name': model_name,
        'mode': mode,
        'classes_num': 3,
        'model_path': model_path,
        'source': 'local'
    }

def save_model_info(model_info, model_path):
    """Save model information locally"""
    basename = os.path.basename(model_path).replace('.pth', '')
    info_file = os.path.join(TRAINED_MODELS_DIR, f"{basename}_info.json")
    
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)

def predict_with_local_model(model_info, rgb_image=None, ir_image=None):
    """Make prediction using local model"""
    if not MODELS_AVAILABLE:
        raise Exception("Model definitions not available for offline prediction")
    
    try:
        DEVICE = get_device()
        
        # Load model
        net, transform_flag, target_size = get_model_instance(
            model_info['model_name'], 
            model_info.get('classes_num', 3)
        )
        
        if net is None:
            raise Exception(f"Model {model_info['model_name']} not supported")
        
        # Load state dict
        state_dict = torch.load(model_info['model_path'], map_location=DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        
        # Prepare transforms
        if transform_flag:
            transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor()
            ])
        
        # Process images
        rgb_tensor = None
        ir_tensor = None
        
        if rgb_image:
            rgb_tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)
        
        if ir_image:
            ir_tensor = transform(ir_image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            if model_info['model_name'] in MODEL_CUSTOM_LIST:
                mode = model_info['mode']
                if mode == 'both':
                    output = net(rgb_tensor, ir_tensor, mode=mode)
                elif mode == 'rgb':
                    output = net(rgb_tensor, rgb_tensor, mode=mode)
                else:  # ir
                    output = net(ir_tensor, ir_tensor, mode=mode)
            else:
                if model_info['mode'] == 'rgb':
                    output = net(rgb_tensor)
                else:  # ir
                    output = net(ir_tensor)
            
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class to label
        class_labels = {0: "No Fire", 1: "Fire", 2: "Smoke"}
        predicted_label = class_labels.get(predicted_class, f"Class {predicted_class}")
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                class_labels[i]: float(probabilities[0][i]) 
                for i in range(len(probabilities[0]))
            }
        }
        
    except Exception as e:
        raise Exception(f"Local prediction failed: {str(e)}")

# API Functions (for online mode)
def safe_api_call(func, *args, **kwargs):
    """Safely make API calls with error handling"""
    try:
        return func(*args, **kwargs)
    except requests.exceptions.ConnectionError:
        st.session_state.offline_mode = True
        return None
    except requests.exceptions.Timeout:
        st.error("❌ API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None

def get_available_models_api():
    """Get available models from API"""
    def _get_models():
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()["models"]
        return []
    
    result = safe_api_call(_get_models)
    return result if result is not None else []

def get_trained_models_api():
    """Get trained models from API"""
    def _get_models():
        response = requests.get(f"{API_BASE_URL}/trained-models", timeout=10)
        if response.status_code == 200:
            return response.json()["models"]
        return []
    
    result = safe_api_call(_get_models)
    return result if result is not None else []

def predict_image_api(model_id, rgb_image=None, ir_image=None):
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

def start_training_api(model_name, mode, epochs, batch_size, learning_rate):
    """Start training via API"""
    def _start_training():
        data = {
            "model_name": model_name,
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        response = requests.post(f"{API_BASE_URL}/train", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Training failed to start: {response.text}")
            return None
    
    return safe_api_call(_start_training)

def get_training_status_api(task_id):
    """Get training status via API"""
    def _get_status():
        response = requests.get(f"{API_BASE_URL}/training-status/{task_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    
    return safe_api_call(_get_status)

def delete_model_api(model_id):
    """Delete model via API"""
    def _delete_model():
        response = requests.delete(f"{API_BASE_URL}/models/{model_id}", timeout=10)
        return response.status_code == 200
    
    return safe_api_call(_delete_model)

# Main App Layout
st.markdown('<h1 class="main-header">🔥 Fire Detection System</h1>', unsafe_allow_html=True)

# Check connection and set mode
api_connected = test_api_connection()
st.session_state.offline_mode = not api_connected

# Connection status and mode indicator
if api_connected:
    st.markdown("""
    <div class="online-mode">
        🟢 <strong>Online Mode</strong> - Connected to API server. Full functionality available.
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.success("🟢 API Connected")
else:
    st.markdown("""
    <div class="offline-mode">
        🟡 <strong>Offline Mode</strong> - API server not available. Using local models for predictions.
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.warning("🟡 Offline Mode")
    if not MODELS_AVAILABLE:
        st.error("❌ Model definitions not found. Cannot use offline mode.")

# Sidebar Navigation
st.sidebar.title("Navigation")
if st.session_state.offline_mode:
    available_pages = ["🏠 Home", "🔍 Predict (Offline)", "📊 Local Models"]
else:
    available_pages = ["🏠 Home", "🚀 Train Model", "🔍 Predict", "📊 Model Management"]

page = st.sidebar.selectbox("Choose a page", available_pages)

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
            <h3>⚡ Offline Ready</h3>
            <p>Works with or without server connection</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.offline_mode:
        st.markdown("""
        ### 🔧 Offline Mode Features
        - **Local Model Loading**: Use previously trained models stored locally
        - **Offline Predictions**: Make predictions without server connection
        - **Model Management**: View and manage local model files
        
        ### 📁 Local Models Directory
        Make sure your trained models are stored in the `trained_models/` directory.
        """)
        
        # Show local models count
        local_models = load_local_model_info()
        st.info(f"📊 Found {len(local_models)} local models")
        
    else:
        st.markdown("""
        ### 🎯 Online Mode Features
        - **Train Custom Models**: Select from multiple architectures and train on your data
        - **Real-time Monitoring**: Track training progress with live updates
        - **Multi-Modal Prediction**: Support for RGB, thermal (IR), and combined inputs
        - **Model Management**: Manage and compare trained models
        - **Easy Deployment**: RESTful API for integration
        
        ### 🚀 Get Started
        1. Navigate to **Train Model** to start training a new model
        2. Use **Predict** to test your trained models
        3. Manage your models in **Model Management**
        """)

# Training Page (Online Only)
elif page == "🚀 Train Model" and not st.session_state.offline_mode:
    st.markdown('<h2 class="sub-header">Train New Model</h2>', unsafe_allow_html=True)
    
    # Get available models
    available_models = get_available_models_api()
    
    if not available_models:
        st.error("❌ Could not retrieve available models from API.")
        st.stop()
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Configuration")
        model_name = st.selectbox("Select Model Architecture", available_models)
        mode = st.selectbox("Input Mode", ["rgb", "ir", "both"])
        
        st.info(f"Selected: **{model_name}** with **{mode.upper()}** input mode")
    
    with col2:
        st.markdown("### ⚙️ Training Parameters")
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.0001, 0.00001], index=1)
    
    # Start training
    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("🚀 Starting training..."):
            result = start_training_api(model_name, mode, epochs, batch_size, learning_rate)
            
            if result:
                st.success(f"✅ Training started successfully! Task ID: {result['task_id']}")
                
                # Add to session state for monitoring
                st.session_state.training_tasks.append({
                    'task_id': result['task_id'],
                    'model_name': model_name,
                    'mode': mode,
                    'epochs': epochs,
                    'start_time': time.time()
                })
                
                st.info("🔄 You can monitor training progress below or refresh the page.")
    
    # Training Monitoring
    if st.session_state.training_tasks:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🔄 Training Progress</h3>', unsafe_allow_html=True)
        
        for i, task in enumerate(st.session_state.training_tasks):
            with st.container():
                st.markdown(f"""
                <div class="training-card">
                    <h4>Task {i+1}: {task['model_name']} ({task['mode']})</h4>
                    <p><strong>Task ID:</strong> {task['task_id']}</p>
                    <p><strong>Epochs:</strong> {task['epochs']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get status
                status = get_training_status_api(task['task_id'])
                
                if status:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Status", status.get('status', 'Unknown'))
                    
                    with col2:
                        if 'current_epoch' in status:
                            st.metric("Current Epoch", f"{status['current_epoch']}/{task['epochs']}")
                    
                    with col3:
                        if 'accuracy' in status:
                            st.metric("Accuracy", f"{status['accuracy']:.2%}")
                    
                    # Progress bar
                    if status.get('status') == 'training' and 'current_epoch' in status:
                        progress = status['current_epoch'] / task['epochs']
                        st.progress(progress)
                    
                    # Training completed
                    if status.get('status') == 'completed':
                        st.success("✅ Training completed successfully!")
                        if st.button(f"Remove Task {i+1}", key=f"remove_{i}"):
                            st.session_state.training_tasks.pop(i)
                            st.rerun()
                    
                    # Training failed
                    elif status.get('status') == 'failed':
                        st.error("❌ Training failed!")
                        st.error(f"Error: {status.get('error', 'Unknown error')}")
                        if st.button(f"Remove Task {i+1}", key=f"remove_failed_{i}"):
                            st.session_state.training_tasks.pop(i)
                            st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Auto-refresh option
        if st.checkbox("🔄 Auto-refresh (every 10 seconds)"):
            time.sleep(10)
            st.rerun()

# Predict Page (works both online and offline)
elif "🔍 Predict" in page:
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    # Get trained models (online or offline)
    if st.session_state.offline_mode:
        trained_models = load_local_model_info()
        st.info("🔧 Using local models for prediction")
    else:
        trained_models = get_trained_models_api()
        st.info("🌐 Using API models for prediction")
    
    if not trained_models:
        if st.session_state.offline_mode:
            st.warning("⚠️ No local models found. Please ensure trained models are in the 'trained_models/' directory.")
            st.markdown("""
            ### 📁 Expected Directory Structure:
            ```
            trained_models/
            ├── model1.pth
            ├── model1_info.json  (optional)
            ├── model2.pth
            └── model2_info.json  (optional)
            ```
            """)
        else:
            st.warning("⚠️ No trained models available. Please train a model first.")
        
        if st.button("🔄 Refresh Models"):
            st.rerun()
        st.stop()
    
    # Model selection
    model_options = {f"{model['model_name']} ({model['mode']}) - {model.get('source', 'API')}": model for model in trained_models}
    selected_model_name = st.selectbox("Select Trained Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    mode = selected_model['mode']
    st.info(f"📋 Selected model mode: **{mode.upper()}** | Source: **{selected_model.get('source', 'API').upper()}**")
    
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
                    result = None
                    
                    if st.session_state.offline_mode or selected_model.get('source') == 'local':
                        # Use local prediction
                        result = predict_with_local_model(selected_model, rgb_image, ir_image)
                    else:
                        # Use API prediction
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
                        
                        result = predict_image_api(selected_model['id'], rgb_data, ir_data)
                    
                    if result:
                        # Display results
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
                    if st.session_state.offline_mode:
                        st.error("Make sure model files are accessible and model definitions are available.")

# Model Management (Online) / Local Models (Offline)
elif page in ["📊 Model Management", "📊 Local Models"]:
    if st.session_state.offline_mode:
        st.markdown('<h2 class="sub-header">Local Model Management</h2>', unsafe_allow_html=True)
        
        # Get local models
        local_models = load_local_model_info()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📈 Local Models ({len(local_models)} found)")
        with col2:
            if st.button("🔄 Refresh Models"):
                st.rerun()
        
        if not local_models:
            st.info("📭 No local models found in 'trained_models/' directory.")
            st.markdown("""
            ### 💡 Tips:
            - Place your .pth model files in the `trained_models/` directory
            - Optionally create `modelname_info.json` files with model metadata
            - Model info will be extracted from filenames if no info file exists
            """)
            st.stop()
        
        # Display local models
        for i, model in enumerate(local_models):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{model['model_name']}**")
                    st.write(f"Mode: {model['mode'].upper()}")
                    st.write(f"Classes: {model['classes_num']}")
                    st.write(f"File: {os.path.basename(model['model_path'])}")
                
                with col2:
                    st.write("**Model ID**")
                    st.code(model['id'][:12] + "..." if len(model['id']) > 12 else model['id'])
                
                with col3:
                    # Model architecture info
                    if 'two_stream' in model['model_name'].lower():
                        st.write("🔄 **Two-Stream**")
                    else:
                        st.write("➡️ **Single-Stream**")
                    
                    # File size
                    try:
                        file_size = os.path.getsize(model['model_path']) / (1024 * 1024)
                        st.write(f"📁 {file_size:.1f} MB")
                    except:
                        st.write("📁 Size unknown")
                
                with col4:
                    if st.button(f"🗑️ Delete", key=f"delete_local_{i}", help="Delete this local model"):
                        try:
                            os.remove(model['model_path'])
                            # Also remove info file if exists
                            info_file = model['model_path'].replace('.pth', '_info.json')
                            if os.path.exists(info_file):
                                os.remove(info_file)
                            st.success("✅ Model deleted successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting model: {str(e)}")
                
                st.markdown("---")
        
        # Local model statistics
        if len(local_models) > 0:
            st.markdown("### 📊 Local Model Statistics")
            
            try:
                model_df = pd.DataFrame(local_models)
                
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
    
    else:
        # Online Model Management
        st.markdown('<h2 class="sub-header">Model Management</h2>', unsafe_allow_html=True)
        
        # Get trained models
        trained_models = get_trained_models_api()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📈 Trained Models ({len(trained_models)} available)")
        with col2:
            if st.button("🔄 Refresh Models"):
                st.rerun()
        
        if not trained_models:
            st.info("📭 No trained models available. Train a model first!")
            st.stop()
        
        # Display trained models
        for i, model in enumerate(trained_models):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{model['model_name']}**")
                    st.write(f"Mode: {model['mode'].upper()}")
                    st.write(f"Classes: {model.get('classes_num', 3)}")
                    if 'created_at' in model:
                        st.write(f"Created: {model['created_at']}")
                
                with col2:
                    st.write("**Model ID**")
                    st.code(model['id'][:12] + "..." if len(model['id']) > 12 else model['id'])
                
                with col3:
                    # Training metrics
                    if 'accuracy' in model:
                        st.metric("Accuracy", f"{model['accuracy']:.2%}")
                    if 'epochs' in model:
                        st.write(f"Epochs: {model['epochs']}")
                    
                    # Model size
                    if 'model_size' in model:
                        st.write(f"📁 {model['model_size']:.1f} MB")
                
                with col4:
                    # Download model (if supported by API)
                    if st.button(f"⬇️ Download", key=f"download_{i}", help="Download model file"):
                        st.info("Download functionality would be implemented here")
                    
                    # Delete model
                    if st.button(f"🗑️ Delete", key=f"delete_{i}", help="Delete this model"):
                        with st.spinner("Deleting model..."):
                            if delete_model_api(model['id']):
                                st.success("✅ Model deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete model")
                
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
                
                # Performance comparison (if accuracy data available)
                if 'accuracy' in model_df.columns and not model_df['accuracy'].isna().all():
                    st.markdown("### 🎯 Model Performance Comparison")
                    
                    fig_perf = px.bar(
                        model_df,
                        x='model_name',
                        y='accuracy',
                        color='mode',
                        title="Model Accuracy Comparison",
                        labels={'accuracy': 'Accuracy', 'model_name': 'Model'}
                    )
                    fig_perf.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error generating statistics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🔥 Fire Detection System | Built with Streamlit and PyTorch
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ System Info")
if torch.cuda.is_available():
    st.sidebar.success(f"🚀 GPU Available: {torch.cuda.get_device_name()}")
else:
    st.sidebar.info("💻 Using CPU")

st.sidebar.markdown(f"### 📊 Session Stats")
st.sidebar.info(f"Training Tasks: {len(st.session_state.training_tasks)}")

if st.session_state.offline_mode:
    local_models_count = len(load_local_model_info())
    st.sidebar.info(f"Local Models: {local_models_count}")

# Debug mode (optional)
if st.sidebar.checkbox("🔧 Debug Mode"):
    st.sidebar.markdown("### 🐛 Debug Info")
    st.sidebar.json({
        "API Connected": api_connected,
        "Offline Mode": st.session_state.offline_mode,
        "Models Available": MODELS_AVAILABLE,
        "Device": str(get_device()),
        "Training Tasks": len(st.session_state.training_tasks)
    })