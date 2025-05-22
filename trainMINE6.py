from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import json
import os
import uuid
from datetime import datetime
import asyncio
import threading
from models import (
    Logistic_two_stream, Flame_one_stream, VGG16, Vgg_two_stream, 
    Logistic, Flame_two_stream, Mobilenetv2, Mobilenetv2_two_stream,
    LeNet5_one_stream, LeNet5_two_stream, Resnet18, Resnet18_two_stream
)
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

app = FastAPI(title="Fire Detection API", description="API for training and predicting fire detection models")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for training status
training_status = {}
trained_models = {}

class TrainingConfig(BaseModel):
    model_name: str
    mode: str  # 'rgb', 'ir', 'both'
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    classes_num: int = 3
    subset_rate: float = 0.01
    trainset_rate: float = 0.8
    path_rgb: str = "E:/data/254pRGBImages"
    path_ir: str = "E:/data/254pThermalImages"
    test_interval: int = 1

class PredictionRequest(BaseModel):
    model_name: str
    mode: str

# Available models - consistent with training script
AVAILABLE_MODELS = [
    "Flame_one_stream", "Flame_two_stream", "VGG16", "Vgg_two_stream",
    "Logistic", "Logistic_two_stream", "Mobilenetv2", "Mobilenetv2_two_stream",
    "LeNet5_one_stream", "LeNet5_two_stream", "Resnet18", "Resnet18_two_stream"
]

# Custom models that require special handling - exactly as in training script
MODEL_CUSTOM_LIST = [
    'Flame_one_stream', 'Flame_two_stream', 'Mobilenetv2_two_stream',
    'Vgg_two_stream', 'Logistic_two_stream', 'Resnet18_two_stream',
    'LeNet5_one_stream', 'LeNet5_two_stream'
]

def get_model_and_config(model_name: str, classes_num: int = 3):
    """Get model instance and configuration - matches training script exactly"""
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model configurations exactly matching the training script
    if model_name == 'Flame_one_stream':
        net = Flame_one_stream().to(DEVICE)
        transform_flag = False
        target_size = 254
    elif model_name == 'Flame_two_stream':
        net = Flame_two_stream().to(DEVICE)
        transform_flag = False
        target_size = 254
    elif model_name == 'VGG16':
        net = VGG16(classes_num).to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'Vgg_two_stream':
        net = Vgg_two_stream().to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'Logistic':
        net = Logistic(classes_num).to(DEVICE)
        transform_flag = False  # Fixed: was missing in original API
        target_size = 254
    elif model_name == 'Logistic_two_stream':
        net = Logistic_two_stream(classes_num).to(DEVICE)
        transform_flag = False  # Fixed: was missing in original API
        target_size = 254
    elif model_name == 'Mobilenetv2':
        net = Mobilenetv2(classes_num).to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'Mobilenetv2_two_stream':
        net = Mobilenetv2_two_stream().to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'Resnet18':
        net = Resnet18(classes_num).to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'Resnet18_two_stream':
        net = Resnet18_two_stream().to(DEVICE)
        transform_flag = True
        target_size = 224
    elif model_name == 'LeNet5_one_stream':
        net = LeNet5_one_stream().to(DEVICE)
        transform_flag = False
        target_size = 254
    elif model_name == 'LeNet5_two_stream':
        net = LeNet5_two_stream().to(DEVICE)
        transform_flag = False
        target_size = 254
    else:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")
    
    return net, transform_flag, target_size

def test_model(dataloader, net, DEVICE, MODE, Model_name, Model_custom_list):
    """Test function matching the training script"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        net.eval()
        for (rgb, ir, y) in dataloader:
            y = y.to(DEVICE)
            
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE), ir.to(DEVICE), mode=MODE)
            else:
                if MODE == 'rgb':
                    x = rgb.to(DEVICE)
                elif MODE == 'ir':
                    x = ir.to(DEVICE)
                y_pre = net(x)
            
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y).sum().item()
    
    return correct / total

def train_model_background(config: TrainingConfig, task_id: str):
    """Background training function - improved to match training script"""
    try:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Update status
        training_status[task_id]['status'] = 'initializing'
        training_status[task_id]['progress'] = 10
        
        # Get model and configuration
        net, transform_flag, target_size = get_model_and_config(config.model_name, config.classes_num)
        
        # Load dataset - exactly as in training script
        training_status[task_id]['status'] = 'loading_data'
        training_status[task_id]['progress'] = 20
        
        dataset = MyDataset(config.path_rgb, config.path_ir, input_size=target_size, transform=transform_flag)
        dataset, _ = torch.utils.data.random_split(
            dataset, 
            [int(len(dataset) * config.subset_rate), len(dataset) - int(len(dataset) * config.subset_rate)]
        )
        
        train_set, val_set = torch.utils.data.random_split(
            dataset, 
            [int(len(dataset) * config.trainset_rate), len(dataset) - int(len(dataset) * config.trainset_rate)]
        )
        
        train_dataloader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)
        
        # Setup training - exactly as in training script
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=0.00)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.2)
        
        training_status[task_id]['status'] = 'training'
        training_status[task_id]['progress'] = 30
        
        # Training loop - exactly as in training script
        net.train()
        total_step = len(train_dataloader)
        max_accuracy = 0
        correct_on_test = []
        correct_on_train = []
        
        for epoch in range(config.epochs):
            net.train()
            epoch_loss = 0
            
            for i, (rgb, ir, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                if config.model_name in MODEL_CUSTOM_LIST:
                    y_pre = net(rgb.to(DEVICE), ir.to(DEVICE), mode=config.mode)
                else:
                    if config.mode == 'rgb':
                        x = rgb.to(DEVICE)
                    elif config.mode == 'ir':
                        x = ir.to(DEVICE)
                    y_pre = net(x)
                
                loss = loss_function(y_pre, y.to(DEVICE))
                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                # Update progress
                progress = 30 + (epoch * 60 / config.epochs) + (i * 60 / (config.epochs * total_step))
                training_status[task_id]['progress'] = int(progress)
                training_status[task_id]['current_loss'] = loss.item()
                training_status[task_id]['epoch'] = epoch + 1
                training_status[task_id]['step'] = i + 1
                training_status[task_id]['total_steps'] = total_step
            
            # Test at intervals - as in training script
            if ((epoch + 1) % config.test_interval) == 0:
                current_accuracy = test_model(test_dataloader, net, DEVICE, config.mode, 
                                            config.model_name, MODEL_CUSTOM_LIST)
                current_accuracy_train = test_model(train_dataloader, net, DEVICE, config.mode, 
                                                  config.model_name, MODEL_CUSTOM_LIST)
                
                correct_on_test.append(round(current_accuracy * 100, 2))
                correct_on_train.append(round(current_accuracy_train * 100, 2))
                
                if current_accuracy > max_accuracy:
                    max_accuracy = current_accuracy
                
                training_status[task_id]['test_accuracy'] = correct_on_test[-1]
                training_status[task_id]['train_accuracy'] = correct_on_train[-1]
                training_status[task_id]['max_test_accuracy'] = max(correct_on_test)
        
        # Save trained model
        model_path = f"trained_models/{config.model_name}_{config.mode}_{task_id}.pth"
        os.makedirs("trained_models", exist_ok=True)
        torch.save(net.state_dict(), model_path)
        
        # Store model info
        trained_models[task_id] = {
            'model_name': config.model_name,
            'mode': config.mode,
            'model_path': model_path,
            'classes_num': config.classes_num,
            'target_size': target_size,
            'transform_flag': transform_flag,
            'max_accuracy': max_accuracy
        }
        
        training_status[task_id]['status'] = 'completed'
        training_status[task_id]['progress'] = 100
        training_status[task_id]['model_path'] = model_path
        training_status[task_id]['final_accuracy'] = max_accuracy
        
    except Exception as e:
        training_status[task_id]['status'] = 'failed'
        training_status[task_id]['error'] = str(e)

def get_transforms(transform_flag: bool, target_size: int):
    """Get transforms exactly as used in training"""
    if transform_flag:
        # For models that need ImageNet normalization
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # For custom models
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])

@app.get("/")
async def root():
    return {"message": "Fire Detection API is running"}

@app.get("/models")
async def get_available_models():
    return {"models": AVAILABLE_MODELS}

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    # Initialize training status
    training_status[task_id] = {
        'status': 'queued',
        'progress': 0,
        'started_at': datetime.now().isoformat(),
        'model_name': config.model_name,
        'mode': config.mode,
        'test_accuracy': 0,
        'train_accuracy': 0,
        'max_test_accuracy': 0
    }
    
    # Start background training
    background_tasks.add_task(train_model_background, config, task_id)
    
    return {"task_id": task_id, "message": "Training started"}

@app.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return training_status[task_id]

@app.post("/predict")
async def predict_image(
    model_id: str,
    rgb_image: Optional[UploadFile] = File(None),
    ir_image: Optional[UploadFile] = File(None)
):
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Trained model not found")
    
    model_info = trained_models[model_id]
    mode = model_info['mode']
    
    # Validate required images based on mode
    if mode == 'rgb' and not rgb_image:
        raise HTTPException(status_code=400, detail="RGB image required for RGB mode")
    if mode == 'ir' and not ir_image:
        raise HTTPException(status_code=400, detail="IR image required for IR mode")
    if mode == 'both' and (not rgb_image or not ir_image):
        raise HTTPException(status_code=400, detail="Both RGB and IR images required for both mode")
    
    try:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model - exactly as trained
        net, _, _ = get_model_and_config(model_info['model_name'], model_info['classes_num'])
        net.load_state_dict(torch.load(model_info['model_path'], map_location=DEVICE))
        net.eval()
        
        # Use the same transforms as during training
        transform = get_transforms(model_info['transform_flag'], model_info['target_size'])
        
        # Process images
        rgb_tensor = None
        ir_tensor = None
        
        if rgb_image:
            rgb_pil = Image.open(io.BytesIO(await rgb_image.read())).convert('RGB')
            rgb_tensor = transform(rgb_pil).unsqueeze(0).to(DEVICE)
        
        if ir_image:
            ir_pil = Image.open(io.BytesIO(await ir_image.read())).convert('RGB')
            ir_tensor = transform(ir_pil).unsqueeze(0).to(DEVICE)
        
        # Make prediction - exactly as in training/testing
        with torch.no_grad():
            if model_info['model_name'] in MODEL_CUSTOM_LIST:
                if mode == 'both':
                    output = net(rgb_tensor, ir_tensor, mode=mode)
                elif mode == 'rgb':
                    # For RGB mode, some models might still need both inputs
                    output = net(rgb_tensor, rgb_tensor, mode=mode)
                else:  # ir
                    output = net(ir_tensor, ir_tensor, mode=mode)
            else:
                if mode == 'rgb':
                    output = net(rgb_tensor)
                else:  # ir
                    output = net(ir_tensor)
            
            # Get predictions using the same method as test function
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output.data, dim=-1)
            predicted_class = predicted_class.item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class to label (0: no fire, 1: fire, 2: smoke)
        class_labels = {0: "No Fire", 1: "Fire", 2: "Smoke"}
        predicted_label = class_labels.get(predicted_class, f"Class {predicted_class}")
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                class_labels.get(i, f"Class {i}"): float(probabilities[0][i]) 
                for i in range(len(probabilities[0]))
            },
            "model_training_accuracy": model_info.get('max_accuracy', 'Unknown')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/trained-models")
async def get_trained_models():
    return {
        "models": [
            {
                "id": task_id,
                "model_name": info["model_name"],
                "mode": info["mode"],
                "classes_num": info["classes_num"],
                "max_accuracy": info.get("max_accuracy", "Unknown")
            }
            for task_id, info in trained_models.items()
        ]
    }

@app.delete("/trained-models/{model_id}")
async def delete_trained_model(model_id: str):
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[model_id]
    model_path = model_info['model_path']
    
    # Delete model file
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Remove from memory
    del trained_models[model_id]
    if model_id in training_status:
        del training_status[model_id]
    
    return {"message": "Model deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)