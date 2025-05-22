import torch
import torch.nn as nn
import torchvision.models as models
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.nn.functional as F

import datetime
import os
import random

from models import Logistic_two_stream, Flame_one_stream, VGG16, Vgg_two_stream, Logistic
from models import Flame_two_stream, Mobilenetv2, Mobilenetv2_two_stream, LeNet5_one_stream
from models import LeNet5_two_stream, Resnet18, Resnet18_two_stream

import json
import csv

from torchmetrics.functional import f1_score
import numpy as np
import sys


def test(dataloader, net, DEVICE, MODE, Model_name, name_index, log_path, correct_on_test, correct_on_train, Model_custom_list, flag='test_set', output_flag=False):
    
    correct = 0
    total = 0
    
    y_all_true = torch.tensor([]).to(DEVICE)
    y_all_pre = torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        net.eval()
        for (rgb, ir, y) in dataloader:
            y = y.to(DEVICE)
            y_all_true = torch.cat((y_all_true, y))
            
            if Model_name in Model_custom_list:
                y_pre = net(rgb.to(DEVICE), ir.to(DEVICE), mode=MODE)
            else:
                if MODE=='rgb':
                    x = rgb.to(DEVICE)
                elif MODE=='ir':
                    x = ir.to(DEVICE)
                    
                y_pre = net(x)
            
            _, label_index = torch.max(y_pre.data, dim=-1)
            y_all_pre = torch.cat((y_all_pre, label_index))
            
            total += label_index.shape[0]
            correct += (label_index == y).sum().item()
        
        acc = 100 * correct / total if total > 0 else 0
        if flag == 'test_set':
            correct_on_test.append(round(acc, 2))
        elif flag == 'train_set':
            correct_on_train.append(round(acc, 2))
        print(f'Accuracy on {flag}: {acc:.2f} %')
        
    if output_flag == True:
        try:
            # Fixed f1_score call by adding the 'task' parameter
            macro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), task="multiclass", num_classes=3, average='macro')
            micro_f1 = f1_score(y_all_pre.int(), y_all_true.int(), task="multiclass", average='micro')
            
            # Calculate precision and recall separately
            try:
                from torchmetrics.functional import precision, recall
                # Fixed precision and recall calls with task parameter
                macro_p = precision(y_all_pre.int(), y_all_true.int(), task="multiclass", average='macro', num_classes=3)
                macro_r = recall(y_all_pre.int(), y_all_true.int(), task="multiclass", average='macro', num_classes=3)
                micro_p = precision(y_all_pre.int(), y_all_true.int(), task="multiclass", average='micro')
                micro_r = recall(y_all_pre.int(), y_all_true.int(), task="multiclass", average='micro')
                
                with open(log_path, 'a+', newline='') as f:
                    csv_write = csv.writer(f)
                    data_row = [Model_name, MODE, name_index, macro_f1.item(), micro_f1.item(), 
                               macro_p.item(), macro_r.item(), micro_p.item(), micro_r.item()]
                    csv_write.writerow(data_row)
            except ImportError:
                # If precision/recall functions aren't available, just log the F1 scores
                with open(log_path, 'a+', newline='') as f:
                    csv_write = csv.writer(f)
                    data_row = [Model_name, MODE, name_index, macro_f1.item(), micro_f1.item()]
                    csv_write.writerow(data_row)
                print("Warning: precision/recall functions not available, logging only F1 scores")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Still return the accuracy even if metrics calculation fails
        
    return correct / total if total > 0 else 0


if __name__ == "__main__":
    # Simple configuration using variables instead of argparse
    path_rgb = 'E:/data/254pRGBImages'  # path to RGB images
    path_ir = 'E:/data/254pThermalImages'  # path to IR images
    
    # Model configuration
    name_index = 0
    BATCH_SIZE = 64
    LR = 1e-3
    classes_num = 3
    subset_rate = 0.01
    trainset_rate = 0.8
    
    # Change these variables to select different models/modes
    Model_name = 'Flame_one_stream'  # Options: VGG16, Flame_one_stream, etc.
    MODE = 'rgb'  # Options: rgb, ir, both
    
    # Training parameters
    EPOCH = 1
    test_interval = 1
    log_path = './log/results.csv'
    log_loss_path = './log/'
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(log_loss_path, exist_ok=True)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print(f'use device: {DEVICE}')
    
    Transform_flag = False # sometime use it to resize the image
    
    file_name = Model_name + '_' + MODE + '_' + str(name_index)
    
    # Define model custom list
    Model_custom_list = ['Flame_one_stream', 'Flame_two_stream', 'Mobilenetv2_two_stream', 
                        'Vgg_two_stream', 'Logistic_two_stream', 'Resnet18_two_stream', 
                        'LeNet5_one_stream', 'LeNet5_two_stream']
    
    # Model selection
    if Model_name == 'Flame_one_stream':
        net = Flame_one_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'Flame_two_stream':
        net = Flame_two_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'VGG16':
        net = VGG16(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Vgg_two_stream':
        Transform_flag = True
        target_size = 224
        net = Vgg_two_stream().to(DEVICE)
        
    elif Model_name == 'Logistic':
        net = Logistic(classes_num).to(DEVICE)
        target_size = 254
        
    elif Model_name == 'Logistic_two_stream':
        net = Logistic_two_stream(classes_num).to(DEVICE)
        target_size = 254
    
    elif Model_name == 'Mobilenetv2':
        net = Mobilenetv2(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Mobilenetv2_two_stream':
        net = Mobilenetv2_two_stream().to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Resnet18':
        net = Resnet18(classes_num).to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'Resnet18_two_stream':
        net = Resnet18_two_stream().to(DEVICE)
        Transform_flag = True
        target_size = 224
        
    elif Model_name == 'LeNet5_one_stream':
        net = LeNet5_one_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
        
    elif Model_name == 'LeNet5_two_stream':
        net = LeNet5_two_stream().to(DEVICE)
        Transform_flag = False
        target_size = 254
    
    print(net)
    
    # Path verification - check if directories exist
    if not os.path.exists(path_rgb):
        print(f"Error: RGB path does not exist: {path_rgb}")
        sys.exit(1)
    
    if not os.path.exists(path_ir):
        print(f"Error: IR path does not exist: {path_ir}")
        sys.exit(1)
    
    try:
        Dataset = MyDataset(path_rgb, path_ir, input_size=target_size, transform=Transform_flag)
        print(f"Successfully loaded dataset with {len(Dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    if len(Dataset) == 0:
        print("Error: Dataset is empty. Check if the directories contain images.")
        sys.exit(1)
    
    # Split into subset, train and validation sets
    try:
        # Set random seed for reproducibility
        random_seed = 42
        torch.manual_seed(random_seed)
        
        subset_size = int(len(Dataset) * subset_rate)
        if subset_size <= 0:
            print("Error: Subset size is zero. Increase subset_rate.")
            sys.exit(1)
        
        Dataset, _ = torch.utils.data.random_split(
            Dataset, 
            [subset_size, len(Dataset) - subset_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        split_rate = trainset_rate
        train_size = int(len(Dataset) * split_rate)
        val_size = len(Dataset) - train_size
        
        if train_size <= 0 or val_size <= 0:
            print("Error: Training or validation set size is zero. Check trainset_rate.")
            sys.exit(1)
            
        train_set, val_set = torch.utils.data.random_split(
            Dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False)
        
        print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")
        
        # Check class distribution in train and validation sets
        train_classes = {}
        val_classes = {}
        
        # Use a small batch iterator to check class distributions
        for rgb, ir, label in DataLoader(dataset=train_set, batch_size=1):
            label_val = label.item()
            train_classes[label_val] = train_classes.get(label_val, 0) + 1
            
        for rgb, ir, label in DataLoader(dataset=val_set, batch_size=1):
            label_val = label.item()
            val_classes[label_val] = val_classes.get(label_val, 0) + 1
            
        print("Class distribution in training set:", train_classes)
        print("Class distribution in validation set:", val_classes)
        
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        sys.exit(1)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.01)  # Increased weight decay
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)  # Reduced label smoothing
    
    correct_on_train = [0]
    correct_on_test = [0]
    
    total_step = (train_size // BATCH_SIZE) + (1 if train_size % BATCH_SIZE != 0 else 0)
    
    net.train()
    max_accuracy = 0
    Loss_accuracy = []
    test_acc = []
    loss_list = []
    train_acc = []
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    for index in range(EPOCH):
        net.train()
        epoch_loss = 0
        for i, (rgb, ir, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            try:
                if Model_name in Model_custom_list:
                    y_pre = net(rgb.to(DEVICE), ir.to(DEVICE), mode=MODE)
                else:
                    if MODE == 'rgb':
                        x = rgb.to(DEVICE)
                    elif MODE == 'ir':
                        x = ir.to(DEVICE)
                        
                    y_pre = net(x)
                
                loss = loss_function(y_pre, y.to(DEVICE))
                epoch_loss += loss.item()
                
                # Print status with clear formatting
                print(f'Epoch: {index + 1}/{EPOCH}  Step: {i+1:3d}/{total_step:3d}  Loss: {loss.item():.6f}', end='\r')
                
                loss_list.append(loss.item())
        
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
            except Exception as e:
                print(f"\nError during training step: {e}")
                continue
        
        # Print newline after epoch progress reporting
        print()
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {index+1} average loss: {avg_loss:.6f}")
        
        # Evaluate model performance
        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader, net, DEVICE, MODE, Model_name, name_index, 
                                   log_path, correct_on_test, correct_on_train, Model_custom_list, 
                                   flag='test_set', output_flag=False)
            test_acc.append(current_accuracy)
            
            current_accuracy_train = test(train_dataloader, net, DEVICE, MODE, Model_name, name_index, 
                                        log_path, correct_on_test, correct_on_train, Model_custom_list, 
                                        flag='train_set', output_flag=False)
            train_acc.append(current_accuracy_train)
            
            print(f'Current max accuracy\t test set: {max(correct_on_test):.2f}%\t train set: {max(correct_on_train):.2f}%')
    
            # Update learning rate based on validation accuracy
            scheduler.step(current_accuracy)
            
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                # Save best model
                torch.save(net.state_dict(), os.path.join(log_loss_path, f"best_{file_name}.pth"))
                print(f"Saved best model with accuracy {max_accuracy:.2f}%")
    
    # Final evaluation with detailed metrics
    final_acc = test(test_dataloader, net, DEVICE, MODE, Model_name, name_index, log_path, 
                   correct_on_test, correct_on_train, Model_custom_list, 
                   flag='test_set', output_flag=True)
    
    print(f"Final model accuracy: {final_acc*100:.2f}%")
    
    # Save training results
    results_array = np.zeros(3, dtype=object)
    results_array[0] = loss_list
    results_array[1] = train_acc
    results_array[2] = test_acc
    
    save_path = os.path.join(log_loss_path, file_name + '.npy')
    np.save(save_path, results_array)
    print(f"Results saved to {save_path}")
    
    # Save final model
    torch.save(net.state_dict(), os.path.join(log_loss_path, f"final_{file_name}.pth"))
    print(f"Final model saved to {os.path.join(log_loss_path, f'final_{file_name}.pth')}")
    
    sys.exit()