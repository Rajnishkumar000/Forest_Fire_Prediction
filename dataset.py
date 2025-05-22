

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


# import numpy as np

# import re
# import torchvision.transforms.functional as TF
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image
# import random
# import torch
# import numpy as np
# from pathlib import Path
# from torchvision import datasets
# from torch.utils.data import DataLoader

# '''
# rename the image, make the piared images have the same name in the two folder

# '''

    
# '''
# prepare data for pytorch

# THREE CLASESS:
#     0:NN
#     1:YY
#     2:YN

# '''


# class MyDataset(Dataset):
#     def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
#         self.path_rgb = path_rgb
#         self.path_noise = path_ir
#         self.angle_array = [90, -90, 180, -180, 270, -270]
#         # self.target_size = target_size
#         self.transform = transform
#         self.pil2tensor = transforms.ToTensor()
    
#         self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
#         self.T = transforms.Compose([
#         # transforms.RandomResizedCrop(input_size),
#         transforms.Resize(input_size),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     def __getitem__(self, index):
#         name = os.listdir(self.path_rgb)[index]
#         ID = re.findall(r"\d+",name)[0]  
#         rgb = Image.open(os.path.join(self.path_rgb, name))
#         ir = Image.open(os.path.join(self.path_noise , name))
#         ID = int(ID)
        
      
        
#         if (1<= ID and ID <=13700):
#             y = 0
#         elif   (13701	<= ID and ID <=14699) \
#             or (15981	<= ID and ID <=19802) \
#             or (19900	<= ID and ID <=27183) \
#             or (27515	<= ID and ID <=31294) \
#             or (31510	<= ID and ID <=33597) \
#             or (33930	<= ID and ID <=36550) \
#             or (38031	<= ID and ID <=38153) \
#             or (41642	<= ID and ID <=45279) \
#             or (51207	<= ID and ID <=52286):
                
#             y = 1
#         else:
#             y=2
        
            
#         rgb = self.pil2tensor(rgb)
#         ir = self.pil2tensor(ir)
        
#         if self.transform is True:
            
#             rgb = self.T (rgb)
#             ir  = self.T (ir)
        
#         return rgb, ir,y
    
#     def __len__(self):
#         return len(os.listdir(self.path_rgb))
    
    
    

# class MyDataset_train(Dataset):  # train for cross dataset validation
#     def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
#         self.path_rgb = path_rgb
#         self.path_noise = path_ir
#         self.angle_array = [90, -90, 180, -180, 270, -270]
#         # self.target_size = target_size
#         self.transform = transform
#         self.pil2tensor = transforms.ToTensor()
    
#         self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
#         self.T = transforms.Compose([
#         # transforms.RandomResizedCrop(input_size),
#         transforms.Resize(input_size),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     def __getitem__(self, index):
#         name = os.listdir(self.path_rgb)[index]
#         ID = re.findall(r"\d+",name)[0]  
#         rgb = Image.open(os.path.join(self.path_rgb, name))
#         ir = Image.open(os.path.join(self.path_noise , name))
#         ID = int(ID)
        
#         if (1<= ID and ID <=13700):
#             y = 0
       
#         else:
#             y=1
        
        
   
            
#         rgb = self.pil2tensor(rgb)
#         ir = self.pil2tensor(ir)
        
#         if self.transform is True:
            
#             rgb = self.T (rgb)
#             ir  = self.T (ir)
        
#         return rgb, ir,y
    
#     def __len__(self):
#         return len(os.listdir(self.path_rgb))
    


# def MyDataset_test(path_test,input_size=254, transform=False):
    
    
#     pil2tensor = transforms.ToTensor()
#     T = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.RandomResizedCrop(input_size),
#         transforms.Resize(input_size),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     if transform is True:
#         test_dataset = datasets.ImageFolder(path_test,T)
#     else:
#         test_dataset = datasets.ImageFolder(path_test,pil2tensor)
#     return test_dataset





















































import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import re
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, path_rgb, path_ir, input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_ir = path_ir  # Fixed variable name from path_noise to path_ir for clarity
        self.angle_array = [90, -90, 180, -180, 270, -270]
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
        self.T = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index):
        rgb_files = os.listdir(self.path_rgb)
        ir_files=os.listdir(self.path_ir)
        if not rgb_files:
            raise ValueError(f"No files found in RGB directory: {self.path_rgb}")
            
        name = rgb_files[index]
        name1=ir_files[index]
        
        # Check if file exists in both directories
        rgb_path = os.path.join(self.path_rgb, name)
        ir_path = os.path.join(self.path_ir, name1)
        
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR image not found: {ir_path}")
        
        # Extract ID
        id_match = re.findall(r"\d+", name)
        if not id_match:
            raise ValueError(f"Could not extract ID from filename: {name}")
        
        ID = int(id_match[0])
        
        # Open images
        try:
            rgb = Image.open(rgb_path)
            ir = Image.open(ir_path)
        except Exception as e:
            raise IOError(f"Error opening images: {e}")
        
        # Assign class label
        if (1 <= ID and ID <= 13700):
            y = 0
        elif ((13701 <= ID and ID <= 14699) or
              (15981 <= ID and ID <= 19802) or
              (19900 <= ID and ID <= 27183) or
              (27515 <= ID and ID <= 31294) or
              (31510 <= ID and ID <= 33597) or
              (33930 <= ID and ID <= 36550) or
              (38031 <= ID and ID <= 38153) or
              (41642 <= ID and ID <= 45279) or
              (51207 <= ID and ID <= 52286)):
            y = 1
        else:
            y = 2
        
        # Convert to tensor
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        # Apply transforms if needed
        if self.transform is True:
            rgb = self.T(rgb)
            ir = self.T(ir)
        
        return rgb, ir, y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))
    
class MyDataset_train(Dataset):  # train for cross dataset validation
    def __init__(self, path_rgb, path_ir, input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_ir = path_ir  # Fixed variable name from path_noise to path_ir for clarity
        self.angle_array = [90, -90, 180, -180, 270, -270]
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
        self.T = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index):
        rgb_files = os.listdir(self.path_rgb)
        if not rgb_files:
            raise ValueError(f"No files found in RGB directory: {self.path_rgb}")
            
        name = rgb_files[index]
        
        # Check if file exists in both directories
        rgb_path = os.path.join(self.path_rgb, name)
        ir_path = os.path.join(self.path_ir, name)
        
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR image not found: {ir_path}")
            
        ID = int(re.findall(r"\d+", name)[0])
        
        # Open images
        try:
            rgb = Image.open(rgb_path)
            ir = Image.open(ir_path)
        except Exception as e:
            raise IOError(f"Error opening images: {e}")
        
        # Binary classification for cross-dataset
        if (1 <= ID and ID <= 13700):
            y = 0
        else:
            y = 1
        
        # Convert to tensor
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        # Apply transforms if needed
        if self.transform is True:
            rgb = self.T(rgb)
            ir = self.T(ir)
        
        return rgb, ir, y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))

def MyDataset_test(path_test, input_size=254, transform=False):
    pil2tensor = transforms.ToTensor()
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if directory exists
    if not os.path.exists(path_test):
        raise FileNotFoundError(f"Test directory not found: {path_test}")
    
    if transform is True:
        test_dataset = datasets.ImageFolder(path_test, T)
    else:
        test_dataset = datasets.ImageFolder(path_test, pil2tensor)
        
    return test_dataset