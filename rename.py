

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

from torchvision import datasets
from torch.utils.data import DataLoader

'''
rename the image, make the piared images have the same name in the two folder

'''
path = 'E:/data'+'/254pRGBImages/'
files = os.listdir(path)
for i, file in enumerate(files):
    OldFileName = os.path.join(path, file)
    ID =  re.findall(r"\d+",OldFileName[-10:])[0]
    # print(ID)
    NewFileName = os.path.join(path, str(ID)+'.jpg')
    os.rename(OldFileName, NewFileName)



path = 'E:/data'+'/254pThermalImages/'
files = os.listdir(path)
for i, file in enumerate(files):
    OldFileName = os.path.join(path, file)
    ID =  re.findall(r"\d+",OldFileName[-10:])[0]
    # print(ID)
    NewFileName = os.path.join(path, str(ID)+'.jpg')
    os.rename(OldFileName, NewFileName)