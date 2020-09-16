import numpy as np
import os
import sys
import shutil 
from tqdm import tqdm

from PIL import Image

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

class_names = ['female', 'male']
batch_size  = 128
data_dir    = sys.argv[1]
model_file  = "face_model.pth"

class FaceDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames  = os.listdir(data_dir)
        self.data_dir   = data_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.filenames[idx])
        img      = Image.open(img_name)

        if self.transform:
            img = self.transform(img)
        return img, self.filenames[idx]

    def __len__(self):
        return len(self.filenames)
        
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

male_dataset    = FaceDataset(data_dir, train_transforms)
male_dataloader = DataLoader(male_dataset,   batch_size=batch_size, shuffle=False)

model = models.resnet18(num_classes=2)
model.load_state_dict(torch.load(model_file))
model.eval()

model.cuda()

img_names   = []
img_classes = []

for X, y in tqdm(male_dataloader):
    pred = model(X.cuda()).argmax(axis=1)
    img_classes.extend([class_names[i] for i in pred])
    img_names.extend(y)
    
result_dict = dict(zip(img_names, img_classes))

with open('process_results.json', 'w') as f:
    f.write(json.dumps(result_dict))