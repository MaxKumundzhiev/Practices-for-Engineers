# ------------------------------------------
#
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import torch
import torch.nn as nn
from  torch.utils.data import DataLoader
from torchvision import transforms


from main.model import ColorNet
from main.dataloader import GrayscaleImageFolder

# Check if GPU is available
use_gpu = torch.cuda.is_available()

model = ColorNet()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

# Train
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip()
])
train_image_folder = GrayscaleImageFolder('images/train', train_transforms)
train_loader = DataLoader(train_image_folder, batch_size=64, shuffle=True)

# Validation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
val_image_folder = GrayscaleImageFolder('images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_image_folder, batch_size=64, shuffle=False)
