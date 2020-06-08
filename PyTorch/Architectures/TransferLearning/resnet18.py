import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


#defien device
devicec = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define mean and std
mean = np.array([.485, .456, .406])
std = np.array([.229, .224, .225])




