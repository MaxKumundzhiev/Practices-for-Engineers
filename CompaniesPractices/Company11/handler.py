"""
Create follwoing Neural Net:
Input size: 4 + 52 = 56
Ouput: 3 (softmax)  
Hidden Layers: 2
Initialize weight with random numbers
"""

import torch 
import torch.nn as nn
import torch.functional as F


class SampeNet(nn.Module):
    def __init__(self):
        super(SampeNet).__init__()
        


