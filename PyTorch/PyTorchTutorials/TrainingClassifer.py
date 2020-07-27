# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


import torch
import torchvision
from torchvision.transforms import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='/Users/macbook/Documents/GitRep/PracticesForEngineers/Practices-for-Engineers/PyTorch/PyTorchTutorials/data',
                                        train=True,
                                        download=True,
                                        transform=transform
                                        )


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)