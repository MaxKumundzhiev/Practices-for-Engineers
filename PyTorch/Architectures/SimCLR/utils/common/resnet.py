# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


"""
An encoder network, E(·), which maps an augmented image x˜ to a representation vector, r = E(x˜) ∈ RDE . In our
framework, both augmented images for each input image are separately input to the same encoder, resulting in a pair
of representation vectors. We experiment with two commonly used encoder architectures, ResNet-50 and ResNet-200
[20], where the activations of the final pooling layer (DE = 2048) are used as the representation vector. This representation layer is always normalized to the unit hypersphere in RDE . We find from experiments that this normalization
always improves performance, consistent with other papers that have used metric losses e.g. [40]. We also find that
the new supervised loss is able to train both of these architectures to a high accuracy with no special hyperparameter
tuning. In fact, as reported in Sec. 4, we found that the supervised contrastive loss was less sensitive to small changes
in hyperparameters, such as choice of optimizer or data augmentation.
"""

import torch
import torch.nn as nn

## dont forget before passing data into net, apply ".unsqueeze_(1)"


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x






class ResNet50:
    def __init__(self):
        pass

    def forward(self):
        pass


