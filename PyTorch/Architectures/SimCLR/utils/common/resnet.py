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


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
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


class ResNet(nn.Module):  # layers: list: [3, 4,, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        """function to make ResNet layers."""

        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4)
                                                )
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=2):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=2):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=2):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)


def test():
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    return y.shape


if __name__ == "__main__":
    print(test())


