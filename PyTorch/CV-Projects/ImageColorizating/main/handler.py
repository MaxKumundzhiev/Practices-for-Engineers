# ------------------------------------------
#
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import time

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision.models as models

from main.model import ColorNet
from main.utils import Metrics
from main.dataloader import GrayscaleImageFolder

# Check if GPU is available
use_gpu = torch.cuda.is_available()

model = ColorNet()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

# Train transformation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip()
])
train_image_folder = GrayscaleImageFolder('images/train', train_transforms)
train_loader = DataLoader(train_image_folder, batch_size=64, shuffle=True)

# Validation transformation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
val_image_folder = GrayscaleImageFolder('images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_image_folder, batch_size=64, shuffle=False)


# Validation
def validate(validation_model, model, criterion, save_images,epoch):
    def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from skimage.color import lab2rgb

        plt.clf()  # clear matplotlib
        color_image = torch.cat ((grayscale_input, ab_input), 0).numpy ()  # combine channels
        color_image = color_image.transpose ((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze ().numpy ()
        if save_path is not None and save_name is not None:
            plt.imsave (arr=grayscale_input, fname='{}{}'.format (save_path['grayscale'], save_name), cmap='gray')
            plt.imsave (arr=color_image, fname='{}{}'.format (save_path['colorized'], save_name))

    model.eval()
    batch_time, data_time, losses = Metrics(), Metrics(), Metrics()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

    # Use GPU
    if use_gpu:
        input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_ab = model(input_gray)
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
        already_saved_images = True
        for j in range (min (len (output_ab), 10)):  # save at most 5 images
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
            save_name = 'img-{}-epoch-{}.jpg'.format (i * val_loader.batch_size + j, epoch)
            to_rgb(input_gray[j].cpu (), ab_input=output_ab[j].detach ().cpu (), save_path=save_path,
                    save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # report accuracy
    if i % 25 == 0:
        print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg



