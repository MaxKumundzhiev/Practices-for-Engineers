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

from main.model import ColorNet
from main.utils import Metrics, to_rgb
from main.dataloader import GrayscaleImageFolder


def validate(validation_model, model, criterion, save_images,epoch):
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
            save_path = {'grayscale': '../outputs/gray/', 'colorized': '../outputs/color/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
            to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

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


def train(train_loader, model, criterion, optimizer, epoch):
    import os
    print(f'Starting training epoch {epoch}')
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = Metrics(), Metrics(), Metrics()

    end = time.time ()
    for i, (input_gray, input_ab, target) in enumerate (train_loader):

        # Use GPU if available
        if use_gpu:
            input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format (
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))


if __name__ == "__main__":
    import os
    # Make folders and set parameters
    os.makedirs('../outputs/color', exist_ok=True)
    os.makedirs('../outputs/gray', exist_ok=True)
    os.makedirs('../checkpoints', exist_ok=True)

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()

    model = ColorNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Move model and loss function to GPU
    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    # Train transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    train_image_folder = GrayscaleImageFolder('../images/train', train_transforms)
    train_loader = DataLoader(train_image_folder, batch_size=64, shuffle=True)

    # Validation transformation
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    val_image_folder = GrayscaleImageFolder('../images/val', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_image_folder, batch_size=64, shuffle=False)

    save_images = True
    best_losses = 1e10
    epochs = 50

    # Train model
    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch)

        with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch)

        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), f'checkpoints/model-epoch-{epoch+1}-losses-{losses:.3f}.pth')

