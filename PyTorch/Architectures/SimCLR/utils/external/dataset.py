# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# GENERAL OVERVIEW OF DATASET
# -------------------------------------------

from __future__ import print_function, division

from settings import PATHS

import os
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


landmarks_frame = pd.read_csv(PATHS['CSV_PATH'])

n = 43  # image number by its index
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)

plt.show()