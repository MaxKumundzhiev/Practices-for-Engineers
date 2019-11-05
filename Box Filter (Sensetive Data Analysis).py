# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

#Imports
import requests as re
import bs4
import os
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from pathlib import Path
import logging
import timeit

import matplotlib.pyplot as plt



def open_img(path):
    return cv2.imread ('%s' % path, cv2.IMREAD_UNCHANGED)

# Get all the files from folder PNG
path = '/Users/macbook/Documents/I/University ELTE/3 Semester/Sense Data/PNG'
list_of_png = [f for f in listdir (path) if isfile (join (path, f))]



def _run_box_filter(gray_image, filter_size):
    start = timeit.default_timer ()
    input_image = gray_image
    height, width = gray_image.shape[:2]
    output_image = np.zeros ([height, width], dtype=np.uint8)
    border = int ((filter_size - 1) / 2)
    s = np.zeros (width)

    '''initialisation  '''
    for j in range (width):
        s[j] = gray_image[0:filter_size, j].sum ()

    for i in range (height - filter_size + 1):
        '''update s'''
        if i != 0:
            for j in range (width):
                s[j] = s[j] + gray_image[i + filter_size - 1][j] - gray_image[i - 1][j]

        new_value = s[0: filter_size].sum ()

        for j in range (width - filter_size + 1):
            if j != 0:
                new_value = new_value + s[j + filter_size - 1] - s[j - 1]

            output_image[i + border][j + border] = new_value / filter_size ** 2
    stop = timeit.default_timer ()

    f = plt.figure (figsize=(10, 12))
    f.add_subplot (1, 2, 1)
    plt.title ('Input Image --> Original Image')
    plt.imshow (input_image, cmap='gray')
    f.add_subplot (1, 2, 2)
    plt.title ('Outpit Image --> Filtered Image')
    plt.imshow (output_image, cmap='gray')
    plt.show (block=True)
    print ('FAST BOX FILTER', '\n', 'Input image matrix: ', '\n', input_image, '\n', 'Output image matrix: ', '\n', output_image, '\n', 'Time: ', stop - start)


# #Check for all occures (images)
# for image in list_of_png:
#     _run_box_filter(open_img(os.path.join(path, image)), 3)


def compute_window_mean_and_var(image, window_w, window_h):
    start = timeit.default_timer ()
    w, h = image.shape
    w_new, h_new = w - window_w + 1, h - window_h + 1
    means = np.zeros ([w_new, h_new])
    maximums = np.zeros ([w_new, h_new])
    variations = np.zeros ([w_new, h_new])
    for i in range (w_new):
        for j in range (h_new):
            window = image[i:i + window_w, j:j + window_h]
            means[i, j] = np.mean (window)
            maximums[i, j] = np.max (window)
            variations[i, j] = np.var (window)
    stop = timeit.default_timer ()
    f = plt.figure (figsize=(10, 12))
    f.add_subplot (1, 2, 1)
    plt.title ('Input Image --> Original Image')
    plt.imshow (image, cmap='gray')
    f.add_subplot (1, 2, 2)
    plt.title ('Outpit Image --> Filtered Image')
    plt.imshow (means, cmap='gray')
    plt.show (block=True)
    print ('SIMPLE BOX FILTER', '\n', 'Input image matrix: ', '\n', image, '\n', 'Output image matrix: ', '\n', means, '\n',  'Time: ', stop - start)
    return means


#Check for one occure --> Fast Box Filter
_run_box_filter(open_img(os.path.join(path, list_of_png[32])), 3)

#Check for one occure --> Simple Box Filter (Mean Filter)
compute_window_mean_and_var (open_img (os.path.join (path, list_of_png[6])), 3, 3)