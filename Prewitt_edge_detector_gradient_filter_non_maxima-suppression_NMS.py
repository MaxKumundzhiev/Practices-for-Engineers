# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


# The process requires me to first find the gradient values of a image that has been converted to grayscale,
# this allows for each detection of ‘edge-like’ regions.
#
# We’ll then apply a Canny edge detection and some other blurring techniques to give us a much better chance of detecting
# the parts of the hand we want to be focusing on.

#Importing Libraries
from os.path import isfile, join
import os
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv




path = '/Users/macbook/Documents/I/University ELTE/3 Semester/Sense Data/PNG'
os.chdir(path)

list_of_png = [f for f in os.listdir (path) if isfile (join (path, f))]
print('Before removing', list_of_png)
list_of_png.remove('.DS_Store')
print('After removing', list_of_png)

# -------------------------------------------THEORY-------------------------------------------
# Useful link with different implementation approaches:
# https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/
# https://www.programcreek.com/python/example/89325/cv2.Sobel
#https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
#https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e --> read and use
#https://www.youtube.com/watch?v=55rW6LNU2KI --> look and compare with previous link

# There are a number of feature descriptors out there. Here are a few of the most popular ones:
#
# -  HOG: Histogram of Oriented Gradients
# -  SIFT: Scale Invariant Feature Transform
#  - SURF: Speeded-Up Robust Feature

#HOG - Histogram of Orientated(different angles) Gradient

# - For function f(x, y) the gradient is a vector (f_x, f_y)
# - An image is a discrete function of (x, y), so image gradient can be calculated as well
# - At each pixel image gradient horizontal (x - direction), vertical (y - direction) are calculated
# - These vectors have a direction atan(f_y/f_x), and magnitude sqrt(f_xˆ2 + f_yˆ2)
# - Gradient values are mapped to 0 - 255. Pixels with large negative change will be black,
# pixels with large positive change will be white and pixels with low or no change will be gray


# - Imagine a situation, that we took a small part of image (few pixels) and got following matrix:

#  0  100   0
# 70    0  120
#  0   50   0

#And we would like to calculate the oriented gradient --> gradientd magnitute and gradient angle;
#How we do that:
# - The Gradient value in the X - derection is 120 - 70 = 50
# - The Gradient value in the Y - derection is 100 - 50 = 50

#Gradient Magnitude: sqrt(50ˆ2 + 50ˆ2) = 70.1
#Gradient Angle: tanˆ(-1)*(50 / 50) = 45 degrees

#It means, we can assume, that in direction of 45 degrees, the gradien(the changes) of color will be 70.1 --> maybe;

#Now we would like to get Hhistogram of oriented(different angles) gradients;

#Step 1:
# - Using 8*8 pixel cell, compute mag/dir gradient; --> in both directions (X -> horizontal and Y -> Vertical) --> SobelX and SobelY

#Step2:
# - Create a histogram of generated 64 (8*8) gradient vectors
# - Each sell is then split into angular bins, eacj bin corresponds to a gradient direction

# -------------------------------------------Implementing HOG Feature Descriptor-------------------------------------------


#Reading the Image
img = imread(list_of_png[32])
print(img.shape, type(img))
cv.imshow('Original Image', img)
cv.waitKey(0)

# We can see that the shape of the image is 512 x 512.
# We will have to resize this image into 64 x 128. Note that we are using skimage which takes the input as height x width.

#Resizing Image --> We don't need to resize the image, but code can be used --> executed
# resized_img = resize(img, (128, 64))
# print(resized_img.shape, type(resized_img))
# cv.imshow('Resized Image', resized_img)
# cv.waitKey(0)

#Parameters of HOG feature describer
#  - The orientations are the number of buckets we want to create. Since I want to have a 9 x 1 matrix, I will set the orientations to 9
# pixels_per_cell defines the size of the cell for which we create the histograms. In the example we used 8 x 8 cells and here I will set the same value.
# As mentioned previously, you can choose to change this value
# -  We have another hyperparameter cells_per_block which is the size of the block over which we normalize the histogram.
# Here, we mention the cells per blocks and not the number of pixels. So, instead of writing 16 x 16, we will use 2 x 2 here
# The feature matrix from the function is stored in the variable fd, and the image is stored in hog_image. Let us check the shape of the feature matrix:


#What will we get
#Here, I am going to use the hog function from skimage.features directly. So we don’t have to calculate the gradients,
# magnitude (total gradient) and orientation individually. The hog function would internally calculate it and return the feature matrix.

#Creating HOG features
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=False)

print(fd.shape)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()


#Calculating Horizaontal and Vertical gradients separetly (SobelX, SobelY) ------> SOBEL METHOD

SCALE = 1
DELTA = 0
DDEPTH = cv.CV_16S  ## to avoid overflow

#Horizontal
gradx = cv.Sobel(img, DDEPTH, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
gradx = cv.convertScaleAbs(gradx)
print('Horizontal(X) Gradient using Sobel method:\n', gradx)
plt.title('Horizontal(X) Gradient Histogram using Sobel method ')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(gradx.ravel())
plt.show()
cv.imshow('Horizaontal Gradient using Sobel method', gradx)
cv.waitKey(0)

#Vertical
grady = cv.Sobel(img, DDEPTH, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
grady = cv.convertScaleAbs(grady)
print('Vertical(Y) Gradient using Sobel method: \n', grady)
plt.title('Vertical(Y) Gradient Histogram using Sobel method')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(grady.ravel());
plt.show()
cv.imshow('Vertical Gradient using Sobel method', grady)
cv.waitKey(0)


#Combined Horizontal and Vertical
grad = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
print('Combined Horizontal and Vertical using Sobel method: \n', grad)
plt.title('Combined Horizontal and Vertical using Sobel method ')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(grad.ravel())
plt.show()
cv.imshow('Combined Horizontal and Vertical using Sobel method: ', grad)
cv.waitKey(0)


# #Iamge Orientation
# # Compute the orientation of the image
# orientation = cv.phase(gradx, grady, angleInDegrees = True)
# print('Orientation: \n', orientation)


#Calculating Horizaontal and Vertical gradients separetly (prewittx, prewitty) ------> #PREWITT METHOD
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv.filter2D(img, -1, kernelx)
print('Horizontal(X) Gradient using Prewitt method:\n', prewittx)
plt.title('Horizontal(X) Gradient Histogram using Prewitt method ')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(prewittx.ravel())
plt.show()
cv.imshow('Horizaontal Gradient using Prewitt method', prewittx)
cv.waitKey(0)


prewitty = cv.filter2D(img, -1, kernely)
print('Vertical(Y) Gradient using Prewitt method:\n', prewitty)
plt.title('Vertical(Y) Gradient Histogram using Prewitt method ')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(prewitty.ravel())
plt.show()
cv.imshow('Horizaontal Gradient using Prewitt method', prewitty)
cv.waitKey(0)


#Combined Horizontal and Vertical
prewitt = cv.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
print('Combined Horizontal and Vertical using Prewitt method: \n', prewitt)
plt.title('Combined Horizontal and Vertical using Prewitt method ')
plt.xlabel('Orientation')
plt.ylabel('Gradient Magnitude')
plt.hist(prewitt.ravel())
plt.show()
cv.imshow('Combined Horizontal and Vertical using Prewitt method: ', prewitt)
cv.waitKey(0)
