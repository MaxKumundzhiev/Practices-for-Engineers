# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------
#Link https://theailearner.com/tag/thinning-opencv/

import cv2
import numpy as np

# img = cv2.imread ('/Users/macbook/Documents/I/University ELTE/3 Semester/Sense Data/PNG/mmlogo.png', 0)
# size = np.size (img)
# skel = np.zeros (img.shape, np.uint8)
#
# ret, img = cv2.threshold (img, 127, 255, 0)
# element = cv2.getStructuringElement (cv2.MORPH_CROSS, (3, 3))
# done = False
#
# while (not done):
#     eroded = cv2.erode (img, element)
#     temp = cv2.dilate (eroded, element)
#     temp = cv2.subtract (img, temp)
#     skel = cv2.bitwise_or (skel, temp)
#     img = eroded.copy ()
#
#     zeros = size - cv2.countNonZero (img)
#     if zeros == size:
#         done = True
#
# cv2.imshow('Original Image', img)
# cv2.waitKey (0)
# cv2.destroyAllWindows ()
# cv2.imshow ("After Applying Thinning", skel)
# cv2.waitKey (0)
# cv2.destroyAllWindows ()


# Create an image with text on it
img = np.zeros ((100, 400), dtype='uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText (img, 'Dmitri', (10, 70), font, 2, (255), 4, cv2.LINE_AA)
img1 = img.copy ()

# Structuring Element
kernel = cv2.getStructuringElement (cv2.MORPH_CROSS, (6, 6))
# Create an empty output image to hold values
thin = np.zeros (img.shape, dtype='uint8')

# Loop until erosion leads to an empty set
while (cv2.countNonZero (img1) != 0):
    # Erosion
    erode = cv2.erode (img1, kernel)
    # Opening on eroded image
    opening = cv2.morphologyEx (erode, cv2.MORPH_OPEN, kernel)
    # Subtract these two
    subset = erode - opening
    # Union of all previous sets
    thin = cv2.bitwise_or (subset, thin)
    # Set the eroded image for next iteration
    img1 = erode.copy ()

cv2.imshow ('original', img)
cv2.imshow ('thinned', thin)
cv2.waitKey (0)