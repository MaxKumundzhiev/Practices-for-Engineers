# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

#How manually to teach our's own detector


#Imports
import cv2
import os
#dlib --> for teaching our own detector
import dlib
import xml.etree.ElementTree as pars


#For dlib
#detector = dlib.train_simple_object_detector(images, annots, options)

#options --> look below
#options = dlib.simple_object_detector_training_options()


#Lenght of img array = annot array

#images --> array of images; [img1, img2, ..., imgn]
#annotations --> array of coordinates of rectangle in which we can map our object: [(x, y, x2, y2), (x, y, x2, y2), (x, y, x2, y2) ..., (x, y, x2, y2)]



dir_images = '/Users/macbook/PycharmProjects/Images'
dir_annotations = '/Users/macbook/PycharmProjects/Annotations'

ImgList = os.listdir(dir_images)
print(ImgList)

for FileName in ImgList:
    image = cv2.imread(dir_images + FileName)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    OnlyFileName = FileName.split('.')[0]
    print(OnlyFileName)
    #print(os.path.join(dir_annotations, '%s'%OnlyFileName + '.xml'))

    e = pars.parse(os.listdir(dir_annotations))
    print(e)
images = []
annotation = []




