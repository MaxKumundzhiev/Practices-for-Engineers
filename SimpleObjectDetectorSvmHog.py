# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

#Imports

import cv2


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cam = cv2.VideoCapture(0)

while (1):
    ret, frames = cam.read()
    frames = cv2.resize(frames, (400, 300))
    (rects, weights) = hog.detectMultiScale(frames, scale = 1.1, winStride = (2,2))
    #rects = hog.getDefaultPeopleDetector()
    for (x,y,w,h), wei in zip(rects, weights):
        cv2.rectangle(frames, (x,y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Frame', frames)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cam.release()



