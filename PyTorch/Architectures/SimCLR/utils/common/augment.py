# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""
Data augmentation module.
Take a picture and create two augmented ones from it.
- The first step of such augmentation is random crop and then resize to original size.
- The second step is the augmentation itself, the authors try 3 options: autoaugment,
randaugment, simaugment (random color distortion + Gaussian blurring + sparse image warp);
"""

