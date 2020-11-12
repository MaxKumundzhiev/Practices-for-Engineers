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
- The second step is the augmentation itself, the authors try 3 options:
autoaugment:
    - https://github.com/DeepVoltaire/AutoAugment
    - https://github.com/DeepVoltaire/AutoAugment#example-as-a-pytorch-transform---imagenet
    - https://github.com/DeepVoltaire/AutoAugment#autoaugment---learning-augmentation-policies-from-data
    - https://arxiv.org/pdf/1805.09501v1.pdf
    - https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html

randaugment:
    - https://github.com/heartInsert/randaugment/blob/master/Rand_Augment.py
    - https://www.groundai.com/project/randaugment-practical-data-augmentation-with-no-separate-search/1
simaugment (random color distortion + Gaussian blurring + sparse image warp);
"""

import os
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, transforms

from settings import PATHS, SIZE
from utils.external.auto_augment import ImageNetPolicy
from utils.external.random_augment import Rand_Augment

root_dir = PATHS['ROOT_DIR']
save_dir = PATHS['SAVE_DIR']
images_names = os.listdir(PATHS['ROOT_DIR'])


# Define transformations
transforms = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.RandomCrop(size=(128)),
     transforms.Resize(size=SIZE),
     ImageNetPolicy(),  # autoaugment
     #Rand_Augment()  # randaugment

    # simaugment

    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    ]
)


# Plotting images
plt.figure()
for image_index, image_name in enumerate(images_names):
    if image_index % 60 == 0:
        full_image_path = os.path.join(root_dir, image_name)

        original_image = Image.open(full_image_path)  # reading with PIL cause torch works with PIL objects
        transformed_image = transforms(original_image)

        plt.title('original_image')
        plt.imshow(original_image)
        plt.show()

        plt.title('transformed_image')
        plt.imshow(transformed_image)
        plt.show()

