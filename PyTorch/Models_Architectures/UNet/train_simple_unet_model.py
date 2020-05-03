# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np

def fix_random_numbers():
    import random as rd
    import numpy as np
    rd.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return True


path = '/Users/macbook/Desktop/TestUnet/SpineFinder/samples/detection/testing/4517454-1-sample.npy'
image = np.load(path)
print(len(image))
print(image.shape)
print(image[0].shape)

plt.imshow(image[6, :, :])
plt.show()

#When make input for NeuralNett make data unsqeezed
# print(y.unsqueeze_(1).shape)
# print(x.unsqueeze_(1).shape)


######Trainig Process
batch_size = 10

# for epoch in range(50):
#     training_generator = DataGenerator(partition['train'], labels, training_sample_dir, **params)
#     validation_generator = DataGenerator(partition['validation'], labels, val_sample_dir, **params)


