## Representation Learning Framework

:zap:In general, about how such approaches work:zap:

### data augmentation module.
Take a picture and create two augmented ones from it. The first step of such augmentation is random crop and then resize to original size. The second step is the augmentation itself, the authors try 3 options: autoaugment, randaugment, simaugment (random color distortion + Gaussian blurring + sparse image warp);

### encoder is just a backbone.
The authors took ResNet-50 and ResNet-200, at the exit we take the last pooling, i.e. vector size 2048;

### projection network.
Take the output of the encoder and run it through a small network. In this case MLP with one hidden layer, the size of the output layer is 128. The vector is normalized again to calculate the loss. This MLP is only used for supervised contrastive loss training. After training, the MLP is removed and replaced with a regular linear layer. As a result, the encoder gives better results for downstream applications.

![Image 1](https://habrastorage.org/webt/yl/he/4l/ylhe4l7ffdiewrlufvrzxswjc-0.png)

## Contrastive Losses: Self-Supervised and Supervised
In general, the approach is that from each batch of data two new ones are generated - with different augmentations.

![Image 2](https://habrastorage.org/webt/en/vy/z0/envyz0sxyx_woexe16_tzd_p_qc.png)
