## PyTorch implementation of SimCLR "This repository target is to represent step by step SimCLR framework/algorithm."

### SimCLR (Simple Contrastive Learning Representation) Overview.
SimCLR learns basic image representations on an unlabeled corpus and can be fine-tuned with a small set of labeled images for a classification task. The representations are learned through a method called contrastive learning, where the model simultaneously maximizes agreement between differently transformed views of the same image and minimizes agreement between transformed views of different images. 

![Image 0](https://venturebeat.com/wp-content/uploads/2020/04/image4-4.gif?w=540&resize=540%2C600&strip=all)

### Let us define chain of steps to proceed with:
1. Obtain {un}labeled data to work with.
- Under this clause: we will obtain labeled dataset in advance to keep labels information for further easier comparison of results.
To make process easier, we will look for < 5 classes dataset.
So, let us use "Torch" [STL 10](https://cs.stanford.edu/~acoates/stl10/) option for this purpose.    



### :zap:How such approaches work:zap:
#### Data augmentation module.
Take a picture and create two augmented ones from it. The first step of such augmentation is random crop and then resize to original size. The second step is the augmentation itself, the authors try 3 options: autoaugment, randaugment, simaugment (random color distortion + Gaussian blurring + sparse image warp);

#### Encoder is just a backbone.
The authors took ResNet-50 and ResNet-200, at the exit we take the last pooling, i.e. vector size 2048;

#### Projection network.
Take the output of the encoder and run it through a small network. In this case MLP with one hidden layer, the size of the output layer is 128. The vector is normalized again to calculate the loss. This MLP is only used for supervised contrastive loss training. After training, the MLP is removed and replaced with a regular linear layer. As a result, the encoder gives better results for downstream applications.

![Image 1](https://habrastorage.org/webt/yl/he/4l/ylhe4l7ffdiewrlufvrzxswjc-0.png)

### Contrastive Losses: Self-Supervised and Supervised
In general, the approach is that from each batch of data two new ones are generated - with different augmentations.

![Image 2](https://habrastorage.org/webt/en/vy/z0/envyz0sxyx_woexe16_tzd_p_qc.png)
