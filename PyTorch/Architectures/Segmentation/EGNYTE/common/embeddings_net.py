# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

# Here I define the ImageEmbedding neural network which is based on EfficientNet-b0 architecture.
# I swap out the last layer of pre-trained EfficientNet with identity function and add projection
# for image embeddings on top of it (following the SimCLR paper) with Linear-ReLU-Linear layers.
# It was shown in the paper that the non-linear projection head (i.e Linear-ReLU-Linear) improves
# the quality of the embeddings.

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ImageEmbedding(nn.Module):
    class Identity(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=1024):
        super().__init__()

        base_model = EfficientNet.from_pretrained('efficientnet-b0')
        internal_embedding_size = base_model._fc.in_features
        base_model._fc = ImageEmbedding.Identity()

        self.embedding = base_model
        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        return embedding, projection


