# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""
A projection network, P(·), which maps the normalized representation vector r into a vector z = P(r) ∈ RDP
suitable for computation of the contrastive loss. For our projection network, we use a multi-layer perceptron [18] with
a single hidden layer of size 2048 and output vector of size DP = 128. We again normalize this vector to lie on the
unit hypersphere, which enables using an inner product to measure distances in the projection space. The projection
network is only used for training the supervised contrastive loss. After the training is completed, we discard this
network and replace it with a single linear layer (for more details see Sec. 4). Similar to the results for self-supervised
contrastive learning [46, 6], we found representations from the encoder to give improved performance on downstream
tasks than those from the projection network. Thus our inference-time models contain exactly the same number of
parameters as their cross-entropy equivalents.
"""

import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ImageEmbedding (nn.Module):
    class Identity (nn.Module):
        def __init__(self): super().__init__ ()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=1024):
        super ().__init__ ()

        base_model = EfficientNet.from_pretrained ("efficientnet-b0")
        internal_embedding_size = base_model._fc.in_features
        base_model._fc = ImageEmbedding.Identity ()

        self.embedding = base_model

        self.projection = nn.Sequential (
            nn.Linear (in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU (),
            nn.Linear (in_features=embedding_size, out_features=embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding (image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding (image)
        projection = self.projection (embedding)
        return embedding, projection
