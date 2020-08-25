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


class ProjectionNet(nn.Module):
    def __init__(self):
        super(ProjectionNet).__init__(self)
