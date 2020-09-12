# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

"""
Support explanation torch math operations:
torch.tensor - is a multi-dimensional matrix containing elements of a single data type.
torch.ones - tensor filled with the scalar value 1, with the shape defined by the variable argument size.
torch.cat - concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
torch.exp - returns a new tensor with the exponential of the elements of the input tensor input.
torch.log - returns a new tensor with the natural logarithm of the elements of input.
torch.unsqueeze(input, dim) → Tensor - Returns a new tensor with a dimension of size one inserted at the specified position.
                                       The returned tensor shares the same underlying data with this tensor.
torch.scatter - https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html


F.normalize - Performs L_p normalization of inputs over specified dimension. With the default arguments it uses the Euclidean norm over vectors along dimension 1 for normalization.
F.cosine_similarity - Returns cosine similarity between x_1 and x_2, computed along dim.
"""


import torch
from torch import nn
import torch.nn.functional as F


class ContarastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer('temperature', torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, embedding_i, embedding_j):
        """
        Notes:
            {embedding_i and embedding_j} are batches of embeddings,
            where corresponding indices are pairs {z_i, z_j} as in SimCLR paper
        """

        z_i, z_j = F.normalize(embedding_i, dim=1), F.normalize(embedding_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=2)
        if self.verbose:
            print('Similarity Matrix\n', similarity_matrix, '\n')

        def loss_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            similarity_i_j = similarity_matrix[i, j]

            if self.verbose:
                print(f"sim({i}, {j})={similarity_i_j}")

            numerator = torch.exp(similarity_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(embedding_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose:
                print("Denominator", denominator)

            loss_i_j = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_i_j}\n")

            return loss_i_j.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += loss_ij(k, k + N) + loss_ij(k + N, k)
        return 1.0 / (2 * N) * loss