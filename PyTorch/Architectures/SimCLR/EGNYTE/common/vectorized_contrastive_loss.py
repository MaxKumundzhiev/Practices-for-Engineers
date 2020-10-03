# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2,
                                                           batch_size * 2,
                                                           dtype=bool)).float()
                             )
    def forward(self, embedding_i, embedding_j):
        """
        Notes:
            {embedding_i and embedding_j} are batches of embeddings,
            where corresponding indices are pairs {z_i, z_j} as in SimCLR paper

        Args:
            embedding_i: batch projection of images after the first augmentation
            embedding_j: batch projection of images after the second augmentation
        """
        z_i = F.normalize(embedding_i, dim=1)
        z_j = F.normalize(embedding_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=2
                                                )

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

