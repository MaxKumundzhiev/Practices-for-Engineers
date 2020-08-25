# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLossELI5 (nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super ().__init__ ()
        self.batch_size = batch_size
        self.register_buffer ("temperature", torch.tensor (temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize (emb_i, dim=1)
        z_j = F.normalize (emb_j, dim=1)

        representations = torch.cat ([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity (representations.unsqueeze (1), representations.unsqueeze (0), dim=2)
        if self.verbose: print ("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print (f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp (sim_i_j / self.temperature)
            one_for_not_i = torch.ones ((2 * self.batch_size,)).to (emb_i.device).scatter_ (0, torch.tensor ([i]), 0.0)
            if self.verbose: print (f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum (
                one_for_not_i * torch.exp (similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print ("Denominator", denominator)

            loss_ij = -torch.log (numerator / denominator)
            if self.verbose: print (f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze (0)

        N = self.batch_size
        loss = 0.0
        for k in range (0, N):
            loss += l_ij (k, k + N) + l_ij (k + N, k)
        return 1.0 / (2 * N) * loss


def test():
    I = torch.tensor ([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]])
    J = torch.tensor ([[1.0, 0.75], [2.8, -1.75], [1.0, 4.7]])
    loss_eli5 = ContrastiveLossELI5 (batch_size=3, temperature=1.0, verbose=True)
    loss_eli5(I, J)


if __name__ == "__main__":
    print(test())

