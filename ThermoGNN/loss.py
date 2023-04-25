import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np
from scipy.special import lambertw


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_t, weights=None):
        loss = (y - y_t) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        return torch.mean(loss)


class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=1, batch_size=1):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

    def forward(self, logits, targets):
        l_i = F.mse_loss(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.mse_loss(logits, targets, reduction='none') - self.tau) * sigma + self.lam * (
                    torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma


if __name__ == '__main__':
    sl = SuperLoss()
    pred = torch.ones((4, 128)).cuda()
    label = torch.zeros((4, 128)).cuda()
    out = sl(pred, label)
