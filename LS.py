import torch
import torch.nn as nn
import torch.nn.functional as F
from test_ssim import ssim


class lsloss(nn.Module):
    def __init__(self, L=100, alpha=0.2):
        super(lsloss, self).__init__()
        self.L = L
        self.alpha = alpha

    def forward(self, x, y):
        b, c, d, h, w = x.shape

        lossL = F.l1_loss(x, y, size_average=True)
        x1 = x.squeeze(0).permute(1, 0, 2, 3)
        y1 = y.squeeze(0).permute(1, 0, 2, 3)
        t_SSIM = 0
        for i in range(self.L):
            SSIM = ssim(x1[i], y1[i])
            t_SSIM += SSIM
        lossS = 1.0 - t_SSIM / self.L
        loss = self.alpha * lossL + (1.0-self.alpha) * lossS
        return loss


