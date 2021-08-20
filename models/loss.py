import numpy as np
import torch
import torch.nn as nn


class OneHotNLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(OneHotNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if self.reduction == 'sum':
            n = 1
        elif self.reduction == 'mean':
            n = np.array(y).shape[0]
        else:
            raise Exception('reduction 参数错误')
        result = -torch.sum(x * y) / n
        return result
