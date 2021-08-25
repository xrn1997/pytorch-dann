import numpy as np
import torch
import torch.nn as nn


class OneHotNLLLoss(nn.Module):
    """
    target为one-hot的交叉熵损失函数
    """

    def __init__(self, reduction='mean'):
        super(OneHotNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if self.reduction == 'sum':
            n = 1
        elif self.reduction == 'mean':
            n = len(y)
        else:
            raise Exception('reduction 参数错误')
        result = -torch.sum(x * y) / n
        return result
