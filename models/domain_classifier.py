import torch.nn as nn
import models.block as mb
from models.grad_reverse import GradReverse


class MD(nn.Module):

    def __init__(self, d):
        super(MD, self).__init__()
        self.block1 = mb.ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = mb.MaxPooling(num_feature=32)
        self.block2 = mb.ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.soft_max = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(64 * 3 * 3, d)

    def forward(self, x, constant):
        batch_size = x.size(0)
        x = GradReverse.grad_reverse(x, constant)
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.avg_pool2d(x)
        x = self.bn(x)
        x = self.soft_max(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
