import torch.nn as nn
import models.block as mb


class M1(nn.Module):
    """
    标签预测模型M1
    """

    def __init__(self, c):
        super(M1, self).__init__()
        self.block1 = mb.ConvBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = mb.MaxPooling(num_feature=32)
        self.block2 = mb.ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.soft_max = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(64 * 3 * 3, c)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool2d(x)
        x = self.bn(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x


class M2(nn.Module):
    """
    标签预测模型M2
    """

    def __init__(self, c):
        super(M2, self).__init__()
        self.block1 = mb.ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2)
        self.max_pool = mb.MaxPooling(num_feature=32)
        self.block2 = mb.ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.soft_max = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(64 * 5 * 5, c)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool2d(x)
        x = self.bn(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x


class M3(nn.Module):
    """
    标签预测模型M3
    """

    def __init__(self, c):
        super(M3, self).__init__()
        self.block1 = mb.ResidualBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.max_pool = mb.MaxPooling(num_feature=32)
        self.block2 = mb.ResidualBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.soft_max = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(64 * 3 * 3, c)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.avg_pool2d(x)
        x = self.bn(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x
