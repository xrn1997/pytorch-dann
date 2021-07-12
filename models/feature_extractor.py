import torch.nn as nn
import torch.nn.init as init


class FeatureExtractor(nn.Module):
    """
    特征提取器
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 卷积核现在需要输入元组，输入整数pycharm会有警告。
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(5, 5))
        self.drop2d = nn.Dropout2d()
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        x = self.relu(self.max_pool2d(self.conv1(x)))
        x = self.relu(self.max_pool2d(self.drop2d(self.conv2(x))))
        x = x.view(-1, 48 * 4 * 4)
        return x


class SVHNFeatureExtractor(nn.Module):
    """
    SVHN特征提取器
    """

    def __init__(self):
        super(SVHNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.max_pool2d = nn.MaxPool2d(3, 2)
        self.conv3_drop = nn.Dropout2d()
        self.relu = nn.ReLU()
        self.init_params()

    def init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool2d(x)
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.max_pool2d(x)
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.conv3_drop(x)

        return x.view(-1, 128 * 3 * 3)
