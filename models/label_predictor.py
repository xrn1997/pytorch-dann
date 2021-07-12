import torch.nn as nn
import torch.nn.functional as f


class LabelPredictor(nn.Module):
    """
    标签预测器
    """

    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.linear1 = nn.Linear(48 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = f.relu(self.linear1(x))
        out = self.linear2(f.dropout(out))
        out = f.relu(out)
        out = self.linear3(out)
        return f.log_softmax(out, 1)


class SVHNLabelPredictor(nn.Module):
    """
    SVHN标签预测器
    """

    def __init__(self):
        super(SVHNLabelPredictor, self).__init__()
        self.linear1 = nn.Linear(128 * 3 * 3, 3072)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear2 = nn.Linear(3072, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.linear3 = nn.Linear(2048, 10)

    def forward(self, x):
        out = f.relu(self.bn1(self.linear1(x)))
        out = f.dropout(out)
        out = f.relu(self.bn2(self.linear2(out)))
        out = self.linear3(out)

        return f.log_softmax(out, 1)
