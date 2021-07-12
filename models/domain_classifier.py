import torch.nn as nn
import torch.nn.functional as f
from models.grad_reverse import GradReverse


class DomainClassifier(nn.Module):
    """
    域鉴别器
    """

    def __init__(self):
        super(DomainClassifier, self).__init__()
        # self.linear1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.linear2 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        # out = f.relu(self.bn1(self.linear1(input)))
        # out = f.log_softmax(self.linear2(out), 1)
        out = f.relu(self.fc1(x))
        out = f.log_softmax(self.fc2(out), 1)

        return out


class SVHNDomainClassifier(nn.Module):
    """
    SVHN域鉴别器
    """

    def __init__(self):
        super(SVHNDomainClassifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        out = f.relu(self.bn1(self.fc1(x)))
        out = f.dropout(out)
        out = f.relu(self.bn2(self.fc2(out)))
        out = f.dropout(out)
        out = self.fc3(out)

        return f.log_softmax(out, 1)
