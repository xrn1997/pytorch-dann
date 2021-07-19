import torch.nn as nn
import models.block as mb


class ME(nn.Module):
    def __init__(self):
        super(ME, self).__init__()
        self.conv_block = mb.ConvBlock(kernel_size={3, 3}, out_channels=32, padding=1, in_channels=1)
        self.max_pool = mb.MaxPooling(num_feature=32)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.max_pool(x)
        return x
