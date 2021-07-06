from __future__ import print_function
from util.download import download
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import scipy.io as sio


class SynDig(data.Dataset):
    """
    方法暂时弃用，因为找不到数据集。
    """
    url = 'https://raw.githubusercontent.com/domainadaptation/datasets/master/synth/'
    split_list = {
        'train': ["synth_train_32x32.mat"],
        'train_small': ["synth_train_32x32_small.mat"],
        'test': ["synth_test_32x32.mat", ""],
        'test_small': ["synth_test_32x32_small.mat"]
    }

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, is_download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="train_small or split="test" or split="test_small"')
        self.filename = self.split_list[self.split][0]

        if is_download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               'You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat['X']
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        for split in self.split_list:
            filename = self.split_list[split][0]
            path = os.path.join(self.root, filename)
            download(self.url+filename, os.path.abspath(path))
        return
