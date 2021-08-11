import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FingerPrintDataset(Dataset):

    def __init__(self, rss_file_path, date_path, device_path, coordinate_path):
        rss = np.loadtxt(rss_file_path, delimiter=',', dtype=np.float32)
        coordinate = np.loadtxt(coordinate_path, delimiter=',', dtype=np.float32)

        self._data_len = rss.shape[0]
        self.ap_len = rss.shape[1]
        self._data = torch.from_numpy(rss)
        self._coordinate = torch.from_numpy(coordinate)
        self._date = np.loadtxt(date_path, delimiter=',', dtype=str)
        self._device = np.loadtxt(device_path, delimiter=',', dtype=str)

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        return self._data[index], self._coordinate[index], self._date[index], self._device[index]


if __name__ == '__main__':
    path = "./data/Tampere/DISTRIBUTED_OPENSOURCE_version2/FINGERPRINTING_DB/"
    dataset = FingerPrintDataset(rss_file_path=path + "Training_rss_21Aug17.csv",
                                 date_path=path + "Training_date_21Aug17.csv",
                                 device_path=path + "Training_device_21Aug17.csv",
                                 coordinate_path=path + "Training_coordinates_21Aug17.csv")
    train_loader = DataLoader(dataset, batch_size=1)
    for data in train_loader:
        print(data[0].shape)
        print(data[1].shape)
    print(len(dataset))
    print(dataset.ap_len)
