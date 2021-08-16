import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TampereDataset(Dataset):

    def __init__(self, dataset_path, train=True):

        if train:
            train_path = "Training"
        else:
            train_path = "Test"

        rss = np.loadtxt(dataset_path + train_path + "_rss_21Aug17.csv", delimiter=',', dtype=int)

        self._rss_len = rss.shape[0]  # RSS指纹数量
        self.ap_len = rss.shape[1]  # AP数量 992
        self._rss = torch.from_numpy(rss)  # RSS 992维向量
        self._coordinate = np.loadtxt(dataset_path + train_path + "_coordinates_21Aug17.csv", delimiter=',',
                                      )  # X、Y、Z（相对位置坐标）
        self._date = np.loadtxt(dataset_path + train_path + "_date_21Aug17.csv", delimiter=',',
                                dtype=str)  # 采集日期 ，如：2017/8/19 15:52:25
        self._device = np.loadtxt(dataset_path + train_path + "_device_21Aug17.csv", delimiter=',',
                                  dtype=str)  # 采集设备型号，如：HUAWEI T1 7.0

        temp = pd.DataFrame({'date': [i[:-6] for i in self._date], 'device': self._device})
        self._one_hot_label = pd.get_dummies(temp, columns=temp.columns)

    def __len__(self):
        return self._rss_len

    def __getitem__(self, index):
        return self._rss[index], self._coordinate[index], self._one_hot_label.values[index]


class UJIndoorLocDataSet(Dataset):
    def __init__(self, dataset_path, train=True):
        if train:
            train_path = "trainingData.csv"
        else:
            train_path = "validationData.csv"

        all_data = np.loadtxt(dataset_path + train_path, delimiter=',', skiprows=1)

        self._rss_len = all_data.shape[0]  # RSS指纹数量
        self.ap_len = all_data.shape[1] - 9  # AP数量 520
        self._rss = torch.from_numpy(all_data[:, :-9])  # RSS 520维向量
        self._position = all_data[:, -9:-7]  # 经度、纬度
        self._space = all_data[:, -7:-3]  # 楼层 、楼、房间、相对位置（门内1、门外2）
        self._collector = all_data[:, -3:-1]  # 用户、手机
        self._date = all_data[:, -1:]  # 时间戳，如：1371713733
        # 10位时间戳只保留前八位
        temp = pd.DataFrame(
            {'date': [int(i[0] / 100) for i in self._date], 'device': [int(i[1]) for i in self._collector]})
        self._one_hot_label = pd.get_dummies(temp, columns=temp.columns)

    def __len__(self):
        return self._rss_len

    def __getitem__(self, index):
        return self._rss[index], self._position[index], self._one_hot_label.values[index]

    #  测试用代码


if __name__ == '__main__':
    tampere_path = "./Tampere/DISTRIBUTED_OPENSOURCE_version2/FINGERPRINTING_DB/"
    uj_indoor_loc_path = "./UJIndoorLoc/"

    # dataset = TampereDataset(tampere_path, train=True)
    # dataset = TampereDataset(tampere_path, train=False)
    dataset = UJIndoorLocDataSet(uj_indoor_loc_path, train=False)

    train_loader = DataLoader(dataset, batch_size=1)
    print(len(dataset))
    print(dataset.ap_len)

    for data in train_loader:
        print(data)
        break
