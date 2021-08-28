import time

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

        rss = np.loadtxt(dataset_path + train_path + "_rss_21Aug17.csv", delimiter=',', dtype=np.float32)

        self._data_len = rss.shape[0]  # RSS指纹数量
        self.ap_len = rss.shape[1]  # AP数量 992
        self._rss = torch.from_numpy(rss)  # RSS 992维向量

        self._co_data = torch.from_numpy(
            np.loadtxt(dataset_path + train_path + "_coordinates_21Aug17.csv", delimiter=',',
                       dtype=np.float32))  # X、Y、Z（相对位置坐标）
        co_dic = {}  # 字典
        self._co_label = []  # 标签
        co_idx = 0  # 索引
        for c in self._co_data.int():
            temp = c.tolist()
            if temp not in co_dic.values():
                co_dic[co_idx] = temp
                self._co_label.append(co_idx)
                co_idx = co_idx + 1
            else:
                self._co_label.append(list(co_dic.keys())[list(co_dic.values()).index(temp)])
        self.co_size = len(co_dic)  # 位置数量

        self._date = np.loadtxt(dataset_path + train_path + "_date_21Aug17.csv", delimiter=',',
                                dtype=str)  # 采集日期 ，如：2017/8/19 15:52:25
        self._device = np.loadtxt(dataset_path + train_path + "_device_21Aug17.csv", delimiter=',',
                                  dtype=str)  # 采集设备型号，如：HUAWEI T1 7.0
        self.date_domain = []
        for co_idx in self._date:
            if int(co_idx[-8:-6]) < 11:
                self.date_domain.append("A")
            elif int(co_idx[-8:-6]) < 14:
                self.date_domain.append("B")
            else:
                self.date_domain.append("C")
        temp = pd.DataFrame({'date': self.date_domain, 'device': self._device})
        self._one_hot_label = pd.get_dummies(temp, columns=temp.columns)
        self.domain_size = self._one_hot_label.shape[1]  # 域的数量

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        """
        :return: 1.RSS二维向量；2.位置相对坐标；3.位置标签；4.one-hot域标签,one-hot标签前三个为时间，后面为设备。
        """
        # 将1维RSS向量转换为2维RSS向量
        rss_item = self._rss[index].tolist()
        mx = np.matrix(rss_item * self.ap_len, dtype=np.float32).reshape(self.ap_len, self.ap_len).transpose()
        for i in range(0, self.ap_len):
            mx[:, i] = (mx[:, i] - rss_item[i]) / rss_item[i]
        result = mx.A.reshape(1, self.ap_len, self.ap_len)  # 通道数为 1
        return result, self._co_data, self._co_label[index], self._one_hot_label.values[index]


class UJIndoorLocDataSet(Dataset):
    def __init__(self, dataset_path, train=True):
        if train:
            train_path = "trainingData.csv"
        else:
            train_path = "validationData.csv"

        all_data = np.loadtxt(dataset_path + train_path, delimiter=',', skiprows=1, dtype=np.float32)

        self._data_len = all_data.shape[0]  # RSS指纹数量
        self.ap_len = all_data.shape[1] - 9  # AP数量 520
        self._rss = torch.from_numpy(all_data[:, :-9])  # RSS 520维向量
        # TODO 这个数据集的位置信息还没有处理
        self._position = all_data[:, -9:-7]  # 经度、纬度
        self._space = all_data[:, -7:-3]  # 楼层 、楼、房间、相对位置（门内1、门外2）
        self._collector = all_data[:, -3:-1]  # 用户、手机

        self._date = all_data[:, -1:].reshape(self._data_len)  # 时间戳，如：1371713733
        self.date_domain = []
        for i in self._date:
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(i)))
            if int(date[-8:-6]) < 16:
                self.date_domain.append("A")
            elif int(date[-8:-6]) < 19:
                self.date_domain.append("B")
            else:
                self.date_domain.append("C")
        self.temp = pd.DataFrame(
            {'date': self.date_domain, 'device': [int(i[1]) for i in self._collector]})
        self._one_hot_label = pd.get_dummies(self.temp, columns=self.temp.columns)
        self.domain_size = (self._one_hot_label.shape[1] - 3) * 3  # 域的数量

    def __len__(self):
        return self._data_len

    def __getitem__(self, index):
        # 将1维RSS向量转换为2维RSS向量
        rss_item = self._rss[index].tolist()
        mx = np.matrix(rss_item * self.ap_len, dtype=np.float32).reshape(self.ap_len, self.ap_len).transpose()
        for i in range(0, self.ap_len):
            mx[:, i] = (mx[:, i] - rss_item[i]) / rss_item[i]
        result = mx.A.reshape(1, self.ap_len, self.ap_len)  # 通道数为 1
        result = torch.from_numpy(result)
        return result, self._position[index], self._one_hot_label.values[index]


#  测试用代码
if __name__ == '__main__':
    tampere_path = "./Tampere/DISTRIBUTED_OPENSOURCE_version2/FINGERPRINTING_DB/"
    uj_indoor_loc_path = "./UJIndoorLoc/"

    dataset = TampereDataset(tampere_path, train=True)
    # dataset = TampereDataset(tampere_path, train=False)
    # dataset = UJIndoorLocDataSet(uj_indoor_loc_path, train=False)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,
                            shuffle=True,
                            num_workers=3,
                            pin_memory=True)
    print("ap_len", dataset.ap_len)
    print("co_size", dataset.co_size)
    print("domain_size", dataset.domain_size)
    for data in dataloader:
        print(data[1].dtype)
        print(data[3].shape)
        break
    print(dataset.domain_size)
