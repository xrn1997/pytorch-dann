from torch.utils.data import DataLoader

from trains import params
from data import fp_dataset as fp


def get_data_loader(dataset, dataset_path, train=True):
    """
    获得不同数据集的DataLoader
    :return: dataloader数据加载集,domain_size域的数量
    """
    if dataset == 'Tampere':
        data = fp.TampereDataset(dataset_path, train=train)
    elif dataset == 'UJIndoorLoc':
        data = fp.UJIndoorLocDataSet(dataset_path, train=train)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))
    return DataLoader(dataset=data,
                      batch_size=params.batch_size,  # 每次处理的batch大小
                      shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                      num_workers=3,  # 线程数
                      pin_memory=True), data.domain_size, data.co_size, data.ap_len


def optimizer_scheduler(optimizer, p):
    """
    调整学习率 \r\n
    Adjust the learning rate of optimizer

    :param optimizer: optimizer for updating parameters
    :param p: data variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer
