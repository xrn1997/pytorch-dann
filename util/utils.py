from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import params


def get_train_loader(dataset):
    """
    Get train dataloader of source domain or target domain

    :return: dataloader
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])
        data = datasets.MNIST(root=params.mnist_path, transform=transform,
                              download=True)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,  # 每次处理的batch大小
                                shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                                num_workers=3,  # 线程数
                                pin_memory=True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop(28),  # 随机长宽比裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])
        data = datasets.ImageFolder(root=params.mnist_m_path + '/train', transform=transform)
        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=3,
                                pin_memory=True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader


def get_test_loader(dataset):
    """
    Get test dataloader of source domain or target domain

    :return: dataloader
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.MNIST(root=params.mnist_path, train=False, transform=transform,
                              download=True)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=3,
                                pin_memory=True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            # transforms.RandomCrop((28)),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnist_m_path + '/test', transform=transform)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=3,
                                pin_memory=True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader


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
