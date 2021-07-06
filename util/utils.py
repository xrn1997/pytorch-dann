from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data import SynDig
from train import params
import torch
import torchvision
import numpy as np
import os, time
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # agg不显示图片


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

        data = datasets.MNIST(root=params.mnist_path, train=True, transform=transform,
                              download=True)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size, # 每次处理的batch大小
                                shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                                num_workers=2,  # 线程数
                                pin_memory=True)
        '''
      pin_memory就是锁页内存。
      创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
      主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），
      而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。显卡中的显存全部是锁页内存,当计算机的内存充足的时候，可以设置pin_memory=True。
      当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，
      因此pin_memory默认为False。
      '''

    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnist_m_path + '/train', transform=transform)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data1 = datasets.SVHN(root=params.svhn_path, split='train', transform=transform, download=True)
        data2 = datasets.SVHN(root=params.svhn_path, split='extra', transform=transform, download=True)

        data = torch.utils.data.ConcatDataset((data1, data2))

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.RandomCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = SynDig.SynDig(root=params.synth_path, split='train', transform=transform, is_download=True)

        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,
                                shuffle=True,
                                num_workers=2,
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

        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            # transforms.RandomCrop((28)),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnist_m_path + '/test', transform=transform)

        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = datasets.SVHN(root=params.svhn_path, split='test', transform=transform, download=True)

        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = SynDig.SynDig(root=params.synth_path, split='test', transform=transform, is_download=True)

        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer

    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def display_images(dataloader, length=8, img_name=None):
    """
    Randomly sample some images and display

    :param dataloader: maybe train dataloader or test dataloader
    :param length: number of images to be displayed
    :param img_name: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # randomly sample some images.
    data_iter = iter(dataloader)
    images, labels = next(data_iter) # next dataloader 的默认大小就是batch_size的大小
    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()
    images = images / 2 + 0.5
    images = np.transpose(images, (1, 2, 0))

    if params.fig_mode == 'display':
        plt.imshow(images)
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if img_name is None:
            img_name = 'displayImages' + str(int(time.time()))

        # Check extension in case.
        if not (img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg')):
            img_name = os.path.join(folder, img_name + '.jpg')

        plt.imsave(img_name, images)
        plt.close()

    # print labels
    print(' '.join('%5s' % labels[j].item() for j in range(length)))


def plot_embedding(X, y, d, title=None, img_name=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param img_name: the name of saving image

    :return:
    """
    if params.fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)

    if params.fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if img_name is None:
            img_name = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg')):
            img_name = os.path.join(folder, img_name + '.jpg')

        print('Saving ' + img_name + ' ...')
        plt.savefig(img_name)
        plt.close()
