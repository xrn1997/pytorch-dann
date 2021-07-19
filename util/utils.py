from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import params
import torchvision
import numpy as np
import os, time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


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
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def display_images(dataloader, length=8, img_name=None):
    """
    随机展示图片  \r\n
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
    images, labels = next(data_iter)  # next dataloader 的默认大小就是batch_size的大小
    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()  # make_grid的作用是将若干幅图像拼成一幅图像。
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


def plot_embedding(loc, y, d, title=None, img_name=None):
    """
    绘图 \r\n
    Plot an embedding loc with the class label y colored by the domain d.

    :param loc: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param img_name: the name of saving image

    :return:
    """
    if params.fig_mode is None:
        return

    # normalization 标准化
    x_min, x_max = np.min(loc, 0), np.max(loc, 0)
    loc = (loc - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(loc.shape[0]):
        # plot colored number
        plt.text(loc[i, 0], loc[i, 1],
                 str(y[i]),  # 标签的符号
                 color=plt.cm.bwr(d[i] / 1.),  # bwr这个方法找不到
                 fontdict={'weight': 'bold', 'size': 9}  # 标签的样式
                 )

    plt.xticks([]), plt.yticks([])  # 设置X轴和Y轴的刻度

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
