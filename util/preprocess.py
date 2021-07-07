import os, shutil

data_dir = '../data/MNIST_M'
train_labels = data_dir + '/mnist_m_train_labels.txt'
test_labels = data_dir + '/mnist_m_test_labels.txt'
train_images = data_dir + '/mnist_m_train'
test_images = data_dir + '/mnist_m_test'


def mk_dirs(path):
    train_dir = path + '/' + 'train'
    test_dir = path + '/' + 'test'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in range(0, 10):
        if not os.path.exists(train_dir + '/' + str(i)):
            os.mkdir(train_dir + '/' + str(i))
        if not os.path.exists(test_dir + '/' + str(i)):
            os.mkdir(test_dir + '/' + str(i))


def process(labels_path, images_path, data_path):
    if not os.path.exists(images_path):
        return
    with open(labels_path) as f:
        for line in f.readlines():
            img = images_path + '/' + line.split()[0]
            path = data_path + '/' + line.split()[1]
            shutil.move(img, path)
    shutil.rmtree(images_path)


# 创建文件夹 train 和 test
mk_dirs(data_dir)
# 划分训练集和测试集，根据数字将文件分到对应的文件夹中
process(train_labels, train_images, data_dir + '/train')
process(test_labels, test_images, data_dir + '/test')
