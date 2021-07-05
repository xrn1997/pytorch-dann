"""
Main script for models
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from models import models
from train import test, train, params
from util import utils
from sklearn.manifold import TSNE
from torch.autograd import Variable

import time
import torch
import torch.nn as nn
import numpy as np
import argparse, sys, os


def visualize_performance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                          tgt_test_dataloader, num_of_samples=None, img_name=None):
    """
    Evaluate the performance of DANN and source only by visualization.

    :param feature_extractor: network used to extract feature from target samples 特征提取器
    :param class_classifier: network used to predict labels 标签预测器
    :param domain_classifier: network used to predict domain 域鉴别器
    :param src_test_dataloader: test dataloader of source domain 源域测试数据
    :param tgt_test_dataloader: test dataloader of target domain 目标域测试数据
    :param num_of_samples: the number of samples (from train and test respectively) for t-sne
    :param img_name: the name of saving image

    :return:
    """

    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    # 从源域和目标域随机选取的样本数量
    if num_of_samples is None:
        num_of_samples = params.batch_size
    else:
        # NOT PRECISELY COMPUTATION
        # 非精确计算
        # \是续行的意思。
        assert len(src_test_dataloader) < num_of_samples, \
            'The number of samples can not bigger than dataset.'

    # Collect source data.
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        s_labels.append(labels)

        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * params.batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], torch.cat(s_labels)[:num_of_samples], torch.cat(
        s_tags)[:num_of_samples]

    # Collect test data.
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)
        t_labels.append(labels)

        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * params.batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], torch.cat(t_labels)[:num_of_samples], torch.cat(
        t_tags)[:num_of_samples]

    # Compute the embedding of target domain.
    embedding1 = feature_extractor(s_images)
    embedding2 = feature_extractor(t_images)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    if params.use_gpu:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                       embedding2.detach().numpy())))

    utils.plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), 'Domain Adaptation', img_name)


def main(args):
    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    if params.embed_plot_epoch is None:
        params.embed_plot_epoch = args.embed_plot_epoch
    params.lr = args.lr

    if args.save_dir is not None:
        params.save_dir = args.save_dir
    else:
        print('Figures will be saved in ./experiment folder.')

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    if params.fig_mode is not None:
        print('Images from training on source domain:')

        utils.displayImages(src_train_dataloader, imgName='source')

        print('Images from test on target domain:')
        utils.displayImages(tgt_test_dataloader, imgName='target')

    # init models
    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.extractor_dict[model_index]
    class_classifier = params.class_dict[model_index]
    domain_classifier = params.domain_dict[model_index]

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = torch.optim.SGD([{'params': feature_extractor.parameters()},
                                 {'params': class_classifier.parameters()},
                                 {'params': domain_classifier.parameters()}], lr=params.lr, momentum=0.9)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion,
                    domain_criterion,
                    src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
        test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)

        # Plot embeddings periodically.
        if epoch % params.embed_plot_epoch == 0 and params.fig_mode is not None:
            visualize_performance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                                  tgt_test_dataloader, img_name='embedding_' + str(epoch))


def parse_arguments(argv):
    """Command line parse.
        name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
        action - 命令行遇到参数时的动作，默认值是 store。
            store_const，表示赋值为const；
            append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
            append_const，将参数规范中定义的一个值保存到一个列表；
            count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
        nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument 使用 default，
            对于 Optional argument 使用 const；或者是 * 号，表示 0 或多个参数；或者是 + 号表示 1 或多个参数。
        const - action 和 nargs 所需要的常量值。
        default - 不指定参数时的默认值。
        type - 命令行参数应该被转换成的类型。
        choices - 参数可允许的值的一个容器。
        required - 可选参数是否可以省略 (仅针对可选参数)。
        help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
        metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
        dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type=str, default='MNIST', help='Choose source domain.')

    parser.add_argument('--target_domain', type=str, default='MNIST_M', help='Choose target domain.')

    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')

    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')

    parser.add_argument('--embed_plot_epoch', type=int, default=100, help='Epoch number of plotting embeddings.')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
