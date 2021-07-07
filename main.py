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
    params.embed_plot_epoch = args.embed_plot_epoch
    params.learning_rate = args.lr
    params.save_dir = args.save_dir

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    # 如果fig_mode不为空，会在save_dir中保存前8个（默认）图片，保存为1张。
    if params.fig_mode is not None:
        print('Images from training on source domain:')
        utils.display_images(src_train_dataloader, img_name='source')
        print('Images from test on target domain:')
        utils.display_images(tgt_test_dataloader, img_name='target')

    # init models
    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.feature_extractor_dict[model_index]
    class_classifier = params.label_predictor_dict[model_index]
    domain_classifier = params.domain_classifier_dict[model_index]

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
                                 {'params': domain_classifier.parameters()}], lr=params.learning_rate, momentum=0.9)

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
    """
    执行参数 \r\n
    更多提示请执行 \r\n
    python main.py -h

    :param argv: arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type=str, default=params.source_domain, help='Choose source domain. 选择源域。')

    parser.add_argument('--target_domain', type=str, default=params.target_domain, help='Choose target domain. 选择目标域。')

    parser.add_argument('--fig_mode', type=str, default=params.fig_mode, help='Plot experiment '
                                                                              'figures.有两个选项，一是save，二是display（不保存）。')

    parser.add_argument('--save_dir', type=str, default=params.save_dir, help='Path to save plotted images. ')

    parser.add_argument('--training_mode', type=str, default=params.training_mode, help='Choose a mode to train the '
                                                                                        'model. 训练模型')

    parser.add_argument('--max_epoch', type=int, default=params.epochs, help='The max number of epochs.最大训练轮数。')

    parser.add_argument('--embed_plot_epoch', type=int, default=params.embed_plot_epoch, help='Epoch number of '
                                                                                              'plotting embeddings.')

    parser.add_argument('--lr', type=float, default=params.learning_rate, help='Learning rate. 学习率。')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
