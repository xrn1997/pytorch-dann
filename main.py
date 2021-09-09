"""
主程序
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from trains import test, train, params
from tool import utils
import torch
import torch.nn as nn
import models


def main():
    # prepare the source data and target1 data
    train_dataloader, domain_size, position_size, ap_len = utils.get_data_loader(params.dataset_name,
                                                                                 params.dataset_path,
                                                                                 train=True)
    test_dataloader, _, _, _ = utils.get_data_loader(params.dataset_name,
                                                     params.dataset_path,
                                                     train=False)
    # init models
    feature_extractor = models.ME()

    label_predictor_1 = models.M1(ap_len, position_size)
    para = sum([np.prod(list(p.size())) for p in label_predictor_1.parameters()])
    print('Model {} : params: {:4f}M'.format("label_predictor_1", para / 500 / 500))
    label_predictor_2 = models.M2(ap_len, position_size)
    label_predictor_3 = models.M3(ap_len, position_size)

    domain_classifier = models.MD(ap_len, domain_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    label_predictor_1.to(device)
    label_predictor_2.to(device)
    label_predictor_3.to(device)
    domain_classifier.to(device)

    # init criterions 损失函数
    label_criterion = nn.NLLLoss()
    domain_criterion = models.OneHotNLLLoss()
    location_criterion = nn.NLLLoss()

    # init optimizer 优化器
    optimizer = torch.optim.SGD([{'params': feature_extractor.parameters()},
                                 {'params': label_predictor_1.parameters()},
                                 {'params': label_predictor_2.parameters()},
                                 {'params': label_predictor_3.parameters()},
                                 {'params': domain_classifier.parameters()}],
                                lr=params.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0001)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(feature_extractor,  # 特征提取
                    label_predictor_1, label_predictor_2, label_predictor_3,  # 标签预测
                    domain_classifier,  # 域鉴别
                    label_criterion, domain_criterion,location_criterion,  # 损失
                    train_dataloader,  # dataloader
                    optimizer,  # 优化器
                    epoch  # 轮数`
                    )
    test.test(feature_extractor,  # 特征提取
              label_predictor_1, label_predictor_2, label_predictor_3,  # 标签预测
              domain_classifier,  # 域鉴别
              test_dataloader,  # dataloader
              )


if __name__ == '__main__':
    main()
