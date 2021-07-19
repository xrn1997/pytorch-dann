"""
Main script for models
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from train import test, train, params
from util import utils
import torch
import torch.nn as nn
import models


def main():
    # prepare the source data and target data
    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    # init models
    feature_extractor = models.ME()
    label_predictor_1 = models.M1()
    label_predictor_2 = models.M2()
    label_predictor_3 = models.M3()
    domain_classifier = models.MD()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    label_predictor_1.to(device)
    label_predictor_2.to(device)
    label_predictor_3.to(device)
    domain_classifier.to(device)

    # init criterions 损失函数
    label_criterion = nn.NLLLoss(reduction='sum')
    domain_criterion = nn.NLLLoss()

    # init optimizer 优化器
    optimizer = torch.optim.SGD([{'params': feature_extractor.parameters()},
                                 {'params': label_predictor_1.parameters()},
                                 {'params': domain_classifier.parameters()}], lr=params.learning_rate, momentum=0.9,
                                weight_decay=0.0001)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(feature_extractor, label_predictor_1, domain_classifier, label_criterion,
                    domain_criterion, src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
        test.test(feature_extractor, label_predictor_1, domain_classifier, src_test_dataloader, tgt_test_dataloader)


if __name__ == '__main__':
    main()
