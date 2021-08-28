"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np

from trains import params


def test(feature_extractor,
         label_predictor_1, label_predictor_2, label_predictor_3,
         domain_classifier,
         source_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param label_predictor_1: 标签预测器1
    :param label_predictor_2: 标签预测器2
    :param label_predictor_3: 标签预测器3
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    label_predictor_1.eval()
    label_predictor_2.eval()
    label_predictor_3.eval()
    domain_classifier.eval()

    position_correct = 0.0
    domain_correct = 0.0

    for batch_idx, s_data in enumerate(source_dataloader):
        # setup hyper_parameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        source_rss, source_position, position_label, domain_label = s_data
        if torch.cuda.is_available():
            source_rss, source_position, position_label, domain_label = Variable(source_rss.cuda()), Variable(
                source_position.cuda()), Variable(position_label.cuda()), Variable(domain_label.cuda())
        else:
            source_rss, source_position, position_label, domain_label = Variable(source_rss), Variable(
                source_position), Variable(position_label), Variable(domain_label)

        src_feature = feature_extractor(source_rss)

        # compute the class loss of src_feature
        class_preds_1 = label_predictor_1(src_feature)
        class_preds_2 = label_predictor_2(src_feature)
        class_preds_3 = label_predictor_3(src_feature)
        pred1 = class_preds_1.data.max(1, keepdim=True)[1]
        pred2 = class_preds_2.data.max(1, keepdim=True)[1]
        pred3 = class_preds_3.data.max(1, keepdim=True)[1]
        position_correct += pred1.eq(position_label.data.view_as(pred1)).cpu().sum()

        domain_preds = domain_classifier(src_feature, constant)
        preds_first = domain_preds[:, 0:3].data.max(1, keepdim=True)[1]  # 时间预测最大值位置
        preds_second = domain_preds[:, 3:].data.max(1, keepdim=True)[1]  # 设备预测最大值位置
        labels_first = domain_label[:, 0:3].data.max(1, keepdim=True)[1]  # 时间实际最大值位置
        labels_second = domain_label[:, 3:].data.max(1, keepdim=True)[1]  # 设备实际最大值位置
        time_result = preds_first.eq(labels_first)
        device_result = preds_second.eq(labels_second)
        domain_correct += (time_result * device_result).cpu().sum()

    print('Position Accuracy: {}/{} ({:.4f}%)'.format(position_correct, len(source_dataloader.dataset),
                                                      100. * float(position_correct) / len(source_dataloader.dataset)))
    print('Domain Accuracy: {}/{} ({:.4f}%)'.format(domain_correct, len(source_dataloader.dataset),
                                                    100. * float(domain_correct) / len(source_dataloader.dataset)))
