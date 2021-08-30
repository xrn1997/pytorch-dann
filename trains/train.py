import os

from torch.autograd import Variable

from trains import params
from tool import utils
import torch
import numpy as np


def train(feature_extractor,
          label_predictor_1, label_predictor_2, label_predictor_3,
          domain_classifier,
          label_criterion, domain_criterion,
          source_dataloader,
          optimizer,
          epoch
          ):
    """
    Execute target1 domain adaptation

    :param feature_extractor: 特征提取器
    :param label_predictor_1: 标签预测器1
    :param label_predictor_2: 标签预测器2
    :param label_predictor_3: 标签预测器3
    :param domain_classifier: 域鉴别器
    :param label_criterion: 标签预测损失函数
    :param domain_criterion: 域损失函数
    :param source_dataloader: 源域数据
    :param optimizer: 优化器
    :param epoch: 训练轮数
    :return: 无返回值
    """
    # setup models
    feature_extractor.train()
    label_predictor_1.train()
    label_predictor_2.train()
    label_predictor_3.train()
    domain_classifier.train()

    # 加载已保存的参数
    if not os.path.exists(params.save_dir):
        os.mkdir(params.save_dir)
    if not os.path.exists(params.train_params_save_path):
        os.mkdir(params.train_params_save_path)
    if os.path.exists(params.train_params_save_path + "/fe.pth"):
        feature_extractor.load_state_dict(torch.load(params.train_params_save_path + "/fe.pth"))
    if os.path.exists(params.train_params_save_path + "/dc.pth"):
        domain_classifier.load_state_dict(torch.load(params.train_params_save_path + "/dc.pth"))
    if os.path.exists(params.train_params_save_path + "/lp1.pth"):
        label_predictor_1.load_state_dict(torch.load(params.train_params_save_path + "/lp1.pth"))
        label_predictor_2.load_state_dict(torch.load(params.train_params_save_path + "/lp2.pth"))
        label_predictor_3.load_state_dict(torch.load(params.train_params_save_path + "/lp3.pth"))

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)
    for batch_idx, s_data in enumerate(source_dataloader):
        # setup hyper_parameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-params.gamma * p)) - 1
        # prepare the data
        source_rss, source_position, position_label, domain_label = s_data
        if torch.cuda.is_available():
            source_rss, source_position, position_label, domain_label = Variable(source_rss.cuda()), Variable(
                source_position.cuda()), Variable(position_label.cuda()), Variable(domain_label.cuda())
        else:
            source_rss, source_position, position_label, domain_label = Variable(source_rss), Variable(
                source_position), Variable(position_label), Variable(domain_label)
        # setup optimizer
        optimizer = utils.optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()
        # compute the output of source domain and target1 domain
        src_feature = feature_extractor(source_rss)

        # compute the class loss of src_feature
        class_preds_1 = label_predictor_1(src_feature)
        class_preds_2 = label_predictor_2(src_feature)
        class_preds_3 = label_predictor_3(src_feature)

        l1_loss = label_criterion(class_preds_1, position_label)
        l2_loss = label_criterion(class_preds_2, position_label)
        l3_loss = label_criterion(class_preds_3, position_label)
        class_loss = l1_loss + l2_loss + l3_loss

        # compute the domain loss of src_feature and target_feature
        src_preds = domain_classifier(src_feature, constant)
        domain_loss = domain_criterion(src_preds, domain_label)

        loss = class_loss + params.theta * domain_loss
        loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(source_rss),
                len(source_dataloader.dataset),
                100. * batch_idx / len(source_dataloader),
                loss.item(),
                class_loss.item(),
                domain_loss.item()
                ))

    torch.save(feature_extractor.state_dict(), params.train_params_save_path + "/fe.pth")
    torch.save(domain_classifier.state_dict(), params.train_params_save_path + "/dc.pth")
    torch.save(label_predictor_1.state_dict(), params.train_params_save_path + "/lp1.pth")
    torch.save(label_predictor_2.state_dict(), params.train_params_save_path + "/lp2.pth")
    torch.save(label_predictor_3.state_dict(), params.train_params_save_path + "/lp3.pth")


# 测试用代码
if __name__ == '__main__':
    if not os.path.exists(params.save_dir):
        os.mkdir(params.save_dir)
    if not os.path.exists(params.train_params_save_path):
        os.mkdir(params.train_params_save_path)
