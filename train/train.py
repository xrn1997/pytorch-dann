import os

from torch.autograd import Variable
from train import params
from util import utils
import torch
import numpy as np


def train(feature_extractor, label_predictor, domain_classifier, class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch):
    """
    Execute target domain adaptation

    :param feature_extractor:
    :param label_predictor:
    :param domain_classifier:
    :param class_criterion:
    :param domain_criterion:
    :param source_dataloader:
    :param target_dataloader:
    :param optimizer:
    :param epoch:
    :return:
    """
    # setup models
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()

    # 加载已保存的参数
    if not os.path.exists(params.dann_params_save_path):
        os.mkdir(params.dann_params_save_path)
    if os.path.exists(params.dann_params_save_path + "/fe.pth"):
        feature_extractor.load_state_dict(torch.load(params.dann_params_save_path + "/fe.pth"))
    if os.path.exists(params.dann_params_save_path + "/dc.pth"):
        domain_classifier.load_state_dict(torch.load(params.dann_params_save_path + "/dc.pth"))
    if os.path.exists(params.dann_params_save_path + "/lp.pth"):
        label_predictor.load_state_dict(torch.load(params.dann_params_save_path + "/lp.pth"))
    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)
    for batch_idx, (s_data, t_data) in enumerate(zip(source_dataloader, target_dataloader)):
        # setup hyper_parameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

        # prepare the data
        input1, label1 = s_data
        input2, label2 = t_data
        size = min((input1.shape[0], input2.shape[0]))
        input1, label1 = input1[0:size, :, :, :], label1[0:size]
        input2, label2 = input2[0:size, :, :, :], label2[0:size]
        if torch.cuda.is_available():
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            input2, label2 = Variable(input2), Variable(label2)

        # setup optimizer
        optimizer = utils.optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # prepare domain labels
        if torch.cuda.is_available():
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
            target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))
        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)
        tgt_feature = feature_extractor(input2)

        # compute the class loss of src_feature
        class_preds = label_predictor(src_feature)
        class_loss = class_criterion(class_preds, label1)

        # compute the domain loss of src_feature and target_feature
        tgt_preds = domain_classifier(tgt_feature, constant)
        src_preds = domain_classifier(src_feature, constant)
        tgt_loss = domain_criterion(tgt_preds, target_labels)
        src_loss = domain_criterion(src_preds, source_labels)
        domain_loss = tgt_loss + src_loss

        loss = class_loss + params.theta * domain_loss
        loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(input2), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
                domain_loss.item()
            ))

    torch.save(feature_extractor.state_dict(), params.dann_params_save_path + "/fe.pth")
    torch.save(domain_classifier.state_dict(), params.dann_params_save_path + "/dc.pth")
    torch.save(label_predictor.state_dict(), params.dann_params_save_path + "/lp.pth")
