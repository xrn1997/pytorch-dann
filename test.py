import torch.nn as nn
import torch
import models

if __name__ == '__main__':
    # 交叉熵
    data = torch.Tensor([[-0.2733, 0.3222, 0.2605],
                         [1.5393, 1.1688, -0.0975],
                         [0.3943, 0.5172, -0.9425]])  # 一个3*3的矩阵
    print(data)
    '''
    程序运行的一次结果
    tensor([[-0.2733,  0.3222,  0.2605],
            [ 1.5393,  1.1688, -0.0975],
            [ 0.3943,  0.5172, -0.9425]])
    '''
    sm = nn.Softmax(dim=1)  # 按行 softmax
    print(sm(data))
    '''
    程序运行的一次结果
    tensor([[0.2213, 0.4014, 0.3774],
            [0.5305, 0.3663, 0.1032],
            [0.4178, 0.4725, 0.1098]])
    '''
    print(torch.log(sm(data)))  # log(softmax)
    '''
    程序运行的一次结果
    tensor([[-1.5084, -0.9129, -0.9746],
            [-0.6339, -1.0044, -2.2707],
            [-0.8728, -0.7498, -2.2095]])
    '''
    slm = nn.LogSoftmax(dim=1)
    print(slm(data))  # LogSoftmax
    '''
    程序运行的一次结果
    tensor([[-1.5084, -0.9129, -0.9746],
            [-0.6339, -1.0044, -2.2707],
            [-0.8728, -0.7498, -2.2095]])
    '''
    # 结论：nn.LogSoftmax = torch.log(nn.Softmax)

    loss = nn.NLLLoss(reduction='sum')
    target1 = torch.tensor([0, 1, 2])  # 随便写一个目标tensor
    print(loss(data, target1))  # NLLLoss原始损失
    '''
    程序运行的一次结果
    tensor(0.0157)
    (0.2733+0.9425-1.1688)/3 ≈ 0.0157
    '''
    print(loss(slm(data), target1))  # NLLLoss对LogSoftmax处理后的数据的损失
    '''
    程序运行的一次结果
    tensor(1.5741) 
    (1.5084+1.0044+2.2095)/3=1.5741
    '''
    loss2 = nn.CrossEntropyLoss()
    print(loss2(data, target1))  # CrossEntropyLoss损失
    '''
    程序运行的一次结果
    tensor(1.5741)
    '''
    # 结论：nn.CrossEntropyLoss(input, target1) = nn.NLLLoss(nn.LogSoftmax(input), target1)

    target2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one-hot 标签
    custom_loss = -torch.sum(slm(data) * target2) / 3
    test2 = models.OneHotNLLLoss(reduction="sum")
    custom_loss2 = test2(slm(data), target2)
    print(custom_loss)
    print(custom_loss2)
    '''
    程序运行的一次结果
    tensor(1.5741)
    '''
    # Tensor与tensor的区别
    A = torch.Tensor([1, 2])  # Tensor float32
    print(A.dtype)

    B = torch.tensor([1, 2])  # tensor int
    print(B.dtype)

    # 测试字典
    dic = {}
    data = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    position_label = []
    i = 0
    for d in data:
        if d not in dic.values():
            dic[i] = d
            position_label.append(i)
            i = i + 1
        else:
            position_label.append(list(dic.keys())[list(dic.values()).index(d)])
    print(dic)
    print(position_label)

    # tensor 测试
    c = torch.tensor([1, 2, 3])
    print(c.shape)
    d = torch.tensor([4])
    print(d.shape)
    t = torch.cat((c, d), 0)
    print(t)
