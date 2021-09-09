# model params
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
learning_rate = 0.1  # 学习率
batch_size = 2  # batch块大小
epochs = 200  # 训练轮数
gamma = 50  # γ
theta = 0.3  # θ
beta = 0.5  # β
# path params
data_root = './data'

save_dir = './experiment'
# 训练参数保存路径
train_params_save_path = save_dir + "/params"

# UJIndoorLoc dataset
dataset_name = "UJIndoorLoc"
dataset_path = data_root + "/UJIndoorLoc/"
