# model params
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
learning_rate = 0.1  # 学习率
batch_size = 2  # batch块大小
epochs = 200  # 训练轮数
gamma = 10  # γ
theta = 1  # θ

# path params
data_root = './data'

save_dir = './experiment'
# 训练参数保存路径
train_params_save_path = save_dir + "/tampere"

# Tampere dataset
dataset_name = "Tampere"
dataset_path = data_root + "/Tampere/DISTRIBUTED_OPENSOURCE_version2/FINGERPRINTING_DB/"

# UJIndoorLoc dataset
# dataset_name = "UJIndoorLoc"
# dataset_path = data_root + "/UJIndoorLoc/"
