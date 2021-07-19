# utility params
fig_mode = 'save'  # 有两个选择，一是save，二是display。
embed_plot_epoch = 10

# model
training_mode = 'dann'  # 有两个选择，一是dann，二是source

# model params
use_gpu = True  # 是否使用GPU
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
learning_rate = 0.1  # 学习率
batch_size = 512  # batch块大小
epochs = 200  # 训练轮数
gamma = 10  # γ
theta = 1  # θ

# path params
data_root = './data'

mnist_path = data_root + '/MNIST'
mnist_m_path = data_root + '/MNIST_M'

save_dir = './experiment'
# 训练参数保存路径
dann_params_save_path = save_dir + "/dann"
source_params_save_path = save_dir + "/source"

# dataset type
source_domain = 'MNIST'
target_domain = 'MNIST_M'
