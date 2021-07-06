from models import models

# utility params
fig_mode = None
embed_plot_epoch = 100

# model
training_mode = 'dann'

# model params
use_gpu = True  # 是否使用GPU
# dataset_mean = (0.5, 0.5, 0.5) 报错
# dataset_std = (0.5, 0.5, 0.5) 报错
dataset_mean = 0.5  # 均值
dataset_std = 0.5  # 标准差
learning_rate = 0.01  # 学习率
batch_size = 512  # batch块大小
epochs = 100  # 训练轮数
gamma = 10  # γ
theta = 1  # θ

# path params
data_root = './data'

mnist_path = data_root + '/MNIST'
mnist_m_path = data_root + '/MNIST_M'
svhn_path = data_root + '/SVHN'
synth_path = data_root + '/SynthDigits'

save_dir = './experiment'

# specific dataset params
extractor_dict = {'MNIST_MNIST_M': models.Extractor(),
                  'SVHN_MNIST': models.SVHN_Extractor(),
                  'SynDig_SVHN': models.SVHN_Extractor()}

class_dict = {'MNIST_MNIST_M': models.Class_classifier(),
              'SVHN_MNIST': models.SVHN_Class_classifier(),
              'SynDig_SVHN': models.SVHN_Class_classifier()}

domain_dict = {'MNIST_MNIST_M': models.Domain_classifier(),
               'SVHN_MNIST': models.SVHN_Domain_classifier(),
               'SynDig_SVHN': models.SVHN_Domain_classifier()}

# dataset type
source_domain = 'MNIST'
target_domain = 'MNIST_M'
