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
feature_extractor_dict = {'MNIST_MNIST_M': models.FeatureExtractor(),
                          'SVHN_MNIST': models.SVHNFeatureExtractor(),
                          'SynDig_SVHN': models.SVHNFeatureExtractor()}  # 这个地方打个问号，我不知道SynDig和SVHN数据集的区别。

label_predictor_dict = {'MNIST_MNIST_M': models.LabelPredictor(),
                        'SVHN_MNIST': models.SVHNLabelPredictor(),
                        'SynDig_SVHN': models.SVHNLabelPredictor()}

domain_classifier_dict = {'MNIST_MNIST_M': models.DomainClassifier(),
                          'SVHN_MNIST': models.SVHNDomainClassifier(),
                          'SynDig_SVHN': models.SVHNDomainClassifier()}

# dataset type
source_domain = 'MNIST'
target_domain = 'MNIST_M'
