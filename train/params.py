import models.feature_extractor as mf
import models.label_predictor as ml
import models.domain_classifier as md

# utility params
fig_mode = 'save'  # 有两个选择，一是save，二是display。
embed_plot_epoch = 10

# model
training_mode = 'dann'

# model params
use_gpu = True  # 是否使用GPU
# dataset_mean = (0.5, 0.5, 0.5) 报错
# dataset_std = (0.5, 0.5, 0.5) 报错
dataset_mean = [0.5]  # 均值
dataset_std = [0.5]  # 标准差
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
# 训练参数保存路径
dann_params_save_path = save_dir + "/dann"
source_params_save_path=save_dir+"/source"

# specific dataset params
feature_extractor_dict = {'MNIST_MNIST_M': mf.FeatureExtractor(),
                          'SVHN_MNIST': mf.SVHNFeatureExtractor(),
                          'SynDig_SVHN': mf.SVHNFeatureExtractor()}  # 这个地方打个问号，我不知道SynDig和SVHN数据集的区别。

label_predictor_dict = {'MNIST_MNIST_M': ml.LabelPredictor(),
                        'SVHN_MNIST': ml.SVHNLabelPredictor(),
                        'SynDig_SVHN': ml.SVHNLabelPredictor()}

domain_classifier_dict = {'MNIST_MNIST_M': md.DomainClassifier(),
                          'SVHN_MNIST': md.SVHNDomainClassifier(),
                          'SynDig_SVHN': md.SVHNDomainClassifier()}

# dataset type
source_domain = 'MNIST'
target_domain = 'MNIST_M'
