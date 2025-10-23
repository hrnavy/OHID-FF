# 数据集配置
DATASET_PATH = './data/ohid_ff/'
IMAGE_SIZE = 512  # 原始图像大小
RESIZE_TO = 224   # 模型输入大小
TRAIN_SIZE = 597  # 训练集图像数量
TEST_SIZE = 600   # 测试集图像数量

# 训练配置
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_RUNS = 3      # 每个模型训练的次数

# 模型配置
MODELS = [
    'resnet18',
    'resnet50',
    'vgg16',
    'logistic',
    'mobilenetv2',
    'densenet121',
    'shufflenetv2',
    'inceptionv3'
]

# 数据增强配置
AUGMENTATION = True

# 保存路径
CHECKPOINT_DIR = './checkpoints/'
RESULTS_DIR = './results/'