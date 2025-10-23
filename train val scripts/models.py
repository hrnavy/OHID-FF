import torch
import torch.nn as nn
import torchvision.models as models

class LogisticRegression(nn.Module):
    """简单的逻辑回归模型"""
    def __init__(self, input_size=224*224*3):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        x = self.linear(x)
        return x

def get_model(model_name, pretrained=True):
    """获取指定的模型架构"""
    if model_name == 'logistic':
        model = LogisticRegression()
    
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    
    elif model_name == 'shufflenetv2':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    elif model_name == 'inceptionv3':
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
        # InceptionV3 有一个辅助分类器，我们也需要修改它
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    
    else:
        raise ValueError(f"未知模型名称: {model_name}")
    
    # 添加softmax层用于生成类别概率
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    
    return model