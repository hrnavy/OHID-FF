import os
import torch
import numpy as np
import random
from config import *
from models import get_model
from data_loader import get_data_loaders
from trainer import train_model
from evaluate_all import evaluate_all_models

def setup_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_all_models():
    """训练所有模型，每个模型训练多次"""
    # 创建必要的目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for model_name in MODELS:
        print(f"\n开始训练模型: {model_name}")
        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n  运行 {run_id}/{NUM_RUNS}")
            setup_seed(42 + run_id)  # 使用不同的种子
            
            # 获取数据加载器
            train_loader, test_loader = get_data_loaders()
            
            # 创建模型
            if model_name == 'logistic':
                model = get_model(model_name, pretrained=False)  # 逻辑回归不需要预训练
            else:
                model = get_model(model_name, pretrained=True)
            
            # 训练模型
            print(f"  开始训练 {model_name} (运行 {run_id})")
            train_model(model, train_loader, test_loader, model_name, run_id)

if __name__ == "__main__":
    # 训练所有模型
    train_all_models()
    
    # 评估所有模型并生成结果报告
    print("\n开始评估所有模型...")
    results = evaluate_all_models()
    
    print("\n模型评估完成！")
    print("结果已保存到:", RESULTS_DIR)