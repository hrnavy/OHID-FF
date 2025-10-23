import os
import shutil
import random
import pandas as pd
from config import *

def create_sample_dataset():
    """创建示例数据集结构（用于演示）"""
    # 创建目录结构
    os.makedirs(os.path.join(DATASET_PATH, 'fire'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, 'non_fire'), exist_ok=True)
    
    print(f"已创建数据集目录结构在: {DATASET_PATH}")
    print("请将您的火灾图像放入 fire 文件夹，非火灾图像放入 non_fire 文件夹")
    print(f"数据集应包含总共 {TRAIN_SIZE + TEST_SIZE} 张图像")
    print(f"其中火灾图像 {int(0.54 * (TRAIN_SIZE + TEST_SIZE))} 张，非火灾图像 {int(0.46 * (TRAIN_SIZE + TEST_SIZE))} 张")

if __name__ == "__main__":
    create_sample_dataset()