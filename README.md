# 微调预训练的卷积神经网络实现鸟类识别

## 项目简介

这个项目用于处理数据集并微调预训练模型以进行图像分类任务。

## 数据处理（data.py）

数据处理脚本用于从数据集中提取图像并将其分为训练集和测试集。

### 数据集路径

数据集位于 `D:\\PycharmProjects\\Image_Cls\\CUB_200_2011` 目录下。

### 使用方法

运行 `data.py` 脚本以执行数据处理。

## 微调模型（fintune.py）

微调模型脚本用于加载预训练模型并在新数据集上进行微调。

### 使用方法

运行 `fintune.py` 脚本以执行微调模型的训练和评估。

## 依赖

- Python 3.x
- PyTorch
- torchvision
- pandas

## 使用指南

1. 克隆项目仓库：`git clone https://github.com/yourusername/yourproject.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 执行数据处理：`python data.py`
4. 执行微调模型：`python fintune.py`
5. 调整如下代码实现全模型训练
   ```python
   model = models.resnet50(pretrained=False)
6. 在如下部分可以调整超参数的实验
   ```python
   def run_experiments():
    configurations = [
        #{'batch_size': 64, 'learning_rate': 0.002},
        {'batch_size': 16, 'learning_rate': 0.01},
       #{'batch_size': 32, 'learning_rate': 0.0015}
    ]


