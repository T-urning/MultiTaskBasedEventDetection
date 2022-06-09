# 面向 DuEE 数据集实现事件检测模型 M-RoBERTa-CRF

## DuEE 数据集
DuEE 是由百度发布的中文事件抽取数据集，包含 65 种事件类型，共 17,000 个标注句子，以及 20,000 个标注事件。有关 DuEE 更详细的介绍可见[这里](https://ai.baidu.com/broad/introduction?dataset=duee)。

## 环境要求
- PyTorch 1.4.0+
- Transformer 4.13.0+

## 项目结构
- data: 保存原始标注数据或预处理后的数据
- outputs: 保存模型训练过程中产生的文件
    - checkpoints: 模型权重等文件
    - logs: 训练日志
    - record_as_imgs: 模型损失与指标变化趋势图
    - badcases: 模型预测表现较差时对应的样本
- models: 保存模型定义代码
- pretrained_models: 保存预训练模型文件，该项目主要使用由 Cui 提供的 [RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD) 中文预训练模型
- 其他
    - data_exploration.ipynb: 数据集探索
    - data_prepare.py: 数据预处理
    - losses.py: 定义损失函数
    - utils.py: 各种工具代码
    - **sequence_labeling.py**: 常规模型的训练与评估代码
    - **sequence_labeling_multi_task.py**: 基于多任务学习的模型的训练与评估代码
    - plot_metrics.py: 绘制指标图表

## 模型介绍
本项目主要提供基于多任务学习的事件检测模型 M-RoBERTa-CRF 的实现，该模型引入事件类型分类（文本分类）任务作为辅助任务为事件检测任务更好地引入文本上下文信息。具体介绍可参阅论文：

>@article{MultiTaskED,
author = {Xia, Jing and Li, Xiaolong and Tan, Yongbin and Zhang, Wu and Li, Dajun and Xiong, Zhengkun},
title = {Event Detection via Context Understanding Based on Multi-Task Learning},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2375-4699},
url = {https://doi.org/10.1145/3529388},
doi = {10.1145/3529388}}

为做对比，本项目还实现了如下模型，具体实现可见 `models` 文件夹：
- DMCNN
- BiLSTM
- BiLSTM-CRF
- RoBERTa
- RoBERTa-CRF
- M-RoBERa

## 如何使用
可直接从如下百度网盘链接下载预训练模型文件，再解压到 `pretrained_models\bert_wwm_ext` 文件夹下：

> 链接：https://pan.baidu.com/s/17bNjOXU1y1t36_6C9XBcWg?pwd=wwne 
提取码：wwne

进行数据预处理：

``
python data_prepare.py
``

训练并评估基于多任务学习的模型：

``
python sequence_labeling_multi_task.py 
``

训练并评估其它模型：

``
python sequence_labeling.py 
``