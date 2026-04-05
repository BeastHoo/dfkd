# FedDistill：基于分层知识聚合的无数据联邦蒸馏系统

本项目在 FedIOD（AAAI 2024）基础上进行改进，引入
**基于 Sinkhorn 距离的客户端聚类**与**分层知识聚合机制**，
面向异构客户端与非IID数据分布场景，实现保护隐私的跨客户端知识蒸馏。

---

## 核心改进点

| 模块 | FedIOD 原始 | 本项目改进 |
|------|------------|-----------|
| 客户端分组 | 无 | Sinkhorn 距离 + K-means 聚类 |
| 知识聚合 | 简单 π_k 加权平均 | 簇内熵加权 + 簇间温度-规模平衡加权 |
| 软标签生成 | 单层加权 q̄ | 分层聚合全局软标签 Z |

---

## 项目结构
```
fed_distill/
├── config.yaml              # 所有超参数配置
├── main.py                  # 训练入口
├── server.py                # 联邦训练主循环
├── data/
│   ├── data_utils.py        # 数据加载与非IID划分
│   └── partitions/          # 自动缓存的.npy划分文件
├── models/
│   ├── generator.py         # 生成器 G
│   ├── discriminator.py     # 判别器 D_k
│   └── task_model.py        # 本地模型与中心模型定义
├── clustering/
│   ├── sinkhorn.py          # Sinkhorn距离计算
│   └── clustering.py        # 聚类与簇头选取
├── client/
│   └── client.py            # 客户端逻辑
├── aggregation/
│   └── aggregation.py       # 分层聚合
├── trainer/
│   └── trainer.py           # 损失计算与参数更新
├── utils/
│   ├── logger.py            # 日志
│   └── metrics.py           # 指标记录
├── checkpoints/             # 训练快照（自动生成）
├── teacher_ckpts/           # 客户端预训练模型（需外部提供）
│   ├── client_0.pt
│   ├── client_1.pt
│   └── ...
└── requirements.txt
```

---

## 环境配置
```bash
pip install -r requirements.txt
```

主要依赖：Python 3.9+、PyTorch 2.x、torchvision、scikit-learn、numpy、pyyaml

---

## 使用前准备

### 1. 准备客户端预训练模型

本项目**不包含**客户端模型预训练逻辑。
请使用配套的预训练项目完成各客户端模型训练后，
将 .pt 文件按以下命名规则放入 `teacher_ckpts/` 目录：
```
teacher_ckpts/
├── client_0.pt   # 客户端 0 的模型权重
├── client_1.pt   # 客户端 1 的模型权重
└── ...           # 共 num_clients 个文件
```

.pt 文件内容为 `model.state_dict()`，加载时会自动匹配模型结构。

### 2. 配置超参数

编辑 `config.yaml`，重点关注以下字段：
```yaml
dataset: cifar10              # 数据集：cifar10 | cifar100
num_clients: 10               # 客户端数量（需与 .pt 文件数量一致）
teacher_ckpt_dir: ./teacher_ckpts
data_dir: ./data/raw
partition_dir: ./data/partitions
dirichlet_alpha: 0.5          # 非IID程度，越小越异构
num_clusters: 3               # 聚类簇数
```

---

## 运行训练
```bash
# 从头开始训练
python main.py

# 从指定 checkpoint 恢复训练
python main.py --resume checkpoints/round_0050.pt

# 指定配置文件
python main.py --config config.yaml
```

---

## 数据集划分缓存机制

首次运行时，系统会自动对数据集进行 Dirichlet 非IID划分，
并将划分结果保存为 `.npy` 文件（路径由 `partition_dir` 指定），
文件名包含数据集名称、alpha值、客户端数量和随机种子，例如：
```
data/partitions/cifar10_alpha0.5_clients10_seed42.npy
```

后续运行时若检测到该文件存在，将直接加载而不重新划分，
确保不同运行之间数据划分一致，实验结果可复现。

如需重新划分，删除对应 `.npy` 文件后重新运行即可。

---

## Checkpoint 机制

训练过程中每隔 `save_ckpt_every` 轮自动保存一次快照：
```
checkpoints/
├── round_0010.pt   # 第10轮快照
├── round_0020.pt   # 第20轮快照
├── ...
└── latest.pt       # 最新轮次快照（每轮更新）
```

每个 checkpoint 包含：生成器、中心模型、所有判别器的权重及优化器状态、
当前轮次、测试集准确率、聚类结果。

---

## 核心算法说明

### Sinkhorn 聚类
基于熵正则化最优传输计算客户端间数据分布距离，
使用 log 域迭代保证数值稳定性，
对距离矩阵执行 K-means 聚类将客户端分为 M 个簇。

### 分层知识聚合
- **簇内聚合**：以 `exp(-H(q_k))` 为权重，预测越确定的客户端贡献越大
- **簇间聚合**：联合簇规模指数 γ 与聚合温度 T_agg，
  平衡高质量知识利用与跨簇均衡融合

### 生成器三路损失
| 损失项 | 作用 |
|--------|------|
| L_conf | 最小化客户端预测熵，生成语义清晰的数据 |
| L_div  | 最大化全局软标签熵，生成多样化数据 |
| L_adv  | 对抗判别器，提升生成数据真实性 |

---

## 实验评估

每 10 轮在完整测试集上评估中心模型 S 的 top-1 accuracy，
结果自动记录至 `utils/metrics.py` 并打印至终端。

---

## 引用

本项目是对FedIOD上进行的改进：
```bibtex
@article{gong2023federated,
  title={Federated Learning via Input-Output Collaborative Distillation},
  author={Gong, Xuan and Li, Shanglin and others},
  journal={arXiv preprint arXiv:2312.14478},
  year={2023}
}
```