# diagnose.py
# 诊断教师模型质量、生成数据分布、聚合软标签质量
# 运行：python diagnose.py

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from data.data_utils import prepare_data
from models.generator import Generator
from models.task_model import load_teacher_model

cfg = OmegaConf.load("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("诊断 1：各客户端教师模型在测试集上的 Accuracy")
print("=" * 60)
_, test_loader, _, pi_k_list = prepare_data(cfg)

for k in range(cfg.num_clients):
    try:
        teacher = load_teacher_model(k, cfg.teacher_ckpt_dir,
                                     cfg.num_classes, cfg.image_channels, device)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = teacher(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"  Client {k:2d} ({cfg.teacher_ckpt_dir}/client_{k}.pt): "
              f"Acc={acc:.4f}  π_k={pi_k_list[k]:.4f}")
    except Exception as e:
        print(f"  Client {k:2d}: 加载失败 → {e}")

print()
print("=" * 60)
print("诊断 2：教师模型对随机噪声图像的预测分布（验证是否退化）")
print("=" * 60)
noise = torch.randn(64, cfg.image_channels, cfg.image_size, cfg.image_size).to(device)
# 同时测试生成器输出
G = Generator(cfg.latent_dim, cfg.image_channels).to(device)
with torch.no_grad():
    z = torch.randn(64, cfg.latent_dim, device=device)
    x_fake = G(z)

for k in range(cfg.num_clients):
    try:
        teacher = load_teacher_model(k, cfg.teacher_ckpt_dir,
                                     cfg.num_classes, cfg.image_channels, device)
        with torch.no_grad():
            # 测试对生成器输出的预测
            logits = teacher(x_fake)
            probs  = F.softmax(logits / cfg.temperature, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
            pred_cls = logits.argmax(1)
            unique_cls = pred_cls.unique().numel()
            max_prob = probs.max(-1).values.mean().item()
        print(f"  Client {k:2d}: 平均熵={entropy:.4f}  "
              f"预测类别数={unique_cls}/{cfg.num_classes}  "
              f"平均最大概率={max_prob:.4f}")
    except Exception as e:
        print(f"  Client {k:2d}: 加载失败 → {e}")

print()
print("=" * 60)
print("诊断 3：Z（全局软标签）的质量")
print("  若所有客户端 T_k 对 fake data 都预测同一类，Z 退化为 one-hot")
print("  若 T_k 质量差，预测随机，Z 退化为均匀分布（高熵=无信息）")
print("=" * 60)
print("→ 请对照诊断2的熵值判断：")
print("  熵接近 log(10)=2.30 → T_k 预测近乎随机，软标签无信息")
print("  熵接近 0            → T_k 预测极度自信（可能过拟合或崩溃）")
print("  熵在 0.5~1.5 之间   → 正常范围")