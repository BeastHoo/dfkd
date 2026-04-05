# models/generator.py
# 功能：生成器 G，从潜在向量 ω ~ N(0,I) 生成伪数据 x = G(ω)
# 结构：FC → reshape(256,4,4) → 转置卷积×3 → Tanh
# 对应阶段 1：合成数据生成与分发
# ============================================================

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN 风格生成器。

    输入：潜在向量 ω ∈ R^{latent_dim}，从标准正态分布采样
    输出：伪图像 x，shape = (B, image_channels, 32, 32)

    输出归一化说明：
      - 转置卷积输出经 Tanh 后范围为 [-1, 1]
      - 客户端教师模型 T_k 在 CIFAR 归一化数据上训练
        CIFAR-10 归一化：mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)
        归一化后真实数据范围约 [-2.4, 2.7]
      - 若直接将 [-1,1] 的 fake data 送入 T_k，属于分布外输入，T_k 输出退化
      - 解决方案：在 Tanh 输出后对每个通道做仿射变换，
        将 [-1,1] 映射到与 CIFAR 归一化数据相同的统计范围
        pixel_normalized = (tanh_out * std_cifar) + mean_cifar_normalized
        其中 mean_cifar_normalized = mean/std（归一化后的均值）
    """

    # CIFAR-10/100 归一化参数（与 data_utils.py 中的 get_transforms 保持一致）
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2023, 0.1994, 0.2010)

    def __init__(self, latent_dim: int, image_channels: int = 3):
        """
        Args:
            latent_dim:     潜在向量维度（config.yaml: latent_dim）
            image_channels: 输出图像通道数（CIFAR=3）
        """
        super().__init__()

        self.latent_dim = latent_dim

        # ---- 全连接层：将潜在向量映射到初始特征图 ----
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # ---- 转置卷积上采样层 ----
        self.deconv = nn.Sequential(
            # Block 1: 4×4 → 8×8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 16×16 → 32×32，输出层不使用 BN
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),  # 输出值域 [-1, 1]，后续通过 _normalize 映射至 CIFAR 统计范围
        )

        # ---- 注册 CIFAR 归一化参数为 buffer（随模型移动设备，不参与梯度）----
        # 将 Tanh 输出 [-1,1] 仿射变换到 CIFAR 归一化空间：
        #   normalized_pixel = (tanh_out + 1) / 2          → [0, 1]
        #   cifar_pixel      = (pixel_01 - mean) / std     → CIFAR 归一化空间
        # 合并为：cifar_pixel = tanh_out * (0.5/std) + (0.5 - mean) / std
        if image_channels == 3:
            mean = torch.tensor(self.CIFAR_MEAN).view(1, 3, 1, 1)
            std  = torch.tensor(self.CIFAR_STD).view(1, 3, 1, 1)
        else:
            # 灰度图（MNIST 等）：使用单通道归一化参数
            mean = torch.tensor([0.5]).view(1, 1, 1, 1)
            std  = torch.tensor([0.5]).view(1, 1, 1, 1)

        # 仿射变换系数：out = tanh * scale + shift
        scale = 0.5 / std          # shape (1, C, 1, 1)
        shift = (0.5 - mean) / std # shape (1, C, 1, 1)
        self.register_buffer('_norm_scale', scale)
        self.register_buffer('_norm_shift', shift)

        # 权重初始化（DCGAN 标准初始化）
        self._init_weights()

    def _init_weights(self):
        """DCGAN 标准权重初始化：卷积层 N(0,0.02)，BN γ=1 β=0。"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播：潜在向量 → 伪图像（CIFAR 归一化空间）。

        输出与真实 CIFAR 数据在同一统计空间，可直接送入 T_k 和判别器。

        Args:
            z: shape = (B, latent_dim)

        Returns:
            x: shape = (B, image_channels, 32, 32)，统计范围与 CIFAR 归一化数据一致
        """
        h = self.fc(z)                      # (B, 256*4*4)
        h = h.view(h.size(0), 256, 4, 4)   # (B, 256, 4, 4)
        x = self.deconv(h)                  # (B, C, 32, 32)，Tanh 输出 [-1, 1]

        # 仿射变换：将 [-1,1] 映射到 CIFAR 归一化空间
        # x_cifar = x_tanh * scale + shift
        x = x * self._norm_scale + self._norm_shift

        return x

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        便捷采样接口：从标准正态分布采样 ω 并生成伪图像。
        输出已归一化至 CIFAR 统计空间，可直接送入教师模型。

        Args:
            batch_size: 生成样本数量
            device:     目标设备

        Returns:
            x: shape = (batch_size, image_channels, 32, 32)
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.forward(z)