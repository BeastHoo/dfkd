# models/discriminator.py
# 功能：轻量判别器 D_k，对真实数据与生成数据进行二分类
# 输出：原始标量 logit（不加 sigmoid），配合 BCEWithLogitsLoss 使用
# 对应阶段 2-A：判别器训练，公式 L_{D_k}
# ============================================================

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    轻量判别器 D_k。

    输入：图像 x，shape = (B, image_channels, 32, 32)
    输出：标量 logit，shape = (B, 1)，不经过 sigmoid

    结构：
        Conv(ch→64,   4×4, stride=2, pad=1) + LeakyReLU   # 32→16
        Conv(64→128,  4×4, stride=2, pad=1) + BN + LeakyReLU  # 16→8
        Conv(128→256, 4×4, stride=2, pad=1) + BN + LeakyReLU  # 8→4
        Flatten → FC(256*4*4 → 1)

    损失函数（外部使用）：nn.BCEWithLogitsLoss
      - 真实数据标签：1.0
      - 生成数据标签：0.0
    """

    def __init__(self, image_channels: int = 3):
        """
        Args:
            image_channels: 输入图像通道数（CIFAR=3）
        """
        super().__init__()

        # ---- 卷积特征提取层 ----
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16，第一层不加 BN（DCGAN 规范）
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 8×8 → 4×4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---- 输出层：全连接，输出原始 logit ----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),   # 输出标量 logit，不加激活函数
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """DCGAN 标准权重初始化：卷积层 N(0,0.02)，BN γ=1 β=0。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：图像 → 标量 logit。

        Args:
            x: shape = (B, image_channels, 32, 32)

        Returns:
            logit: shape = (B, 1)，原始 logit，未经 sigmoid
        """
        feat = self.features(x)       # (B, 256, 4, 4)
        logit = self.classifier(feat) # (B, 1)
        return logit


# ------------------------------------------------------------------ #
#  判别器损失计算工具函数
#  对应公式 L_{D_k}（Step 2-A）和 L_adv(G)（Step 4-A）
# ------------------------------------------------------------------ #

def discriminator_loss(
    d_real: torch.Tensor,
    d_fake: torch.Tensor,
    pi_k: float,
) -> torch.Tensor:
    """
    计算判别器损失 L_{D_k}（最小化目标）。

    公式：
        L_{D_k} = -E[π_k · log D_k(x^priv)]   ← 真实数据项
                - E[log(1 - D_k(G(ω)))]         ← 生成数据项

    使用 BCEWithLogitsLoss 实现（数值稳定，内部含 sigmoid）：
        真实数据：target = 1.0，权重 = π_k
        生成数据：target = 0.0，权重 = 1.0

    Args:
        d_real: 判别器对真实数据的输出 logit，shape = (B, 1)
        d_fake: 判别器对生成数据的输出 logit，shape = (B, 1)
        pi_k:   客户端 k 的全局比例权重 π_k

    Returns:
        loss: 标量 tensor
    """
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    # 真实数据项：-π_k · E[log D(x^priv)]
    real_labels = torch.ones_like(d_real)
    loss_real = pi_k * bce(d_real, real_labels)

    # 生成数据项：-E[log(1 - D(G(ω)))]
    fake_labels = torch.zeros_like(d_fake)
    loss_fake = bce(d_fake, fake_labels)

    return loss_real + loss_fake


def generator_adv_loss(
    d_fake: torch.Tensor,
    pi_k: float,
) -> torch.Tensor:
    """
    计算单个判别器对生成器的对抗损失贡献（Step 4-A，L_adv 的一项）。

    公式：
        L_adv(G) 中客户端 k 的贡献 = -π_k · E[log D_k(G(ω))]

    使用 BCEWithLogitsLoss，target=1（欺骗判别器）：
        G 希望 D_k(G(ω)) → 1，等价于最小化 BCE(logit, 1)

    Args:
        d_fake: 判别器对生成数据的输出 logit，shape = (B, 1)
        pi_k:   客户端 k 的全局比例权重 π_k

    Returns:
        loss: 标量 tensor（已乘以 π_k）
    """
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    real_labels = torch.ones_like(d_fake)
    return pi_k * bce(d_fake, real_labels)