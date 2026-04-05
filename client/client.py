# client/client.py
# 功能：客户端类，封装教师模型加载、判别器训练（Step 2-A）、冻结推理（Step 2-B）
# 对应阶段 2：本地对抗训练与推理
# ============================================================
#
# 公式对应：
#   Step 2-A 判别器损失 L_{D_k}（公式）：
#     L_{D_k} = -E[π_k · log D_k(x^priv)] - E[log(1 - D_k(G(ω)))]
#   Step 2-B 推理：
#     z_k = T_k(x)
#     q_k = softmax(z_k / τ)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional

from models.task_model import load_teacher_model
from models.discriminator import Discriminator, discriminator_loss


class Client:
    """
    联邦学习客户端。

    职责：
      - 持有本地教师模型 T_k（参数全程冻结，仅推理）
      - 持有本地判别器 D_k（参与 GAN 对抗训练）
      - Step 2-A：用真实数据与生成数据训练 D_k
      - Step 2-B：用冻结的 T_k 对生成数据做前向推理，返回 z_k
    """

    def __init__(
        self,
        client_id: int,
        pi_k: float,
        data_loader: DataLoader,
        cfg,
        device: torch.device,
    ):
        """
        初始化客户端。

        Args:
            client_id:   客户端编号 k（从 0 开始）
            pi_k:        全局比例权重 π_k = 本地样本数 / 总样本数
            data_loader: 本地真实数据 DataLoader（用于判别器训练）
            cfg:         OmegaConf 配置对象
            device:      计算设备
        """
        self.client_id = client_id
        self.pi_k = pi_k
        self.data_loader = data_loader
        self.cfg = cfg
        self.device = device

        # ---- 加载教师模型 T_k（冻结）----
        self.teacher: nn.Module = load_teacher_model(
            client_id=client_id,
            teacher_ckpt_dir=cfg.teacher_ckpt_dir,
            num_classes=cfg.num_classes,
            image_channels=cfg.image_channels,
            device=device,
        )

        # ---- 初始化判别器 D_k ----
        self.discriminator: nn.Module = Discriminator(
            image_channels=cfg.image_channels,
        ).to(device)

        # ---- 判别器优化器 ----
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr_d,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        # 本地数据迭代器（循环使用，避免 epoch 边界问题）
        self._data_iter = iter(self.data_loader)

    # ------------------------------------------------------------------ #
    #  内部工具：获取一批真实数据
    # ------------------------------------------------------------------ #

    def _get_real_batch(self) -> torch.Tensor:
        """
        从本地 DataLoader 循环取一个 batch 的真实图像。
        当迭代器耗尽时自动重置，确保每次都能取到数据。

        注意：若 DataLoader 为空（样本数为0，或 drop_last=True 且样本数
        不足一个 batch），将抛出明确错误而非静默失败。

        Returns:
            images: shape = (B, C, H, W)，已移至 device
        """
        # 空 DataLoader 保护：提前检测，给出清晰错误信息
        if len(self.data_loader) == 0:
            raise RuntimeError(
                f"Client {self.client_id} 的本地 DataLoader 为空！\n"
                f"  数据集样本数: {len(self.data_loader.dataset)}\n"
                f"  batch_size:   {self.data_loader.batch_size}\n"
                f"  原因：样本数不足一个 batch，且 drop_last=True 丢弃了所有样本。\n"
                f"  解决方案：data_utils.py 中已将 drop_last 改为 False，\n"
                f"            请重新运行程序。"
            )

        try:
            images, _ = next(self._data_iter)
        except StopIteration:
            # 重置迭代器（正常的 epoch 结束）
            self._data_iter = iter(self.data_loader)
            images, _ = next(self._data_iter)
        return images.to(self.device)

    # ------------------------------------------------------------------ #
    #  Step 2-A：计算判别器损失（不做 backward，由 server 层联合更新）
    # ------------------------------------------------------------------ #

    def compute_discriminator_loss(
        self,
        x_fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算单个判别器 D_k 的损失 L_{D_k}（不执行 backward）。

        对应 FedIOD update_netDS_batch 中单个判别器的 loss_d 计算。
        所有客户端的损失在 server.py 中累加后统一 backward + step，
        与 FedIOD 的联合更新方式完全对齐。

        判别器损失（最小化）：
          L_{D_k} = [-π_k·log D_k(x^priv) - log(1-D_k(G(ω)))] / 2
        使用 BCEWithLogitsLoss（内含 sigmoid，数值稳定）。

        Args:
            x_fake: 生成器产生的伪数据，已 detach，shape=(B, C, H, W)

        Returns:
            loss_d: 标量 tensor，保留计算图供外部 backward
        """
        self.discriminator.train()
        x_real = self._get_real_batch()

        # 对齐 batch size
        b = min(x_real.size(0), x_fake.size(0))
        x_real       = x_real[:b]
        x_fake_batch = x_fake[:b]

        d_real = self.discriminator(x_real)                   # (B, 1)
        d_fake = self.discriminator(x_fake_batch.detach())    # (B, 1)，不回传至 G

        loss_d = discriminator_loss(
            d_real=d_real,
            d_fake=d_fake,
            pi_k=self.pi_k,
        )
        return loss_d

    def train_discriminator(
        self,
        x_fake: torch.Tensor,
        num_steps: int,
    ) -> float:
        """
        执行 local_gan_steps 步判别器独立更新（兼容旧接口，内部调用新接口）。

        注意：当前系统在 server.py 中调用 train_discriminators_jointly，
        本方法保留供单独测试使用。

        Args:
            x_fake:    伪数据，已 detach
            num_steps: 更新步数

        Returns:
            avg_loss: 平均判别器损失
        """
        total_loss = 0.0
        for _ in range(num_steps):
            loss = self.compute_discriminator_loss(x_fake)
            self.optimizer_d.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.optimizer_d.step()
            total_loss += loss.item()
        self.discriminator.eval()
        return total_loss / num_steps

        self.discriminator.eval()
        return total_loss / num_steps

    # ------------------------------------------------------------------ #
    #  Step 2-B：教师模型推理（冻结）
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用冻结的教师模型 T_k 对生成数据 x 做前向推理（Step 2-B）。

        公式：
          z_k = T_k(x)                    ← 原始 logits
          q_k = softmax(z_k / τ)          ← 温度软化概率

        T_k 参数已冻结（eval + requires_grad=False），此处额外用
        torch.no_grad() 确保不建立计算图，节省显存。

        Args:
            x:           生成数据，shape = (B, C, H, W)
            temperature: 蒸馏温度 τ（config: temperature）

        Returns:
            z_k: 原始 logits，shape = (B, num_classes)
            q_k: 温度软化概率，shape = (B, num_classes)
        """
        self.teacher.eval()
        x = x.to(self.device)

        z_k = self.teacher(x)                          # (B, num_classes)
        q_k = F.softmax(z_k / temperature, dim=-1)    # (B, num_classes)

        return z_k, q_k

    # ------------------------------------------------------------------ #
    #  Step 2-B 扩展：判别器对生成数据的输出（供生成器对抗损失使用）
    # ------------------------------------------------------------------ #

    def discriminator_output(self, x_fake: torch.Tensor) -> torch.Tensor:
        """
        返回判别器 D_k 对生成数据的 logit 输出。
        供 trainer.py 计算 L_adv(G) 时调用（不 detach，梯度流向 G）。

        Args:
            x_fake: 生成数据，shape = (B, C, H, W)，需保留梯度

        Returns:
            logit: shape = (B, 1)，原始 logit
        """
        self.discriminator.eval()
        return self.discriminator(x_fake)

    # ------------------------------------------------------------------ #
    #  执行完整的单轮本地计算（Step 2-A + Step 2-B）
    # ------------------------------------------------------------------ #

    def local_compute(
        self,
        x_fake: torch.Tensor,
        temperature: float,
        num_gan_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        执行客户端单轮完整本地计算，对应阶段 2 的两个并行步骤。

        执行顺序：
          1. Step 2-A：训练判别器 D_k（num_gan_steps 步）
          2. Step 2-B：T_k 推理生成数据，返回 z_k 和 q_k

        注意：两步在实现上串行执行，用 ThreadPoolExecutor 在 server.py
        层面跨客户端并行，模拟真实联邦场景的客户端并发。

        Args:
            x_fake:       生成器产生的伪数据（Step 2-A 用于判别器训练）
            temperature:  蒸馏温度 τ
            num_gan_steps: 判别器更新步数

        Returns:
            z_k:      原始 logits，shape = (B, num_classes)
            q_k:      温度软化概率，shape = (B, num_classes)
            d_loss:   本步判别器平均损失（用于日志）
        """
        # Step 2-A：判别器训练
        d_loss = self.train_discriminator(
            x_fake=x_fake,
            num_steps=num_gan_steps,
        )

        # Step 2-B：教师模型推理
        z_k, q_k = self.infer(x=x_fake, temperature=temperature)

        return z_k, q_k, d_loss

    # ------------------------------------------------------------------ #
    #  状态保存与恢复（用于 checkpoint）
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        """返回判别器及其优化器状态，用于 checkpoint 保存。"""
        return {
            'discriminator': self.discriminator.state_dict(),
            'optimizer_d':   self.optimizer_d.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """从 checkpoint 恢复判别器及优化器状态。"""
        self.discriminator.load_state_dict(state['discriminator'])
        self.optimizer_d.load_state_dict(state['optimizer_d'])
        print(f"[client] Client {self.client_id}: 判别器状态已从 checkpoint 恢复")