# trainer/trainer.py
# 功能：生成器 G 三路损失更新（Step 4-A）+ 中心模型 S 蒸馏损失更新（Step 4-B）
# 对应阶段 4：全局训练与参数更新
# ============================================================
#
# 公式对应：
#
# 【Step 4-A：更新生成器 G】
#   L_conf = E[Σ_k π_k · H(softmax(T_k(x)/τ))]        ← 最小化不确定性
#   L_div  = H(softmax(Z/τ))                            ← 最大化多样性（取负）
#   L_adv  = -E[Σ_k π_k · log D_k(G(ω))]              ← 对抗损失
#   L_G    = λ1·L_conf + λ2·(-L_div) + λ3·L_adv
#
# 【Step 4-B：更新中心模型 S】
#   Z_soft = softmax(Z/τ)                               ← 教师软标签
#   q̂     = softmax(S(x)/τ)                            ← 学生预测
#   L_S    = KL(Z_soft ‖ q̂)
#          = F.kl_div(log_softmax(S(x)/τ), Z_soft, reduction='batchmean')
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from models.discriminator import generator_adv_loss

# 数值稳定常数
EPS = 1e-8


# ================================================================== #
#  生成器损失计算
# ================================================================== #

def compute_loss_conf(
    teacher_logits: Dict[int, torch.Tensor],
    pi_k_dict: Dict[int, float],
    temperature: float,
) -> torch.Tensor:
    """
    计算 L_conf：最小化各客户端预测的加权香农熵。
    鼓励生成数据具有清晰的语义信息（低熵 = 高置信度预测）。
    对应 FedIOD 中的 loss_align。

    公式：
      L_conf = E[Σ_k π_k · H(softmax(T_k(x)/τ))]

    Args:
        teacher_logits: {client_id: z_k}，各客户端对生成数据的原始 logits
        pi_k_dict:      {client_id: π_k}，全局比例权重
        temperature:    蒸馏温度 τ

    Returns:
        loss_conf: 标量 tensor
    """
    loss_conf = torch.tensor(0.0, device=next(iter(teacher_logits.values())).device)

    for k, z_k in teacher_logits.items():
        pi_k = pi_k_dict[k]
        q_k = F.softmax(z_k / temperature, dim=-1)           # (B, num_classes)
        h_k = -(q_k * torch.log(q_k + EPS)).sum(dim=-1).mean()
        loss_conf = loss_conf + pi_k * h_k

    return loss_conf


def compute_loss_balance(
    teacher_logits: Dict[int, torch.Tensor],
    pi_k_dict: Dict[int, float],
) -> torch.Tensor:
    """
    计算 L_balance：最大化各客户端预测的 batch 级边际分布熵。
    防止生成器模式崩溃（只生成 1~2 个类别）。

    严格对齐 FedIOD loss_balance：
      pyx  = softmax(T_k(x))              # (B, C)，不用温度
      py   = pyx.mean(dim=0)              # (C,)，batch 边际分布
      H_k  = -(py * log2(py)).sum()       # 用 log2（与 FedIOD 一致）
      L_balance = mean(H_k for k in K)    # 等权平均（不用 π_k）

    G 的优化目标：最大化 L_balance（= 最大化边际分布熵）
    在总损失中以 -lambda_balance * L_balance 形式加入（取负 = 最小化负熵）。
    返回正值熵，由 update_generator 取负后加入总损失，语义更清晰。

    Args:
        teacher_logits: {client_id: z_k}，原始 logits，shape=(B, num_classes)
        pi_k_dict:      保留参数（接口兼容），本函数内部等权平均不使用

    Returns:
        loss_balance: 标量 tensor，正值（边际分布熵均值），越大越好
    """
    device = next(iter(teacher_logits.values())).device
    entropy_list = []

    for k, z_k in teacher_logits.items():
        pyx = F.softmax(z_k, dim=-1)           # (B, C)，不用温度
        py  = pyx.mean(dim=0)                   # (C,)，batch 边际分布
        # 用 log2 与 FedIOD 对齐：H = -Σ py·log2(py)
        h_k = -(py * torch.log2(py + EPS)).sum()   # 正值熵，最大值 log2(C)
        entropy_list.append(h_k)

    # 等权平均（FedIOD 用 sum/len，不用 π_k 加权）
    loss_balance = torch.stack(entropy_list).mean()
    return loss_balance


def compute_loss_div(
    Z: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    计算 L_div：对全局软标签 Z 做温度软化后计算熵，最大化多样性。

    设计说明：
      原设计文档中 L_unique 使用 JSD 度量各客户端预测之间的差异；
      本实现（遵循最新约束文档）改为对聚合软标签 Z 的熵：
        L_div = H(softmax(Z/τ))
      生成器更新时取负（最大化熵），促使 G 生成能激发多样化响应的样本。

    公式：
      L_div = H(softmax(Z/τ)) = -Σ_i p^i · log(p^i + eps)
      G 的优化目标：min(-L_div)，即最大化 L_div

    Args:
        Z:           全局软标签 logits，shape=(B, num_classes)
                     由 aggregation.py 的 hierarchical_aggregate 返回
        temperature: 蒸馏温度 τ

    Returns:
        loss_div: 标量 tensor（正值，G 更新时取负加入总损失）
    """
    q_Z = F.softmax(Z / temperature, dim=-1)              # (B, num_classes)
    # batch 平均熵
    h_Z = -(q_Z * torch.log(q_Z + EPS)).sum(dim=-1).mean()  # 标量
    return h_Z


def compute_loss_adv(
    clients: list,
    x_fake: torch.Tensor,
    pi_k_list: List[float],
) -> torch.Tensor:
    """
    计算生成器对抗损失 L_adv（Step 4-A，冻结所有 D_k）。

    公式：
      L_adv = -E[Σ_k π_k · log D_k(G(ω))]
    使用 BCEWithLogitsLoss(target=1) 实现：G 希望 D_k(G(ω)) → 1。

    注意：此处 x_fake 不 detach，梯度需流回生成器 G。

    Args:
        clients:    Client 对象列表（含判别器 D_k）
        x_fake:     生成数据，shape=(B, C, H, W)，保留计算图
        pi_k_list:  各客户端全局比例权重列表

    Returns:
        loss_adv: 标量 tensor
    """
    device = x_fake.device
    loss_adv = torch.tensor(0.0, device=device)

    for client, pi_k in zip(clients, pi_k_list):
        # 判别器切换为 eval，只计算输出不更新参数
        client.discriminator.eval()
        # 获取判别器对生成数据的 logit（不 detach，梯度流向 G）
        d_fake = client.discriminator_output(x_fake)      # (B, 1)
        # 单个判别器的对抗损失贡献（generator_adv_loss 内已乘以 π_k）
        loss_adv = loss_adv + generator_adv_loss(d_fake, pi_k)

    return loss_adv


# ================================================================== #
#  Step 4-A：更新生成器 G
# ================================================================== #

def update_generator(
    generator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    clients: list,
    central_model: nn.Module,
    Z: torch.Tensor,
    pi_k_list: List[float],
    pi_k_dict: Dict[int, float],
    cfg,
    device: torch.device,
    num_steps: int,
) -> Dict[str, float]:
    """
    执行 generator_steps 步生成器更新（Step 4-A）。

    更新期间：
      - 冻结所有 D_k（通过 client.discriminator.eval() + 不调用 optimizer_d）
      - 冻结 S（central_model.requires_grad_(False) 临时设置）
      - G 的参数自由更新

    总损失：
      L_G = λ1·L_conf + λ2·(-L_div) + λ3·L_adv

    Args:
        generator:     生成器 G
        optimizer_g:   G 的 Adam 优化器
        clients:       所有 Client 对象列表
        central_model: 中心模型 S（本步冻结）
        Z:             当前轮全局软标签 logits，shape=(B, num_classes)
        pi_k_list:     各客户端比例权重列表（与 clients 顺序对应）
        pi_k_dict:     {client_id: π_k}（供 L_conf 使用）
        cfg:           配置对象
        device:        计算设备
        num_steps:     更新步数（config: generator_steps）

    Returns:
        loss_info: {'loss_g', 'loss_conf', 'loss_div', 'loss_adv'} 平均值
    """
    # 临时冻结中心模型 S
    central_model.requires_grad_(False)
    generator.train()

    # ---- L_div：Z 与 x_fake 无关，整个函数内只算一次 ----
    # Z 是上一轮聚合结果，视为常数（detach）
    loss_div = compute_loss_div(Z.detach(), cfg.temperature)

    total_losses = {'loss_g': 0., 'loss_conf': 0., 'loss_div': 0.,
                    'loss_adv': 0., 'loss_balance': 0.}

    for _ in range(num_steps):
        # ---- 重新采样生成数据（每步独立采样，G 需要梯度）----
        z_noise = torch.randn(cfg.gen_batch_size, cfg.latent_dim, device=device)
        x_fake = generator(z_noise)                    # (B, C, H, W)，保留梯度

        # ---- 各客户端 T_k 推理，计算 L_conf 和 L_balance ----
        # T_k 已冻结（eval + requires_grad=False），梯度只流向 x_fake → G
        teacher_logits: Dict[int, torch.Tensor] = {}
        for client in clients:
            z_k_grad = client.teacher(x_fake)          # 梯度流向 G，不流向 T_k
            teacher_logits[client.client_id] = z_k_grad

        # ---- L_conf：最小化加权预测熵（鼓励语义清晰）----
        loss_conf = compute_loss_conf(
            teacher_logits=teacher_logits,
            pi_k_dict=pi_k_dict,
            temperature=cfg.temperature,
        )

        # ---- L_balance：最大化 batch 边际分布熵（防止模式崩溃）----
        # 对应 FedIOD loss_balance，权重建议设为 cfg.lambda_balance（默认10）
        # G 最小化 Σ py·log(py)，等价于最大化 H(py)，迫使 batch 覆盖各类别
        loss_balance = compute_loss_balance(
            teacher_logits=teacher_logits,
            pi_k_dict=pi_k_dict,
        )

        # ---- L_adv：对抗损失（x_fake 不 detach，梯度流向 G）----
        loss_adv = compute_loss_adv(
            clients=clients,
            x_fake=x_fake,
            pi_k_list=pi_k_list,
        )

        # ---- 总损失 ----
        # L_G = λ1·L_conf - λ2·L_div + λ3·L_adv - λ_balance·L_balance
        #
        # 各项符号说明：
        #   L_conf:    正值熵，最小化 → 预测确定（直接加）
        #   L_div:     正值熵，最大化 → 取负加
        #   L_adv:     正值，最小化 → 直接加
        #   L_balance: 正值熵（边际分布熵），最大化 → 取负加
        #              对应 FedIOD: loss = ... + w_baln * (py·log2(py)).sum()
        #              FedIOD 中 (py·log2(py)).sum() 为负值，所以 + w_baln×负值
        #              我们返回正值熵，等价地用 - lambda_balance × loss_balance
        loss_g = (
            cfg.lambda1 * loss_conf
            - cfg.lambda2 * loss_div
            + cfg.lambda3 * loss_adv
            - cfg.lambda_balance * loss_balance  # 取负：最大化边际分布熵
        )

        optimizer_g.zero_grad()
        loss_g.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_g.step()

        total_losses['loss_g']       += loss_g.item()
        total_losses['loss_conf']    += loss_conf.item()
        total_losses['loss_div']     += loss_div.item()
        total_losses['loss_adv']     += loss_adv.item()
        total_losses['loss_balance'] += loss_balance.item()

    # 恢复中心模型梯度
    central_model.requires_grad_(True)

    # 返回平均损失
    return {k: v / num_steps for k, v in total_losses.items()}


# ================================================================== #
#  Step 4-B：更新中心模型 S
# ================================================================== #

def update_central_model(
    central_model: nn.Module,
    optimizer_s: torch.optim.Optimizer,
    x_fake: torch.Tensor,
    Z: torch.Tensor,
    cfg,
    num_steps: int,
) -> float:
    """
    执行 central_steps 步中心模型蒸馏更新（Step 4-B）。

    【重要设计约束】
    x_fake 必须是阶段1生成、阶段2客户端推理、阶段3聚合所用的同一批数据，
    即 Z = Aggregate({T_k(x_fake)})。
    S 在同一批 x_fake 上做前向推理，用对应的 Z 作为教师标签，
    保证"输入-标签"严格对应。
    禁止在本函数内重新采样生成数据，否则标签与输入不匹配，蒸馏信号错误。

    central_steps > 1 时：复用同一批 x_fake（detach 后）多次前向更新 S，
    相当于对同一组(x, Z_soft)做多步梯度下降，等价于增大有效学习步长，
    同时避免引入新的不匹配数据。

    蒸馏损失：
      Z_soft = softmax(Z/τ)                  ← 教师软标签（阶段3聚合结果）
      q̂     = softmax(S(x_fake)/τ)          ← 学生预测（同一批 x_fake）
      L_S    = KL(Z_soft ‖ q̂)
             = F.kl_div(log_softmax(S(x_fake)/τ), Z_soft, reduction='batchmean')

    Args:
        central_model: 中心模型 S
        optimizer_s:   S 的 Adam 优化器
        x_fake:        阶段1生成的伪数据，shape=(B, C, H, W)
                       与 Z 严格对应同一批，调用前已 detach
        Z:             全局软标签 logits，shape=(B, num_classes)
                       由 hierarchical_aggregate({T_k(x_fake)}) 得到
        cfg:           配置对象
        num_steps:     更新步数（config: central_steps）

    Returns:
        avg_loss_s: 平均蒸馏损失（float）
    """
    central_model.train()

    # x_fake 来自阶段1，确保 detach（S 的梯度不回传至 G）
    x_input = x_fake.detach()

    # 教师软标签：对 Z 做温度软化，整个 num_steps 内固定不变
    with torch.no_grad():
        Z_soft = F.softmax(Z / cfg.temperature, dim=-1)   # (B, num_classes)

    total_loss_s = 0.0

    for _ in range(num_steps):
        # ---- S 在同一批 x_fake 上前向推理 ----
        # 复用同一批数据多步更新，保证标签-输入严格对应
        s_logits = central_model(x_input)                  # (B, num_classes)

        # ---- 蒸馏损失：KL(Z_soft ‖ softmax(S(x)/τ)) ----
        # F.kl_div：第一个参数为 log 概率（学生），第二个为概率（教师）
        log_q_hat = F.log_softmax(s_logits / cfg.temperature, dim=-1)
        loss_s = F.kl_div(
            input=log_q_hat,
            target=Z_soft,
            reduction='batchmean',
        )

        optimizer_s.zero_grad()
        loss_s.backward()
        nn.utils.clip_grad_norm_(central_model.parameters(), max_norm=5.0)
        optimizer_s.step()

        total_loss_s += loss_s.item()

    return total_loss_s / num_steps


# ================================================================== #
#  优化器构建工具函数
# ================================================================== #

def build_optimizers(
    generator: nn.Module,
    central_model: nn.Module,
    cfg,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    构建生成器和中心模型的 Adam 优化器。

    Args:
        generator:     生成器 G
        central_model: 中心模型 S
        cfg:           配置对象

    Returns:
        (optimizer_g, optimizer_s)
    """
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.lr_g,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    optimizer_s = torch.optim.Adam(
        central_model.parameters(),
        lr=cfg.lr_s,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    return optimizer_g, optimizer_s