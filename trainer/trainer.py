# trainer/trainer.py
# 功能：生成器 G 三路损失更新（Step 4-A）+ 中心模型 S 蒸馏损失更新（Step 4-B）
# 严格对齐 FedIOD (AAAI 2024) FL.py 实现
#
# FedIOD 对比分析后的关键修复：
#   1. L_conf（loss_align）：去掉温度，使用 log_softmax 形式（更数值稳定）
#   2. L_adv（loss_gan）：GAN loss 等权求和，不用 π_k
#   3. G 损失中新增 loss_adv_distill：G 更新时包含 S 对 ensemble_T 的对齐信号
#   4. S 更新：每步重新采样新数据，同步推理 T 和 S，保证标签-输入对应
#   5. S 更新：去掉温度，直接用原始 logits 做 KL（与 FedIOD 一致）
#   6. 新增 CosineAnnealingLR 调度器支持
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from models.discriminator import generator_adv_loss

EPS = 1e-8


# ================================================================== #
#  生成器损失函数
# ================================================================== #

def compute_loss_align(
    teacher_logits: Dict[int, torch.Tensor],
    pi_k_dict: Dict[int, float],
) -> torch.Tensor:
    """
    L_align（= FedIOD loss_align）：最小化各客户端预测熵，鼓励生成数据语义清晰。

    严格对齐 FedIOD：
      pyx            = softmax(T_k(x))                 # 无温度
      log_softmax_pyx = log_softmax(T_k(x))
      loss_align_k   = -(pyx * log_softmax_pyx).sum(1).mean()   # = H(pyx)
      L_align        = Σ_k π_k · loss_align_k          # π_k 加权

    使用 pyx * log_softmax(pyx) 而非 pyx * log(pyx + eps)：
      数值上等价，但 log_softmax 利用 logsumexp trick，数值更稳定。

    Args:
        teacher_logits: {client_id: z_k}，原始 logits，无温度
        pi_k_dict:      {client_id: π_k}

    Returns:
        loss_align: 标量 tensor，正值
    """
    device = next(iter(teacher_logits.values())).device
    loss_align = torch.tensor(0.0, device=device)

    for k, z_k in teacher_logits.items():
        pi_k = pi_k_dict[k]
        pyx             = F.softmax(z_k, dim=-1)          # (B, C)，无温度
        log_softmax_pyx = F.log_softmax(z_k, dim=-1)      # (B, C)
        # -(pyx * log_softmax_pyx).sum(dim=1).mean() = 交叉熵 H(pyx,pyx) = H(pyx)
        h_k = -(pyx * log_softmax_pyx).sum(dim=1).mean()  # 标量
        loss_align = loss_align + pi_k * h_k

    return loss_align


def compute_loss_balance(
    teacher_logits: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    L_balance（= FedIOD loss_balance）：最大化 batch 边际分布熵，防止模式崩溃。

    严格对齐 FedIOD：
      pyx        = softmax(T_k(x))          # (B, C)，无温度
      py         = pyx.mean(0)              # (C,)，batch 边际分布
      loss_bal_k = (py * log2(py)).sum()    # 负值
      L_balance  = mean(loss_bal_k)         # 等权平均（不用 π_k）

    FedIOD 中直接加入总损失（负值 × w_baln=10），等价于最大化边际熵。
    本函数返回负值（与 FedIOD 原始形式一致），外部直接 + lambda_balance × L_balance。

    Args:
        teacher_logits: {client_id: z_k}，原始 logits

    Returns:
        loss_balance: 标量 tensor，**负值**，直接加入总损失
    """
    balance_list = []
    for k, z_k in teacher_logits.items():
        pyx = F.softmax(z_k, dim=-1)          # (B, C)
        py  = pyx.mean(dim=0)                  # (C,)
        # FedIOD 原始：(py * torch.log2(py)).sum()，负值
        b_k = (py * torch.log2(py + EPS)).sum()
        balance_list.append(b_k)

    # 等权平均（FedIOD: sum/len）
    return torch.stack(balance_list).mean()


def compute_loss_gan(
    clients: list,
    x_fake: torch.Tensor,
) -> torch.Tensor:
    """
    L_gan（= FedIOD loss_gan）：G 对抗判别器的损失。

    严格对齐 FedIOD（is_emsember_generator_GAN_loss="n" 的默认情况）：
      loss_gan = Σ_k BCE(D_k(x), 1) / B    # per-sample 平均，等权求和
    不用 π_k 加权（与我们之前的实现不同）。

    Args:
        clients:  Client 对象列表
        x_fake:   生成数据，不 detach，梯度流向 G

    Returns:
        loss_gan: 标量 tensor
    """
    bce = nn.BCEWithLogitsLoss(reduction='sum')
    loss_gan = torch.tensor(0.0, device=x_fake.device)

    for client in clients:
        client.discriminator.eval()
        d_fake = client.discriminator_output(x_fake)       # (B, 1)
        # BCE(D(x), 1) / B —— per-sample 平均，等权求和
        loss_gan = loss_gan + bce(d_fake, torch.ones_like(d_fake)) / d_fake.size(0)

    return loss_gan


def compute_loss_adv_distill(
    clients: list,
    central_model: nn.Module,
    x_fake: torch.Tensor,
    pi_k_dict: Dict[int, float],
) -> torch.Tensor:
    """
    L_adv_distill（= FedIOD loss_adv）：G 更新时包含 S 对 ensemble_T 的对齐信号。

    FedIOD 原始：
      ensemble_logits_T = ensemble_locals(forward_teacher_outs(syn_img))  # π_k 加权聚合
      logits_S          = netS(syn_img)                                   # S 无温度推理
      loss_adv          = -KL(logits_S, ensemble_logits_T)                # 负号：最大化对齐

    作用：G 被训练成生成"让 S 预测结果接近 T 聚合"的图像，
          使 G 的生成语义与 S 的学习方向一致。

    注意：x_fake 不 detach，梯度通过 S(x) 和 T_k(x) 同时流回 G。
         T_k 已冻结，梯度只流向 x_fake；S 在此处也不更新（requires_grad 不变，
         但我们通过外部 central_model.requires_grad_(False) 控制）。

    Args:
        clients:       Client 对象列表（含冻结的 T_k）
        central_model: 中心模型 S（调用前已 requires_grad_(False)）
        x_fake:        生成数据，保留梯度
        pi_k_dict:     {client_id: π_k}，用于 ensemble

    Returns:
        loss_adv: 标量 tensor（负值，代表"S 对 ensemble_T 的负 KL"）
    """
    # ---- π_k 加权聚合所有 T_k 的 logits ----
    ensemble_logits_T = torch.tensor(0.0, device=x_fake.device)
    for client in clients:
        pi_k = pi_k_dict[client.client_id]
        z_k  = client.teacher(x_fake)           # 梯度流向 x_fake → G
        ensemble_logits_T = ensemble_logits_T + pi_k * z_k

    # ---- S 推理（S 已 requires_grad_(False)，不更新）----
    logits_S = central_model(x_fake)

    # ---- -KL(S ‖ ensemble_T)：最大化 S 与 T 的对齐 ----
    # FedIOD 用 KLDiv(logits_S, ensemble_logits_T, T=args.T)
    # 即 KL(softmax(logits_S/T) || softmax(ensemble_T/T))
    # 取负号：G 希望最大化对齐（最小化 KL）
    log_p_s = F.log_softmax(logits_S, dim=-1)
    p_t     = F.softmax(ensemble_logits_T.detach(), dim=-1)  # T 侧视为常数
    kl      = F.kl_div(log_p_s, p_t, reduction='batchmean')
    return -kl   # 负值，加入总损失后 G 被训练成最小化 KL


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

    对齐 FedIOD update_netG_batch，总损失：
      L_G = w_gan  * L_gan            ← D 对抗，等权求和
          + w_adv  * L_adv_distill    ← S 对 ensemble_T 对齐（负 KL）
          + w_algn * L_align          ← 预测熵最小化（无温度）
          + w_baln * L_balance        ← 边际熵最大化（负值直接加）
          + w_div  * (-L_div)         ← Z 多样性（我们的创新，取负最大化熵）

    Args:
        generator:     生成器 G
        optimizer_g:   G 的 Adam 优化器
        clients:       所有 Client 对象列表
        central_model: 中心模型 S（本步 requires_grad_(False)）
        Z:             上一轮分层聚合的全局软标签 logits
        pi_k_list:     各客户端比例权重列表
        pi_k_dict:     {client_id: π_k}
        cfg:           配置对象
        device:        计算设备
        num_steps:     更新步数（config: generator_steps）

    Returns:
        loss_info: dict，各损失项的平均值
    """
    central_model.requires_grad_(False)
    generator.train()

    # L_div 只与 Z 有关，整个函数内算一次即可
    with torch.no_grad():
        q_Z     = F.softmax(Z / cfg.temperature, dim=-1)
        loss_div = -(q_Z * torch.log(q_Z + EPS)).sum(dim=-1).mean()  # 正值熵

    keys = ['loss_g', 'loss_gan', 'loss_adv', 'loss_align', 'loss_balance', 'loss_div']
    total_losses = {k: 0.0 for k in keys}

    for _ in range(num_steps):
        # ---- 采样生成数据（每步独立，保留梯度）----
        z_noise = torch.randn(cfg.gen_batch_size, cfg.latent_dim, device=device)
        x_fake  = generator(z_noise)                         # (B, C, H, W)

        # ---- 各 T_k 推理（梯度流向 x_fake → G）----
        teacher_logits: Dict[int, torch.Tensor] = {}
        for client in clients:
            teacher_logits[client.client_id] = client.teacher(x_fake)

        # ---- 各损失项计算 ----
        # 1. GAN loss：等权求和（对齐 FedIOD，不用 π_k）
        loss_gan = compute_loss_gan(clients=clients, x_fake=x_fake)

        # 2. Adversarial distill loss：G 与 S 的对齐信号（负 KL）
        loss_adv = compute_loss_adv_distill(
            clients=clients,
            central_model=central_model,
            x_fake=x_fake,
            pi_k_dict=pi_k_dict,
        )

        # 3. Align loss：最小化预测熵（无温度，对齐 FedIOD）
        loss_align = compute_loss_align(
            teacher_logits=teacher_logits,
            pi_k_dict=pi_k_dict,
        )

        # 4. Balance loss：边际熵最大化（负值，直接加入，对齐 FedIOD）
        loss_balance = compute_loss_balance(teacher_logits=teacher_logits)

        # ---- 总损失（对齐 FedIOD 符号约定）----
        # loss_gan:     正值，最小化 → 直接加
        # loss_adv:     负值（负KL），直接加 → 等效于最大化KL对齐
        # loss_align:   正值熵，最小化 → 直接加
        # loss_balance: 负值，直接加 → 等效于最大化边际熵
        # loss_div:     正值熵，最大化 → 取负加（我们的创新项）
        loss_g = (
            cfg.lambda3       * loss_gan       # w_gan
            + cfg.lambda_adv  * loss_adv       # w_adv（负值，加入后 G 最大化KL对齐）
            + cfg.lambda1     * loss_align     # w_algn
            + cfg.lambda_balance * loss_balance  # w_baln（负值，直接加）
            - cfg.lambda2     * loss_div       # 我们的多样性项：取负最大化Z熵
        )

        optimizer_g.zero_grad()
        loss_g.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_g.step()

        total_losses['loss_g']       += loss_g.item()
        total_losses['loss_gan']     += loss_gan.item()
        total_losses['loss_adv']     += loss_adv.item()
        total_losses['loss_align']   += loss_align.item()
        total_losses['loss_balance'] += loss_balance.item()
        total_losses['loss_div']     += loss_div.item()

    central_model.requires_grad_(True)
    return {k: v / num_steps for k, v in total_losses.items()}


# ================================================================== #
#  Step 4-B：更新中心模型 S
# ================================================================== #

def update_central_model(
    generator: nn.Module,
    central_model: nn.Module,
    optimizer_s: torch.optim.Optimizer,
    cluster_assignments: dict,
    logits_dict_getter,          # callable: (x_fake) -> {client_id: z_k}
    pi_k_dict: Dict[int, float],
    cluster_sizes: list,
    cfg,
    device: torch.device,
    num_steps: int,
) -> float:
    """
    执行 central_steps 步中心模型蒸馏更新（Step 4-B）。

    严格对齐 FedIOD update_netS_batch：
      - 每步重新采样新的 syn_img（不复用阶段1的 x_fake）
      - 在新 syn_img 上同步推理 T_k，得到 ensemble_logits_T
      - S 在同一批 syn_img 上推理，用 ensemble_logits_T 作为教师
      - 损失：KL(S(x), ensemble_T(x).detach())，无温度

    与 FedIOD 的差异（我们保留的创新）：
      - 教师聚合使用分层聚合（hierarchical_aggregate）而非简单 π_k 加权
      - 这是本系统相对 FedIOD 的核心改进点

    Args:
        generator:          生成器 G（用于重新采样，不更新）
        central_model:      中心模型 S
        optimizer_s:        S 的优化器
        cluster_assignments: 聚类结果，供 logits_dict_getter 使用
        logits_dict_getter:  callable，接收 x_fake，返回 {client_id: z_k}
                             实际为 server 的 _parallel_local_compute 封装
        pi_k_dict:          {client_id: π_k}
        cluster_sizes:      各簇大小
        cfg:                配置对象
        device:             计算设备
        num_steps:          更新步数（config: central_steps）

    Returns:
        avg_loss_s: 平均蒸馏损失（float）
    """
    from aggregation.aggregation import hierarchical_aggregate

    central_model.train()
    generator.eval()   # G 不更新

    total_loss_s = 0.0

    for _ in range(num_steps):
        # ---- 重新采样新数据（对齐 FedIOD update_netS_batch）----
        with torch.no_grad():
            z_noise    = torch.randn(cfg.gen_batch_size, cfg.latent_dim, device=device)
            x_new      = generator(z_noise)              # (B, C, H, W)

            # T_k 推理，得到新 batch 的教师 logits
            logits_dict = logits_dict_getter(x_new)

            # 分层聚合得到全局软标签 Z（我们的核心创新）
            Z_new, _ = hierarchical_aggregate(
                cluster_assignments=cluster_assignments,
                logits_dict=logits_dict,
                cluster_sizes=cluster_sizes,
                num_clusters=cfg.num_clusters,
                temperature=cfg.temperature,
                gamma=cfg.gamma,
                agg_temperature=cfg.agg_temperature,
            )
            # 教师软标签：对齐 FedIOD，直接用 logits 的 softmax，无额外温度
            # FedIOD: kldiv(logits_S, logits_T.detach(), T=args.T)
            # 我们的蒸馏温度 τ 已在分层聚合内部使用，此处与 FedIOD T 对应
            Z_soft = F.softmax(Z_new / cfg.temperature, dim=-1)   # (B, C)

        # ---- S 在同一新 batch 上推理 ----
        s_logits    = central_model(x_new.detach())
        log_q_hat   = F.log_softmax(s_logits / cfg.temperature, dim=-1)
        loss_s      = F.kl_div(log_q_hat, Z_soft, reduction='batchmean')

        optimizer_s.zero_grad()
        loss_s.backward()
        nn.utils.clip_grad_norm_(central_model.parameters(), max_norm=5.0)
        optimizer_s.step()

        total_loss_s += loss_s.item()

    return total_loss_s / num_steps


# ================================================================== #
#  优化器与调度器构建
# ================================================================== #

def build_optimizers(
    generator: nn.Module,
    central_model: nn.Module,
    clients: list,
    cfg,
    total_steps: int,
) -> Tuple:
    """
    构建所有模型的优化器和 CosineAnnealingLR 调度器。

    对齐 FedIOD init_training：
      - G、S、D 均使用 Adam + CosineAnnealingLR
      - 所有判别器共享一个优化器（FedIOD 风格）
      - 调度器以 total_steps（fed_rounds × 每轮步数）为周期

    Args:
        generator:    生成器 G
        central_model:中心模型 S
        clients:      Client 对象列表（含判别器）
        cfg:          配置对象
        total_steps:  总训练步数（fed_rounds，用于 CosineAnnealingLR T_max）

    Returns:
        optimizer_g, optimizer_s, optimizer_d（所有 D_k 共享）,
        sched_g, sched_s, sched_d
    """
    # G 优化器
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.lr_g,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    # S 优化器
    optimizer_s = torch.optim.Adam(
        central_model.parameters(),
        lr=cfg.lr_s,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    # D 优化器：所有 D_k 参数汇总到一个优化器（对齐 FedIOD）
    all_d_params = []
    for client in clients:
        all_d_params += list(client.discriminator.parameters())
    optimizer_d = torch.optim.Adam(
        all_d_params,
        lr=cfg.lr_d,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    # 将 optimizer_d 注入各客户端（替换原来的独立 optimizer_d）
    for client in clients:
        client.optimizer_d = optimizer_d

    # CosineAnnealingLR 调度器（对齐 FedIOD）
    eta_min = cfg.lr_g * 0.01   # 最小学习率为初始的 1%
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=total_steps, eta_min=eta_min
    )
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s, T_max=total_steps, eta_min=cfg.lr_s * 0.01
    )
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=total_steps, eta_min=cfg.lr_d * 0.01
    )

    return optimizer_g, optimizer_s, optimizer_d, sched_g, sched_s, sched_d