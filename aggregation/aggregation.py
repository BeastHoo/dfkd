# aggregation/aggregation.py
# 功能：分层知识聚合机制，依次完成簇内聚合与服务器级聚合，生成全局软标签 Z
# 对应阶段 3：分层聚合
# ============================================================
#
# 公式对应：
#
# 【簇内聚合】
#   预测熵：  H(q_k) = -Σ_i q_k^i · log(q_k^i + eps)
#   指数权重：α_k = exp(-H(q_k)) / Σ_{j∈c} exp(-H(q_j))
#   簇级logits：z̄_c = Σ_{k∈c} α_k · z_k
#
# 【服务器聚合】
#   簇软概率：q_c = softmax(z̄_c)
#   簇熵：    H(q_c) = -Σ_i q_c^i · log(q_c^i + eps)
#   簇权重：  β_c = [N_c^γ · exp(-H(q_c) / T_agg)]
#                  / Σ_j [N_j^γ · exp(-H(q_j) / T_agg)]
#   全局软标签：Z = Σ_{c=1}^{M} β_c · z̄_c
# ============================================================

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# 数值稳定常数，防止 log(0)
EPS = 1e-8


# ------------------------------------------------------------------ #
#  工具函数：计算香农熵
# ------------------------------------------------------------------ #

def shannon_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算概率分布的香农熵 H(p) = -Σ p^i · log(p^i + eps)。

    Args:
        probs: 概率分布张量，最后一维为类别维
               shape 可为 (num_classes,) 或 (B, num_classes)
        dim:   求和维度，默认 -1（类别维）

    Returns:
        entropy: 标量或 (B,) 向量
    """
    return -(probs * torch.log(probs + EPS)).sum(dim=dim)


# ------------------------------------------------------------------ #
#  簇内聚合（由簇头执行）
# ------------------------------------------------------------------ #

def intra_cluster_aggregate(
    cluster_id: int,
    cluster_members: List[int],
    logits_dict: Dict[int, torch.Tensor],
    temperature: float,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    簇内聚合：基于预测熵的指数加权策略，生成簇级聚合 logits z̄_c。

    设计说明：
      原始设计文档使用 conf_k = max(softmax(z_k)) 计算置信度；
      本实现采用负熵指数加权（α_k ∝ exp(-H(q_k))），
      与设计文档阶段3中"基于负熵的指数加权"描述一致：
        - 熵越低 → 预测越确定 → 权重越大
        - exp(-H) 比直接用 max_prob 对置信度差异更敏感，聚合更稳定

    对应公式：
      q_k = softmax(z_k / τ)
      H(q_k) = -Σ_i q_k^i · log(q_k^i + eps)
      α_k = exp(-H(q_k)) / Σ_{j∈c} exp(-H(q_j))
      z̄_c = Σ_{k∈c} α_k · z_k

    Args:
        cluster_id:      簇编号 c（仅用于日志）
        cluster_members: 簇内客户端编号列表
        logits_dict:     {client_id: z_k}，z_k shape=(B, num_classes)
                         各客户端上传的原始 logits
        temperature:     蒸馏温度 τ（config: temperature）

    Returns:
        z_bar_c:     簇级聚合 logits，shape = (B, num_classes)
        alpha_dict:  各客户端权重 {client_id: alpha_k}（用于日志/调试）
    """
    device = next(iter(logits_dict.values())).device

    # ---- Step 1：计算每个客户端的预测熵 ----
    entropy_list: List[torch.Tensor] = []   # 每项为标量（batch 平均熵）
    valid_members: List[int] = []

    for k in cluster_members:
        if k not in logits_dict:
            # 若某客户端 logits 缺失，跳过（容错处理）
            continue
        z_k = logits_dict[k]                              # (B, num_classes)
        q_k = F.softmax(z_k / temperature, dim=-1)       # (B, num_classes)
        h_k = shannon_entropy(q_k, dim=-1).mean()        # 标量，batch 平均熵
        entropy_list.append(h_k)
        valid_members.append(k)

    if len(valid_members) == 0:
        raise RuntimeError(f"簇 {cluster_id} 中没有有效的客户端 logits")

    # ---- Step 2：指数加权，熵越低权重越大 ----
    # 使用 log-sum-exp 技巧保证数值稳定
    neg_entropies = torch.stack([-h for h in entropy_list])  # (|c|,)
    # softmax(-H) 等价于 exp(-H) / Σexp(-H)，利用 PyTorch softmax 的数值稳定实现
    alphas = F.softmax(neg_entropies, dim=0)                 # (|c|,)，和为1

    # ---- Step 3：加权聚合 logits ----
    z_bar_c = torch.zeros_like(logits_dict[valid_members[0]])  # (B, num_classes)
    alpha_dict: Dict[int, float] = {}

    for idx, k in enumerate(valid_members):
        alpha_k = alphas[idx].item()
        z_bar_c = z_bar_c + alpha_k * logits_dict[k]
        alpha_dict[k] = alpha_k

    return z_bar_c, alpha_dict


# ------------------------------------------------------------------ #
#  服务器级聚合（含温度-规模平衡机制）
# ------------------------------------------------------------------ #

def server_aggregate(
    cluster_logits: Dict[int, torch.Tensor],
    cluster_sizes: List[int],
    num_clusters: int,
    gamma: float,
    agg_temperature: float,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    服务器级聚合：联合簇规模 N_c 与知识质量 Q_c 计算权重 β_c，
    生成全局软标签 Z（logits 空间）。

    对应公式：
      q_c    = softmax(z̄_c)
      H(q_c) = -Σ_i q_c^i · log(q_c^i + eps)
      β_c    = [N_c^γ · exp(-H(q_c) / T_agg)]
               / Σ_j [N_j^γ · exp(-H(q_j) / T_agg)]
      Z      = Σ_{c=1}^{M} β_c · z̄_c

    设计说明：
      - N_c^γ：簇规模指数，γ=1 时等比于样本数，γ→0 时退化为等权
      - exp(-H(q_c)/T_agg)：知识质量项，熵越低（预测越确定）权重越大
      - T_agg：聚合温度，控制质量差异的放大程度；T_agg→∞ 退化为纯规模加权
      - 联合加权兼顾统计代表性（规模）与预测可信度（质量）

    Args:
        cluster_logits:  {cluster_id: z̄_c}，各簇聚合 logits，shape=(B, num_classes)
        cluster_sizes:   List[int]，长度 M，cluster_sizes[c] = 簇 c 客户端数 N_c
        num_clusters:    簇总数 M
        gamma:           簇规模指数 γ（config: gamma）
        agg_temperature: 聚合温度 T_agg（config: agg_temperature）

    Returns:
        Z:          全局软标签 logits，shape = (B, num_classes)
        beta_dict:  各簇权重 {cluster_id: beta_c}（用于日志/调试）
    """
    device = next(iter(cluster_logits.values())).device

    valid_clusters = sorted(cluster_logits.keys())

    # ---- Step 1：计算每个簇的熵 H(q_c) ----
    entropy_list: List[torch.Tensor] = []
    size_list: List[float] = []

    for c in valid_clusters:
        z_bar_c = cluster_logits[c]                        # (B, num_classes)
        q_c = F.softmax(z_bar_c, dim=-1)                  # (B, num_classes)
        h_c = shannon_entropy(q_c, dim=-1).mean()         # 标量，batch 平均熵
        entropy_list.append(h_c)
        size_list.append(float(cluster_sizes[c]))

    # ---- Step 2：计算 β_c（温度-规模平衡加权）----
    # log β_c ∝ γ·log(N_c) + (-H(q_c) / T_agg)
    # 使用 log 域计算再 softmax，避免 N_c^γ 数值溢出

    log_weights: List[torch.Tensor] = []
    for idx, c in enumerate(valid_clusters):
        n_c = size_list[idx]
        h_c = entropy_list[idx]

        # γ·log(N_c)：规模项（log 域）
        log_n = gamma * torch.log(
            torch.tensor(n_c, dtype=torch.float32, device=device) + EPS
        )
        # -H(q_c) / T_agg：质量项（log 域）
        log_q_term = -h_c / agg_temperature

        log_weights.append(log_n + log_q_term)

    log_weights_tensor = torch.stack(log_weights)          # (M,)
    # softmax 在 log 域等价于归一化 exp，数值稳定
    betas = F.softmax(log_weights_tensor, dim=0)           # (M,)，和为1

    # ---- Step 3：加权聚合生成全局软标签 Z ----
    Z = torch.zeros_like(cluster_logits[valid_clusters[0]])  # (B, num_classes)
    beta_dict: Dict[int, float] = {}

    for idx, c in enumerate(valid_clusters):
        beta_c = betas[idx].item()
        Z = Z + beta_c * cluster_logits[c]
        beta_dict[c] = beta_c

    return Z, beta_dict


# ------------------------------------------------------------------ #
#  完整分层聚合流程（统一入口）
# ------------------------------------------------------------------ #

def hierarchical_aggregate(
    cluster_assignments: Dict[int, List[int]],
    logits_dict: Dict[int, torch.Tensor],
    cluster_sizes: List[int],
    num_clusters: int,
    temperature: float,
    gamma: float,
    agg_temperature: float,
) -> Tuple[torch.Tensor, dict]:
    """
    执行完整的两阶段分层聚合，返回全局软标签 Z。

    流程：
      1. 对每个簇执行簇内聚合 → 得到 {cluster_id: z̄_c}
      2. 执行服务器级聚合 → 得到全局软标签 Z

    Args:
        cluster_assignments: {cluster_id: [client_ids]}
        logits_dict:         {client_id: z_k}，各客户端原始 logits
        cluster_sizes:       各簇客户端数量列表
        num_clusters:        簇总数 M
        temperature:         蒸馏温度 τ（簇内聚合使用）
        gamma:               规模指数 γ（服务器聚合使用）
        agg_temperature:     聚合温度 T_agg（服务器聚合使用）

    Returns:
        Z:       全局软标签 logits，shape = (B, num_classes)
        info:    聚合过程信息 dict，包含各层权重，用于日志记录
                 {'intra': {cluster_id: alpha_dict},
                  'inter': {cluster_id: beta_c}}
    """
    # ---- 阶段 3-1：簇内聚合 ----
    cluster_logits: Dict[int, torch.Tensor] = {}
    intra_weights: dict = {}

    for c, members in cluster_assignments.items():
        # 过滤出该簇内有 logits 的客户端
        available = [k for k in members if k in logits_dict]
        if len(available) == 0:
            continue

        z_bar_c, alpha_dict = intra_cluster_aggregate(
            cluster_id=c,
            cluster_members=available,
            logits_dict=logits_dict,
            temperature=temperature,
        )
        cluster_logits[c] = z_bar_c
        intra_weights[c] = alpha_dict

    if len(cluster_logits) == 0:
        raise RuntimeError("所有簇均无有效 logits，无法执行聚合")

    # ---- 阶段 3-2：服务器级聚合 ----
    Z, beta_dict = server_aggregate(
        cluster_logits=cluster_logits,
        cluster_sizes=cluster_sizes,
        num_clusters=num_clusters,
        gamma=gamma,
        agg_temperature=agg_temperature,
    )

    info = {
        'intra': intra_weights,   # {cluster_id: {client_id: alpha_k}}
        'inter': beta_dict,       # {cluster_id: beta_c}
    }

    return Z, info