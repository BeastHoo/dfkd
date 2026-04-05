# clustering/sinkhorn.py
# 功能：基于熵正则化最优传输的 Sinkhorn 距离计算
# 实现：log 域 Sinkhorn-Knopp 迭代（数值稳定，支持 GPU）
# 对应设计文档：基于 Sinkhorn 距离的客户端聚类一节
# ============================================================
#
# 核心公式：
#   W_ε(P_k, P_i) = min_{Γ∈Π(p_k,p_i)} <Γ,C> + ε·KL(Γ‖p_k p_i^T)
#
# log 域迭代（数值稳定版）：
#   log_K  = -C / ε
#   log_u^{t+1} = log(p_k) - logsumexp(log_K + log_v^t,  dim=1)
#   log_v^{t+1} = log(p_i) - logsumexp(log_K^T + log_u^{t+1}, dim=1)
#   log_Γ* = log_u[:,None] + log_K + log_v[None,:]
#   W_ε    = Σ_{uv} exp(log_Γ*_{uv}) · C_{uv}
# ============================================================

import torch
from typing import Optional


# ------------------------------------------------------------------ #
#  代价矩阵构建
# ------------------------------------------------------------------ #

def build_cost_matrix(num_classes: int, device: torch.device) -> torch.Tensor:
    """
    构建类别标签之间的代价矩阵 C ∈ R^{n×n}。

    C_{uv} = (u - v)²，即类别标签之间的平方距离。
    对角线为 0，表示相同类别传输代价为 0。

    Args:
        num_classes: 类别数 n（即概率向量维度）
        device:      目标设备

    Returns:
        C: shape = (n, n)，float32
    """
    idx = torch.arange(num_classes, dtype=torch.float32, device=device)
    # 广播计算平方距离矩阵
    C = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2   # (n, n)
    return C


# ------------------------------------------------------------------ #
#  单对 Sinkhorn 距离
# ------------------------------------------------------------------ #

def sinkhorn_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    num_iters: int,
) -> torch.Tensor:
    """
    计算两个离散分布 p 与 q 之间的 Sinkhorn 距离 W_ε(p, q)。
    使用 log 域迭代，数值稳定，支持 GPU。

    Args:
        p:        分布向量，shape = (n,)，归一化（Σ=1），值 > 0
        q:        分布向量，shape = (n,)，归一化（Σ=1），值 > 0
        C:        代价矩阵，shape = (n, n)
        eps:      熵正则化系数 ε（config: sinkhorn_eps）
        num_iters: Sinkhorn-Knopp 迭代次数（config: sinkhorn_iters）

    Returns:
        dist: 标量 tensor，Sinkhorn 距离值 W_ε(p, q)
    """
    n = p.shape[0]
    device = p.device

    # 数值稳定：将零概率替换为极小值，避免 log(0)
    eps_prob = 1e-8
    p = (p + eps_prob) / (p + eps_prob).sum()
    q = (q + eps_prob) / (q + eps_prob).sum()

    log_p = torch.log(p)   # (n,)
    log_q = torch.log(q)   # (n,)

    # 核矩阵的对数形式：log K = -C / ε
    log_K = -C / eps       # (n, n)

    # 初始化 log_v = 0（等价于 v = 1）
    log_v = torch.zeros(n, device=device, dtype=p.dtype)

    # ---- Sinkhorn-Knopp log 域迭代 ----
    for _ in range(num_iters):
        # log_u^{t+1} = log(p) - logsumexp(log_K + log_v, dim=1)
        # log_K + log_v[None,:] 的每行加上对应的 log_v
        log_u = log_p - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

        # log_v^{t+1} = log(q) - logsumexp(log_K^T + log_u, dim=1)
        log_v = log_q - torch.logsumexp(log_K.t() + log_u.unsqueeze(0), dim=1)

    # ---- 计算最优传输矩阵（log 域）----
    # log_Γ* = log_u[:,None] + log_K + log_v[None,:]
    log_gamma = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)  # (n, n)
    gamma = torch.exp(log_gamma)   # (n, n)

    # ---- Sinkhorn 距离 = <Γ*, C> ----
    dist = (gamma * C).sum()

    return dist


# ------------------------------------------------------------------ #
#  批量构建 K×K 距离矩阵
# ------------------------------------------------------------------ #

def compute_sinkhorn_matrix(
    p_k_list: list,
    eps: float,
    num_iters: int,
    device: torch.device,
) -> torch.Tensor:
    """
    计算所有客户端对之间的 Sinkhorn 距离，构建 K×K 对称距离矩阵。

    对应设计文档：
      d(k, i) = W_ε(P_k, P_i)，构建 K×K 距离矩阵

    Args:
        p_k_list:  List[np.ndarray 或 torch.Tensor]，长度 K，
                   每项为客户端类别分布向量，shape=(num_classes,)
        eps:       Sinkhorn 正则化系数 ε（config: sinkhorn_eps）
        num_iters: 迭代次数（config: sinkhorn_iters）
        device:    计算设备（支持 GPU）

    Returns:
        dist_matrix: shape = (K, K)，对称矩阵，对角线为 0
    """
    import numpy as np

    K = len(p_k_list)

    # 将所有分布向量转为 torch.Tensor 并移至目标设备
    p_tensors = []
    for p in p_k_list:
        if isinstance(p, np.ndarray):
            t = torch.from_numpy(p.astype('float32')).to(device)
        elif isinstance(p, torch.Tensor):
            t = p.float().to(device)
        else:
            raise TypeError(f"p_k_list 中元素类型不支持: {type(p)}")
        p_tensors.append(t)

    num_classes = p_tensors[0].shape[0]

    # 构建代价矩阵（所有对共用同一个 C）
    C = build_cost_matrix(num_classes, device)

    # 初始化距离矩阵
    dist_matrix = torch.zeros(K, K, device=device, dtype=torch.float32)

    # 遍历所有上三角对（对称矩阵只算一半）
    for k in range(K):
        for i in range(k + 1, K):
            d = sinkhorn_distance(
                p=p_tensors[k],
                q=p_tensors[i],
                C=C,
                eps=eps,
                num_iters=num_iters,
            )
            dist_matrix[k, i] = d
            dist_matrix[i, k] = d   # 对称填充

    print(f"[sinkhorn] K×K 距离矩阵计算完成，shape={dist_matrix.shape}，"
          f"min={dist_matrix[dist_matrix > 0].min().item():.4f}，"
          f"max={dist_matrix.max().item():.4f}")

    return dist_matrix