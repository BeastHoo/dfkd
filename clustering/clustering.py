# clustering/clustering.py
# 功能：基于 Sinkhorn 距离矩阵执行 K-means 聚类，选取簇头节点
# 对应设计文档：基于 Sinkhorn 距离的客户端聚类一节
#
# 完整流程：
#   1. 接收 K×K Sinkhorn 距离矩阵
#   2. 以距离矩阵的行向量作为特征，执行 K-means 聚类
#   3. 每簇随机选取一个客户端作为簇头（seed 固定保证可复现）
#   4. 聚类结果在整个训练过程中固定不变
# ============================================================

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple


# ------------------------------------------------------------------ #
#  聚类主函数
# ------------------------------------------------------------------ #

def cluster_clients(
    dist_matrix: torch.Tensor,
    num_clusters: int,
    seed: int,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    对客户端执行 K-means 聚类，并为每个簇随机选取一个簇头。

    对应设计文档：
      - 采用 K-means 聚类算法将客户端划分为 M 个簇 {C_1,...,C_M}
      - 每簇随机选取一个客户端作为簇头节点
      - 簇划分结果在整个训练过程中保持固定

    Args:
        dist_matrix:  K×K Sinkhorn 距离矩阵（torch.Tensor 或 numpy）
        num_clusters: 簇数 M（config: num_clusters）
        seed:         随机种子（config: seed），保证聚类与簇头选取可复现

    Returns:
        cluster_assignments: dict，key=簇编号(int)，value=该簇客户端编号列表
        cluster_heads:       list，长度 M，cluster_heads[c] 为簇 c 的簇头编号
    """
    # ---- 将距离矩阵转为 numpy（K-means 输入）----
    if isinstance(dist_matrix, torch.Tensor):
        dist_np = dist_matrix.cpu().numpy().astype(np.float64)
    else:
        dist_np = np.array(dist_matrix, dtype=np.float64)

    K = dist_np.shape[0]

    if num_clusters > K:
        raise ValueError(
            f"簇数 M={num_clusters} 超过客户端总数 K={K}，请调整 num_clusters"
        )

    # ---- K-means 聚类（以距离矩阵行向量为特征）----
    # 行向量 dist_np[k] 表示客户端 k 到其他所有客户端的距离，
    # 距离相近的客户端在该特征空间中聚集，符合分布相似则同簇的目标
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=10,          # 多次随机初始化取最优，提升稳定性
        max_iter=300,
    )
    labels = kmeans.fit_predict(dist_np)   # shape = (K,)，每个客户端的簇编号

    # ---- 整理簇划分结果 ----
    cluster_assignments: Dict[int, List[int]] = {c: [] for c in range(num_clusters)}
    for client_id, cluster_id in enumerate(labels):
        cluster_assignments[int(cluster_id)].append(client_id)

    # 检查：确保每个簇至少有一个客户端
    for c in range(num_clusters):
        if len(cluster_assignments[c]) == 0:
            raise RuntimeError(
                f"簇 {c} 为空，请减小 num_clusters 或检查数据分布"
            )

    # ---- 每簇随机选取一个簇头 ----
    rng = np.random.default_rng(seed)
    cluster_heads: List[int] = []
    for c in range(num_clusters):
        members = cluster_assignments[c]
        head = int(rng.choice(members))
        cluster_heads.append(head)

    # ---- 打印聚类结果 ----
    _print_cluster_info(cluster_assignments, cluster_heads, dist_np)

    return cluster_assignments, cluster_heads


# ------------------------------------------------------------------ #
#  辅助：获取客户端所属簇编号
# ------------------------------------------------------------------ #

def get_client_cluster(
    client_id: int,
    cluster_assignments: Dict[int, List[int]],
) -> int:
    """
    查询客户端 client_id 所属的簇编号。

    Args:
        client_id:           客户端编号
        cluster_assignments: 聚类结果字典

    Returns:
        cluster_id: 所属簇编号
    """
    for cluster_id, members in cluster_assignments.items():
        if client_id in members:
            return cluster_id
    raise ValueError(f"客户端 {client_id} 不在任何簇中，请检查聚类结果")


# ------------------------------------------------------------------ #
#  辅助：构建客户端 → 簇头的映射
# ------------------------------------------------------------------ #

def build_client_to_head_map(
    cluster_assignments: Dict[int, List[int]],
    cluster_heads: List[int],
) -> Dict[int, int]:
    """
    构建每个客户端到其所属簇头的映射字典。
    供 server.py 在聚合阶段快速查找使用。

    Args:
        cluster_assignments: 聚类结果字典
        cluster_heads:       簇头列表

    Returns:
        client_to_head: dict，{client_id: head_id}
    """
    client_to_head: Dict[int, int] = {}
    for c, members in cluster_assignments.items():
        head = cluster_heads[c]
        for client_id in members:
            client_to_head[client_id] = head
    return client_to_head


# ------------------------------------------------------------------ #
#  辅助：计算各簇客户端数量
# ------------------------------------------------------------------ #

def get_cluster_sizes(
    cluster_assignments: Dict[int, List[int]],
    num_clusters: int,
) -> List[int]:
    """
    返回各簇的客户端数量列表 N_c。
    对应聚合公式中的 N_c（簇规模）。

    Args:
        cluster_assignments: 聚类结果字典
        num_clusters:        簇总数 M

    Returns:
        sizes: List[int]，长度 M，sizes[c] = 簇 c 的客户端数
    """
    return [len(cluster_assignments[c]) for c in range(num_clusters)]


# ------------------------------------------------------------------ #
#  辅助：打印聚类结果
# ------------------------------------------------------------------ #

def _print_cluster_info(
    cluster_assignments: Dict[int, List[int]],
    cluster_heads: List[int],
    dist_np: np.ndarray,
) -> None:
    """打印聚类结果的可读摘要，便于调试和实验记录。"""
    print("[clustering] ── 聚类结果 ──────────────────────────")
    for c, members in cluster_assignments.items():
        head = cluster_heads[c]
        # 计算簇内平均 Sinkhorn 距离（衡量簇内相似度）
        if len(members) > 1:
            intra_dists = [
                dist_np[i, j]
                for idx_i, i in enumerate(members)
                for j in members[idx_i + 1:]
            ]
            avg_dist = float(np.mean(intra_dists))
        else:
            avg_dist = 0.0
        print(
            f"  簇 {c}: 成员={members}，簇头={head}，"
            f"簇内平均距离={avg_dist:.4f}"
        )
    print("[clustering] ────────────────────────────────────────")


# ------------------------------------------------------------------ #
#  聚类结果打包（供 checkpoint 保存）
# ------------------------------------------------------------------ #

def pack_cluster_result(
    cluster_assignments: Dict[int, List[int]],
    cluster_heads: List[int],
) -> dict:
    """
    将聚类结果打包为可序列化的 dict，用于 checkpoint 保存。

    Returns:
        {'assignments': {c: [client_ids]}, 'heads': [head_ids]}
    """
    return {
        'assignments': {int(c): list(v) for c, v in cluster_assignments.items()},
        'heads': [int(h) for h in cluster_heads],
    }


def unpack_cluster_result(
    packed: dict,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    从 checkpoint 中恢复聚类结果。

    Args:
        packed: pack_cluster_result 返回的 dict

    Returns:
        (cluster_assignments, cluster_heads)
    """
    cluster_assignments = {int(c): list(v) for c, v in packed['assignments'].items()}
    cluster_heads = [int(h) for h in packed['heads']]
    return cluster_assignments, cluster_heads