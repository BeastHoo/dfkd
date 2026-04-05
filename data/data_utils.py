# data/data_utils.py
# 功能：数据集加载、Dirichlet 非IID划分、.npy 缓存、DataLoader 构建
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict


# ------------------------------------------------------------------ #
#  1. 数据集变换配置
# ------------------------------------------------------------------ #

def get_transforms(dataset: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    返回训练集和测试集的图像预处理变换。

    Args:
        dataset: 数据集名称，'cifar10' 或 'cifar100'

    Returns:
        (train_transform, test_transform)
    """
    if dataset.lower() in ("cifar10", "cifar100"):
        # CIFAR 标准归一化参数
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError(f"不支持的数据集: {dataset}，请使用 cifar10 或 cifar100")

    return train_transform, test_transform


# ------------------------------------------------------------------ #
#  2. 原始数据集加载
# ------------------------------------------------------------------ #

def load_raw_dataset(dataset: str, data_dir: str):
    """
    下载并加载原始训练集和测试集（不划分，返回完整数据集对象）。

    Args:
        dataset:  数据集名称
        data_dir: 数据存储根目录

    Returns:
        (train_dataset, test_dataset)
    """
    train_transform, test_transform = get_transforms(dataset)
    os.makedirs(data_dir, exist_ok=True)

    name = dataset.lower()
    if name == "cifar10":
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
        test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    elif name == "cifar100":
        train_ds = datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_transform)
        test_ds  = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    return train_ds, test_ds


# ------------------------------------------------------------------ #
#  3. Dirichlet 非IID 划分
# ------------------------------------------------------------------ #

def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    使用 Dirichlet(α) 分布将训练样本索引划分给各客户端，实现非IID。
    α 越小，数据分布越极端（每个客户端倾向只含少数类别）。

    Args:
        targets:     全部训练样本标签数组，shape=(N,)
        num_clients: 客户端数量 K
        num_classes: 类别数
        alpha:       Dirichlet 参数 α
        seed:        随机种子，保证可复现

    Returns:
        dict: {client_id(int): 样本索引数组(np.ndarray)}
    """
    rng = np.random.default_rng(seed)

    # 按类别整理样本索引
    class_indices: List[np.ndarray] = [
        np.where(targets == c)[0] for c in range(num_classes)
    ]

    # 每个客户端的索引桶
    client_indices: Dict[int, List[int]] = {k: [] for k in range(num_clients)}

    for c in range(num_classes):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)

        # 从 Dirichlet 分布采样每个客户端获得该类别的比例
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # 按比例切分索引
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(idx_c, splits)):
            client_indices[k].extend(chunk.tolist())

    # 转换为 np.ndarray 并打乱顺序
    partition: Dict[int, np.ndarray] = {}
    for k in range(num_clients):
        arr = np.array(client_indices[k], dtype=np.int64)
        rng.shuffle(arr)
        partition[k] = arr

    return partition


# ------------------------------------------------------------------ #
#  4. .npy 缓存：加载或创建划分
# ------------------------------------------------------------------ #

def _partition_filename(dataset: str, alpha: float, num_clients: int, seed: int) -> str:
    """生成划分缓存文件名，格式与设计文档一致。"""
    return f"{dataset.lower()}_alpha{alpha}_clients{num_clients}_seed{seed}.npy"


def load_or_create_partition(
    dataset: str,
    data_dir: str,
    partition_dir: str,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    优先从 .npy 文件加载已有划分；若不存在则执行 Dirichlet 划分并保存。

    Args:
        dataset:       数据集名称
        data_dir:      原始数据路径（用于读取标签）
        partition_dir: .npy 文件存储目录
        num_clients:   客户端数 K
        num_classes:   类别数
        alpha:         Dirichlet α
        seed:          随机种子

    Returns:
        dict: {client_id: 样本索引数组}
    """
    os.makedirs(partition_dir, exist_ok=True)
    fname = _partition_filename(dataset, alpha, num_clients, seed)
    fpath = os.path.join(partition_dir, fname)

    if os.path.exists(fpath):
        # ---- 直接加载缓存，跳过重新划分 ----
        print(f"[data_utils] 加载已有划分文件: {fpath}")
        partition = np.load(fpath, allow_pickle=True).item()
        return partition

    # ---- 文件不存在，执行 Dirichlet 划分 ----
    print(f"[data_utils] 未找到划分文件，执行 Dirichlet(α={alpha}) 划分...")

    # 只加载训练集标签（不需要图像）
    _, _ = get_transforms(dataset)   # 确保 dataset 名称合法
    name = dataset.lower()
    if name == "cifar10":
        raw_ds = datasets.CIFAR10(data_dir, train=True, download=True)
    elif name == "cifar100":
        raw_ds = datasets.CIFAR100(data_dir, train=True, download=True)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    targets = np.array(raw_ds.targets, dtype=np.int64)
    partition = dirichlet_partition(targets, num_clients, num_classes, alpha, seed)

    # 保存为 .npy
    np.save(fpath, partition)
    print(f"[data_utils] 划分结果已保存至: {fpath}")

    return partition


# ------------------------------------------------------------------ #
#  5. 计算客户端类别分布向量与全局比例权重
# ------------------------------------------------------------------ #

def compute_distribution(
    partition: Dict[int, np.ndarray],
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    计算每个客户端的类别分布向量 p_k 和全局比例权重 π_k。

    p_k[c] = 客户端 k 中类别 c 的样本数 / 客户端 k 的总样本数
    π_k    = 客户端 k 的样本数 / 所有客户端总样本数

    Args:
        partition:   划分字典
        targets:     全局训练集标签数组
        num_clients: K
        num_classes: 类别数

    Returns:
        p_k_list:  List[np.ndarray]，长度 K，每项 shape=(num_classes,)
        pi_k_list: List[float]，长度 K，全局比例权重
    """
    total_samples = sum(len(partition[k]) for k in range(num_clients))

    p_k_list: List[np.ndarray] = []
    pi_k_list: List[float] = []

    for k in range(num_clients):
        idx = partition[k]
        local_targets = targets[idx]
        counts = np.bincount(local_targets, minlength=num_classes).astype(np.float64)

        # 类别分布向量（归一化，和为1）
        p_k = counts / (counts.sum() + 1e-8)
        p_k_list.append(p_k)

        # 全局比例权重 π_k
        pi_k = len(idx) / total_samples
        pi_k_list.append(pi_k)

    return p_k_list, pi_k_list


# ------------------------------------------------------------------ #
#  6. 构建 DataLoader
# ------------------------------------------------------------------ #

def build_dataloaders(
    dataset: str,
    data_dir: str,
    partition: Dict[int, np.ndarray],
    num_clients: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    根据划分索引构建各客户端的 DataLoader 和全局测试集 DataLoader。

    【重要】num_workers 默认为 0（主进程加载数据）：
      DataLoader 的 num_workers > 0 会启动子进程并注册 SIGCHLD 信号处理器。
      当 server.py 的 ThreadPoolExecutor 线程在 lock.acquire() 时被该信号
      处理器打断，会触发 _error_if_any_worker_fails() 误报，导致训练崩溃。
      这是 PyTorch DataLoader 多进程与 Python 线程池的已知冲突，
      解决方案是保持 num_workers=0，在主进程内完成数据加载。

    Args:
        dataset:     数据集名称
        data_dir:    原始数据路径
        partition:   划分字典 {client_id: 索引数组}
        num_clients: K
        batch_size:  训练批大小
        num_workers: DataLoader 工作进程数，必须为 0（见上方说明）

    Returns:
        client_loaders: List[DataLoader]，长度 K
        test_loader:    DataLoader（全局测试集，不划分）
    """
    train_transform, test_transform = get_transforms(dataset)
    name = dataset.lower()

    if name == "cifar10":
        train_full = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
        test_ds    = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    elif name == "cifar100":
        train_full = datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_transform)
        test_ds    = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    # 各客户端 DataLoader（num_workers=0，避免与线程池冲突）
    client_loaders: List[DataLoader] = []
    for k in range(num_clients):
        subset = Subset(train_full, partition[k].tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,        # 强制主进程加载，不启动子进程
            pin_memory=False,     # num_workers=0 时 pin_memory 无效，显式关闭
            drop_last=True,
        )
        client_loaders.append(loader)

    # 全局测试集 DataLoader（num_workers=0，避免与 ThreadPoolExecutor 冲突）
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return client_loaders, test_loader


# ------------------------------------------------------------------ #
#  7. 统一入口函数
# ------------------------------------------------------------------ #

def prepare_data(cfg) -> Tuple[
    List[DataLoader],
    DataLoader,
    List[np.ndarray],
    List[float],
]:
    """
    数据模块统一入口，被 server.py 调用。

    完整流程：
      1. 加载或创建 .npy 划分
      2. 计算 p_k_list 和 pi_k_list
      3. 构建所有 DataLoader

    Args:
        cfg: OmegaConf/dict 配置对象，包含所有超参数字段

    Returns:
        client_loaders: List[DataLoader]，长度 K
        test_loader:    DataLoader
        p_k_list:       List[np.ndarray]，各客户端类别分布向量
        pi_k_list:      List[float]，各客户端全局比例权重 π_k
    """
    # 确定类别数
    num_classes = cfg.num_classes

    # Step 1：加载或创建划分
    partition = load_or_create_partition(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        partition_dir=cfg.partition_dir,
        num_clients=cfg.num_clients,
        num_classes=num_classes,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )

    # Step 2：读取全局训练集标签（用于计算分布）
    name = cfg.dataset.lower()
    if name == "cifar10":
        raw_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True)
    elif name == "cifar100":
        raw_ds = datasets.CIFAR100(cfg.data_dir, train=True, download=True)
    else:
        raise ValueError(f"不支持的数据集: {cfg.dataset}")

    targets = np.array(raw_ds.targets, dtype=np.int64)

    # Step 3：计算 p_k 和 π_k
    p_k_list, pi_k_list = compute_distribution(
        partition=partition,
        targets=targets,
        num_clients=cfg.num_clients,
        num_classes=num_classes,
    )

    # Step 4：构建 DataLoader（num_workers=0，避免与 ThreadPoolExecutor 冲突）
    client_loaders, test_loader = build_dataloaders(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        partition=partition,
        num_clients=cfg.num_clients,
        batch_size=cfg.gen_batch_size,
        num_workers=0,
    )

    # 打印各客户端数据量分布（调试信息）
    print("[data_utils] 各客户端样本数分布：")
    for k in range(cfg.num_clients):
        n = len(partition[k])
        top_cls = int(np.argmax(p_k_list[k]))
        print(f"  Client {k:2d}: {n:5d} 样本 | 主要类别={top_cls} "
              f"| π_k={pi_k_list[k]:.4f}")

    return client_loaders, test_loader, p_k_list, pi_k_list