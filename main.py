# main.py
# 功能：程序入口，解析命令行参数，加载 config.yaml，启动训练
# 使用方式：
#   正常训练：python main.py
#   指定配置：python main.py --config config.yaml
#   断点续训：python main.py --resume checkpoints/round_0050.pt
#   覆盖参数：python main.py --override fed_rounds=200 lr_g=0.0001
# ============================================================

import os
import sys
import argparse
import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from server import FedDistillServer


# ------------------------------------------------------------------ #
#  命令行参数解析
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    支持：
      --config:   指定 config.yaml 路径（默认 ./config.yaml）
      --resume:   指定 checkpoint 路径，从该点继续训练
      --override: 以 key=value 格式覆盖 config 中的任意参数
                  例如：--override fed_rounds=200 lr_g=0.0001
    """
    parser = argparse.ArgumentParser(
        description="无数据联邦知识蒸馏训练（Data-Free FedKD）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 正常启动训练
  python main.py

  # 使用自定义配置文件
  python main.py --config my_config.yaml

  # 从第 50 轮 checkpoint 恢复训练
  python main.py --resume checkpoints/round_0050.pt

  # 覆盖部分超参数
  python main.py --override num_clients=20 fed_rounds=200 device=cpu
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认：./config.yaml）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint 文件路径，指定后从该轮继续训练",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="以 key=value 格式覆盖 config 参数，例如：--override lr_g=0.0001 fed_rounds=200",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  配置加载与合并
# ------------------------------------------------------------------ #

def load_config(config_path: str, overrides: list) -> DictConfig:
    """
    加载 config.yaml 并应用命令行覆盖参数。

    Args:
        config_path: config.yaml 文件路径
        overrides:   ['key=value', ...] 格式的覆盖列表

    Returns:
        cfg: OmegaConf DictConfig 对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            f"请确认当前目录下存在 config.yaml，或通过 --config 指定路径"
        )

    # 加载基础配置
    cfg = OmegaConf.load(config_path)

    # 应用命令行覆盖（支持嵌套键，如 sinkhorn.eps=0.1）
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


# ------------------------------------------------------------------ #
#  全局随机种子设置
# ------------------------------------------------------------------ #

def set_seed(seed: int):
    """
    设置全局随机种子，确保实验可复现。
    覆盖：Python random、NumPy、PyTorch（CPU + CUDA）。

    Args:
        seed: 随机种子（config: seed）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 多 GPU 场景

    # 确保 cuDNN 确定性（略微降低速度，保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[main] 全局随机种子已设置: seed={seed}")


# ------------------------------------------------------------------ #
#  配置合法性检查
# ------------------------------------------------------------------ #

def validate_config(cfg: DictConfig):
    """
    检查配置参数的合法性，提前发现常见错误。

    Args:
        cfg: OmegaConf 配置对象
    """
    errors = []

    # 数据集名称
    if cfg.dataset.lower() not in ("cifar10", "cifar100"):
        errors.append(f"dataset 必须为 cifar10 或 cifar100，当前: {cfg.dataset}")

    # 类别数与数据集一致性
    expected_classes = {"cifar10": 10, "cifar100": 100}
    expected = expected_classes.get(cfg.dataset.lower(), -1)
    if cfg.num_classes != expected:
        errors.append(
            f"num_classes={cfg.num_classes} 与数据集 {cfg.dataset} "
            f"不匹配（应为 {expected}）"
        )

    # 簇数不超过客户端数
    if cfg.num_clusters > cfg.num_clients:
        errors.append(
            f"num_clusters={cfg.num_clusters} 超过 num_clients={cfg.num_clients}"
        )

    # 学习率正数
    for lr_key in ("lr_g", "lr_d", "lr_s"):
        if getattr(cfg, lr_key) <= 0:
            errors.append(f"{lr_key}={getattr(cfg, lr_key)} 必须为正数")

    # 损失权重非负
    for lam_key in ("lambda1", "lambda2", "lambda3"):
        if getattr(cfg, lam_key) < 0:
            errors.append(f"{lam_key}={getattr(cfg, lam_key)} 不能为负数")

    # Sinkhorn 参数
    if cfg.sinkhorn_eps <= 0:
        errors.append(f"sinkhorn_eps={cfg.sinkhorn_eps} 必须为正数")
    if cfg.sinkhorn_iters <= 0:
        errors.append(f"sinkhorn_iters={cfg.sinkhorn_iters} 必须为正整数")

    # 温度参数
    if cfg.temperature <= 0:
        errors.append(f"temperature={cfg.temperature} 必须为正数")
    if cfg.agg_temperature <= 0:
        errors.append(f"agg_temperature={cfg.agg_temperature} 必须为正数")

    # checkpoint 路径
    if cfg.save_ckpt_every <= 0:
        errors.append(f"save_ckpt_every={cfg.save_ckpt_every} 必须为正整数")

    if errors:
        print("[main] 配置参数错误：")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    print("[main] 配置参数检查通过 ✓")


# ------------------------------------------------------------------ #
#  打印配置摘要
# ------------------------------------------------------------------ #

def print_config_summary(cfg: DictConfig):
    """打印关键配置参数摘要，便于实验记录。"""
    print("\n" + "=" * 60)
    print("  无数据联邦知识蒸馏（Data-Free FedKD）配置摘要")
    print("=" * 60)
    print(f"  数据集:        {cfg.dataset.upper()} ({cfg.num_classes} 类)")
    print(f"  客户端数:      K = {cfg.num_clients}")
    print(f"  簇数:          M = {cfg.num_clusters}")
    print(f"  Dirichlet α:   {cfg.dirichlet_alpha}")
    print(f"  训练轮数:      {cfg.fed_rounds}")
    print(f"  生成 batch:    {cfg.gen_batch_size}")
    print(f"  蒸馏温度 τ:    {cfg.temperature}")
    print(f"  聚合温度 T_agg:{cfg.agg_temperature}")
    print(f"  规模指数 γ:    {cfg.gamma}")
    print(f"  Sinkhorn ε:    {cfg.sinkhorn_eps}  iters={cfg.sinkhorn_iters}")
    print(f"  损失权重:      λ1={cfg.lambda1} λ2={cfg.lambda2} λ3={cfg.lambda3}")
    print(f"  学习率:        lr_g={cfg.lr_g}  lr_d={cfg.lr_d}  lr_s={cfg.lr_s}")
    print(f"  更新步数:      D={cfg.local_gan_steps} G={cfg.generator_steps} S={cfg.central_steps}")
    print(f"  设备:          {cfg.device}")
    print(f"  随机种子:      {cfg.seed}")
    print(f"  教师模型目录:  {cfg.teacher_ckpt_dir}")
    print(f"  Checkpoint 目录:{cfg.ckpt_dir}")
    print("=" * 60 + "\n")


# ------------------------------------------------------------------ #
#  主函数
# ------------------------------------------------------------------ #

def main():
    """程序主入口。"""

    # ---- 解析命令行参数 ----
    args = parse_args()

    # ---- 加载并合并配置 ----
    cfg = load_config(args.config, args.override)

    # ---- 合法性检查 ----
    validate_config(cfg)

    # ---- 打印配置摘要 ----
    print_config_summary(cfg)

    # ---- 设置全局随机种子 ----
    set_seed(cfg.seed)

    # ---- 创建必要目录 ----
    for dir_path in (cfg.data_dir, cfg.partition_dir, cfg.ckpt_dir, cfg.log_dir):
        os.makedirs(dir_path, exist_ok=True)

    # ---- 检查教师模型目录 ----
    if not os.path.exists(cfg.teacher_ckpt_dir):
        print(
            f"[main] 警告：教师模型目录不存在: {cfg.teacher_ckpt_dir}\n"
            f"       请将各客户端预训练模型（client_0.pt ~ client_{cfg.num_clients-1}.pt）"
            f"放入该目录后重新运行。"
        )
        sys.exit(1)

    # ---- 检查 resume checkpoint 是否存在 ----
    if args.resume is not None and not os.path.exists(args.resume):
        print(f"[main] 错误：指定的 checkpoint 文件不存在: {args.resume}")
        sys.exit(1)

    # ---- 初始化服务器并启动训练 ----
    print("[main] 初始化 FedDistillServer...")
    server = FedDistillServer(cfg=cfg)

    print("[main] 启动训练...")
    server.train(resume_path=args.resume)

    print("[main] 训练流程完成。")


if __name__ == "__main__":
    main()