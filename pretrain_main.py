# pretrain_main.py
# 功能：客户端模型预训练入口
# 复用模块：
#   - data/data_utils.py    → prepare_data()，触发 .npy 缓存机制
#   - models/task_model.py  → build_task_model()，构建 T_k
#   - utils/logger.py       → Logger，日志记录
#   - utils/metrics.py      → MetricsTracker（可选，记录各客户端 acc 曲线）
#   - pretrain/trainer.py   → ClientPretrainer，单客户端训练逻辑
#
# 使用方式：
#   串行训练全部客户端：
#     python pretrain_main.py
#   只训练指定客户端：
#     python pretrain_main.py --clients 0 2 5
#   强制覆盖已有 checkpoint：
#     python pretrain_main.py --force
#   并行训练：
#     python pretrain_main.py --parallel
# ============================================================

import os
import sys
import json
import random
import argparse
import numpy as np
from typing import List, Dict, Optional

import torch
from omegaconf import OmegaConf

# ---- 复用已有模块（禁止复制粘贴源码）----
from data.data_utils import prepare_data
from models.task_model import build_task_model, save_teacher_model
from utils.logger import Logger
from utils.metrics import MetricsTracker
from pretrain.trainer import ClientPretrainer


# ------------------------------------------------------------------ #
#  命令行参数解析
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="联邦蒸馏客户端模型预训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 串行训练全部客户端
  python pretrain_main.py

  # 只训练客户端 0、1、3
  python pretrain_main.py --clients 0 1 3

  # 强制覆盖已有 checkpoint
  python pretrain_main.py --force

  # 并行训练（多进程）
  python pretrain_main.py --parallel

  # 指定配置文件
  python pretrain_main.py --config config.yaml
        """,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="配置文件路径（默认：./config.yaml）",
    )
    parser.add_argument(
        "--clients", type=int, nargs="*", default=None,
        help="指定训练哪些客户端（空格分隔的 id），默认全部",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False,
        help="是否并行训练各客户端（多进程），默认串行",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="强制覆盖已有 checkpoint，默认跳过已训练客户端",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="以 key=value 格式覆盖 config 参数",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  模型结构分配
# ------------------------------------------------------------------ #

def assign_model_types(
    num_clients: int,
    assignment: str,
    fixed_type: str,
    seed: int,
) -> Dict[int, str]:
    """
    为每个客户端分配模型结构（Small / Medium / Large）。

    分配规则：
      - random：使用 seed 固定随机性，每次运行结果一致
      - fixed：所有客户端使用 fixed_type 指定的同一结构

    Args:
        num_clients: 客户端总数 K
        assignment:  'random' | 'fixed'
        fixed_type:  仅 assignment='fixed' 时有效
        seed:        随机种子

    Returns:
        {client_id: model_type_str}，例如 {0: 'Small', 1: 'Large', ...}
    """
    model_types = ['small', 'medium', 'large']

    if assignment.lower() == 'fixed':
        ft = fixed_type.lower()
        if ft not in model_types:
            raise ValueError(
                f"fixed_model_type='{fixed_type}' 无效，请使用 Small/Medium/Large"
            )
        return {k: ft for k in range(num_clients)}

    elif assignment.lower() == 'random':
        rng = np.random.default_rng(seed)
        assignments = {}
        for k in range(num_clients):
            assignments[k] = str(rng.choice(model_types))
        return assignments

    else:
        raise ValueError(
            f"model_type_assignment='{assignment}' 无效，请使用 random 或 fixed"
        )


def save_model_type_map(
    type_map: Dict[int, str],
    save_dir: str,
) -> str:
    """
    将模型结构分配结果保存为 JSON 文件。
    供联邦蒸馏主项目加载 T_k 时参考模型结构。

    保存路径：teacher_ckpts/client_model_types.json
    格式：{"0": "small", "1": "large", ...}

    Args:
        type_map: {client_id: model_type}
        save_dir: teacher_ckpts 目录路径

    Returns:
        保存文件的完整路径
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "client_model_types.json")
    serializable = {str(k): v for k, v in type_map.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return save_path


# ------------------------------------------------------------------ #
#  单客户端训练（串行模式调用单元，也作为并行 worker）
# ------------------------------------------------------------------ #

def train_single_client(
    client_id: int,
    model_type: str,
    cfg,
    client_loaders,
    test_loader,
    logger: Logger,
) -> Dict:
    """
    完整训练单个客户端 T_k 并将结果保存为 .pt 文件。

    流程：
      1. 初始化模型
      2. 构建 ClientPretrainer 并执行 run()
      3. 将 best_state 保存至 teacher_ckpts/client_{k}.pt

    Args:
        client_id:      客户端编号 k
        model_type:     模型结构字符串（'small'/'medium'/'large'）
        cfg:            配置对象
        client_loaders: 所有客户端 DataLoader 列表
        test_loader:    全局测试集 DataLoader
        logger:         日志记录器

    Returns:
        result dict，含 client_id/model_type/best_accuracy/best_epoch
    """
    logger.info(
        f"[pretrain] ── Client {client_id} 开始 "
        f"| 模型结构: {model_type.upper()} "
        f"| 本地样本数: {len(client_loaders[client_id].dataset)} ──"
    )

    # ---- 初始化模型 ----
    model = build_task_model(
        model_type=model_type,
        num_classes=cfg.num_classes,
        image_channels=cfg.image_channels,
    )

    # ---- 构建预训练器并执行训练 ----
    pretrainer = ClientPretrainer(
        client_id=client_id,
        model=model,
        train_loader=client_loaders[client_id],
        test_loader=test_loader,
        cfg=cfg,
        logger=logger,
    )
    result = pretrainer.run()

    # ---- 保存最优 checkpoint ----
    ckpt_path = os.path.join(cfg.teacher_ckpt_dir, f"client_{client_id}.pt")
    os.makedirs(cfg.teacher_ckpt_dir, exist_ok=True)

    torch.save(
        {
            # 与 models/task_model.py 的 load_teacher_model 方式A对应
            'model_type':       model_type,
            'state_dict':       result['best_state'],
            # 额外元数据，方便排查问题
            'client_id':        client_id,
            'best_accuracy':    result['best_accuracy'],
            'epoch':            result['best_epoch'],
            'acc_curve':        result['acc_curve'],
        },
        ckpt_path,
    )
    logger.info(
        f"[pretrain] Client {client_id} 模型已保存: {ckpt_path} "
        f"| Best Acc={result['best_accuracy']:.4f} (Epoch {result['best_epoch']})"
    )

    return {
        'client_id':     client_id,
        'model_type':    model_type,
        'best_accuracy': result['best_accuracy'],
        'best_epoch':    result['best_epoch'],
        'num_samples':   len(client_loaders[client_id].dataset),
    }


# ------------------------------------------------------------------ #
#  并行训练 worker（ProcessPoolExecutor 子进程入口）
# ------------------------------------------------------------------ #

def _parallel_worker(args_tuple):
    """
    并行训练的子进程 worker 函数。
    必须为模块级函数（不能是 lambda 或嵌套函数），才能被 pickle 序列化。

    Args:
        args_tuple: (client_id, model_type, cfg_dict, client_loaders, test_loader)
                    注意：cfg 以 dict 传入，子进程内重建为 OmegaConf
    """
    (client_id, model_type, cfg_dict,
     client_loaders, test_loader, log_dir) = args_tuple

    # 子进程内重建配置和日志（每个进程独立）
    cfg = OmegaConf.create(cfg_dict)
    logger = Logger(log_dir=log_dir)

    return train_single_client(
        client_id=client_id,
        model_type=model_type,
        cfg=cfg,
        client_loaders=client_loaders,
        test_loader=test_loader,
        logger=logger,
    )


# ------------------------------------------------------------------ #
#  汇总报告
# ------------------------------------------------------------------ #

def print_summary_table(results: List[Dict]):
    """打印所有客户端预训练结果的汇总表格。"""
    print("\n" + "=" * 70)
    print("  客户端预训练汇总报告")
    print("=" * 70)
    print(f"  {'客户端':^6} | {'模型结构':^8} | {'最优 Accuracy':^13} | "
          f"{'最优 Epoch':^10} | {'样本数':^8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['client_id']):
        print(
            f"  {r['client_id']:^6} | "
            f"{r['model_type'].upper():^8} | "
            f"{r['best_accuracy']:^13.4f} | "
            f"{r['best_epoch']:^10} | "
            f"{r['num_samples']:^8}"
        )
    avg_acc = sum(r['best_accuracy'] for r in results) / len(results)
    print("-" * 70)
    print(f"  {'平均':^6} | {'':^8} | {avg_acc:^13.4f} | {'':^10} | {'':^8}")
    print("=" * 70 + "\n")


def save_summary_json(results: List[Dict], save_dir: str) -> str:
    """
    将汇总结果保存至 teacher_ckpts/pretrain_summary.json。

    Args:
        results:  各客户端结果列表
        save_dir: teacher_ckpts 目录路径

    Returns:
        保存文件路径
    """
    save_path = os.path.join(save_dir, "pretrain_summary.json")
    avg_acc = sum(r['best_accuracy'] for r in results) / len(results)
    summary = {
        'num_clients':    len(results),
        'avg_accuracy':   round(avg_acc, 6),
        'clients':        sorted(results, key=lambda x: x['client_id']),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return save_path


# ------------------------------------------------------------------ #
#  主函数
# ------------------------------------------------------------------ #

def main():
    """预训练主入口。"""
    args = parse_args()

    # ---- 加载配置 ----
    if not os.path.exists(args.config):
        print(f"[pretrain] 错误：配置文件不存在: {args.config}")
        sys.exit(1)
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    # ---- 设置全局随机种子 ----
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ---- 初始化日志 ----
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = Logger(log_dir=cfg.log_dir)
    logger.info("=" * 60)
    logger.info("联邦蒸馏客户端模型预训练启动")
    logger.info(f"数据集: {cfg.dataset.upper()} | 客户端数: {cfg.num_clients}")
    logger.info(f"预训练轮数: {cfg.pretrain_epochs} | 设备: {cfg.device}")
    logger.info("=" * 60)

    # ---- 创建 teacher_ckpts 目录 ----
    os.makedirs(cfg.teacher_ckpt_dir, exist_ok=True)

    # ---- 确定待训练客户端列表 ----
    all_clients = list(range(cfg.num_clients))
    target_clients = args.clients if args.clients is not None else all_clients

    # 过滤非法 client_id
    target_clients = [k for k in target_clients if 0 <= k < cfg.num_clients]
    if not target_clients:
        logger.error(f"待训练客户端列表为空，请检查 --clients 参数")
        sys.exit(1)

    # 跳过已有 checkpoint 的客户端（除非 --force）
    clients_to_train = []
    for k in target_clients:
        ckpt_path = os.path.join(cfg.teacher_ckpt_dir, f"client_{k}.pt")
        if os.path.exists(ckpt_path) and not args.force:
            logger.info(f"[pretrain] Client {k}: checkpoint 已存在，跳过（使用 --force 覆盖）")
        else:
            clients_to_train.append(k)

    if not clients_to_train:
        logger.info("[pretrain] 所有目标客户端均已训练完毕，无需重新训练。")
        logger.info("           使用 --force 参数可强制重新训练。")
        return

    logger.info(f"[pretrain] 待训练客户端: {clients_to_train}")

    # ---- Step 1：数据集划分（复用 data_utils.prepare_data）----
    logger.info("[pretrain] Step 1: 加载/创建数据集划分...")

    # 临时调整 batch_size 为预训练专用值
    original_batch = cfg.gen_batch_size
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['gen_batch_size'] = cfg.pretrain_batch_size
    cfg_pretrain = OmegaConf.create(cfg_dict)

    client_loaders, test_loader, p_k_list, pi_k_list = prepare_data(cfg_pretrain)

    logger.info(f"[pretrain] 数据加载完成，共 {cfg.num_clients} 个客户端")

    # ---- Step 2：模型结构分配 ----
    logger.info("[pretrain] Step 2: 分配客户端模型结构...")
    type_map = assign_model_types(
        num_clients=cfg.num_clients,
        assignment=getattr(cfg, 'model_type_assignment', 'random'),
        fixed_type=getattr(cfg, 'fixed_model_type', 'large'),
        seed=cfg.seed,
    )

    # 打印分配结果
    for k, mt in type_map.items():
        logger.info(f"  Client {k:2d} → {mt.upper()}")

    # 保存 client_model_types.json
    type_map_path = save_model_type_map(type_map, cfg.teacher_ckpt_dir)
    logger.info(f"[pretrain] 模型结构分配已保存: {type_map_path}")

    # ---- Step 3：逐客户端预训练 ----
    logger.info(f"[pretrain] Step 3: 开始预训练（{'并行' if args.parallel else '串行'}模式）...")

    results: List[Dict] = []

    use_parallel = args.parallel or getattr(cfg, 'pretrain_parallel', False)

    if use_parallel:
        # ---- 并行模式：ProcessPoolExecutor ----
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 将 cfg 转为 dict 以便 pickle 序列化传入子进程
        cfg_dict_serial = OmegaConf.to_container(cfg_pretrain, resolve=True)

        worker_args = [
            (k, type_map[k], cfg_dict_serial,
             client_loaders, test_loader, cfg.log_dir)
            for k in clients_to_train
        ]

        # 进程数 = 待训练客户端数，但不超过 CPU 核心数
        max_workers = min(len(clients_to_train), os.cpu_count() or 4)
        logger.info(f"[pretrain] 并行进程数: {max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_parallel_worker, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                client_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"[pretrain] Client {client_id} 完成 | "
                        f"Best Acc={result['best_accuracy']:.4f}"
                    )
                except Exception as e:
                    logger.error(f"[pretrain] Client {client_id} 训练失败: {e}")

    else:
        # ---- 串行模式 ----
        for k in clients_to_train:
            try:
                result = train_single_client(
                    client_id=k,
                    model_type=type_map[k],
                    cfg=cfg_pretrain,
                    client_loaders=client_loaders,
                    test_loader=test_loader,
                    logger=logger,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"[pretrain] Client {k} 训练失败: {e}")
                import traceback
                traceback.print_exc()

    # ---- Step 4：汇总报告 ----
    if results:
        logger.info("[pretrain] Step 4: 生成汇总报告...")
        print_summary_table(results)
        summary_path = save_summary_json(results, cfg.teacher_ckpt_dir)
        logger.info(f"[pretrain] 汇总报告已保存: {summary_path}")
    else:
        logger.error("[pretrain] 没有客户端成功训练，请检查错误日志")

    logger.info("[pretrain] 预训练流程完成。")
    logger.info(
        f"[pretrain] 教师模型已保存至: {cfg.teacher_ckpt_dir}/ "
        f"（client_0.pt ~ client_{cfg.num_clients-1}.pt）"
    )
    logger.info(
        "[pretrain] 现在可以运行联邦蒸馏训练：python main.py"
    )
    logger.finalize(
        final_accuracy=max((r['best_accuracy'] for r in results), default=0.0),
        total_rounds=cfg.pretrain_epochs,
    )


if __name__ == "__main__":
    main()