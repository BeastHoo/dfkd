# utils/logger.py
# 功能：训练过程日志记录，支持控制台输出、文本日志文件、结构化 JSON 记录
# ============================================================

import os
import json
import logging
from datetime import datetime
from typing import Optional


class Logger:
    """
    联邦蒸馏训练日志记录器。

    输出三路：
      1. 控制台（stdout）：实时打印关键信息
      2. 文本日志文件（training.log）：完整训练记录，含时间戳
      3. 结构化 JSON 文件（training_log.json）：每轮详细数据，
         便于后续用 Python/pandas 分析和绘图
    """

    def __init__(self, log_dir: str):
        """
        初始化日志记录器。

        Args:
            log_dir: 日志文件保存目录（config: log_dir）
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ---- 生成带时间戳的日志文件名，避免覆盖历史记录 ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file   = os.path.join(log_dir, f"training_{timestamp}.log")
        self.json_file  = os.path.join(log_dir, f"training_{timestamp}.json")
        # latest 软链接方便脚本直接读取最新日志
        self.latest_log  = os.path.join(log_dir, "latest.log")
        self.latest_json = os.path.join(log_dir, "latest.json")

        # ---- 配置 Python logging ----
        self.logger = logging.getLogger(f"FedDistill_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # 不传播至 root logger，避免重复输出

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )

        # 文件 handler（完整记录，含 DEBUG 级）
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ---- JSON 结构化日志（追加写入）----
        self._json_records = []   # 内存中暂存，训练结束或每轮后刷盘

        self.logger.info(f"日志记录器初始化完成 | log_dir={log_dir}")
        self.logger.info(f"文本日志: {self.log_file}")
        self.logger.info(f"JSON 日志: {self.json_file}")

        # 当前轮开始时间（用于计算单轮耗时）
        self._round_start_time: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  轮次控制
    # ------------------------------------------------------------------ #

    def start_round(self, round_idx: int):
        """
        标记一轮训练的开始，记录时间戳（用于计算耗时）。

        Args:
            round_idx: 当前轮次编号
        """
        self._round_start_time = datetime.now()
        self.logger.debug(f"Round {round_idx} 开始")

    # ------------------------------------------------------------------ #
    #  每轮日志记录
    # ------------------------------------------------------------------ #

    def log_round(
        self,
        round_idx: int,
        loss_g: float,
        loss_conf: float,
        loss_div: float,
        loss_adv: float,
        loss_s: float,
        loss_d: float,
        accuracy: float,
        agg_info: dict,
    ):
        """
        记录每轮训练的完整日志。

        Args:
            round_idx:  当前轮次
            loss_g:     生成器总损失 L_G
            loss_conf:  L_conf 分量
            loss_div:   L_div 分量
            loss_adv:   L_adv 分量
            loss_s:     中心模型蒸馏损失 L_S
            loss_d:     判别器平均损失 L_D（所有客户端均值）
            accuracy:   测试集 top-1 accuracy（未评估轮为 0.0）
            agg_info:   聚合权重信息 dict，含 'intra'/'inter' 权重
        """
        # 计算本轮耗时
        elapsed = ""
        if self._round_start_time is not None:
            delta = (datetime.now() - self._round_start_time).total_seconds()
            elapsed = f"{delta:.1f}s"

        # ---- 文本日志 ----
        self.logger.info(
            f"Round {round_idx:04d} | "
            f"L_G={loss_g:.4f} "
            f"(conf={loss_conf:.4f} div={loss_div:.4f} adv={loss_adv:.4f}) | "
            f"L_S={loss_s:.4f} | "
            f"L_D={loss_d:.4f} | "
            f"Acc={accuracy:.4f} | "
            f"耗时={elapsed}"
        )

        # 记录聚合权重（DEBUG 级，不污染控制台）
        if agg_info:
            # 簇间权重 β_c
            inter = agg_info.get('inter', {})
            if inter:
                beta_str = " ".join(
                    [f"簇{c}:{b:.3f}" for c, b in sorted(inter.items())]
                )
                self.logger.debug(f"  簇间权重 β: {beta_str}")

            # 簇内权重 α_k（每簇前3个客户端）
            intra = agg_info.get('intra', {})
            for c, alpha_dict in intra.items():
                top3 = sorted(alpha_dict.items(), key=lambda x: -x[1])[:3]
                alpha_str = " ".join([f"C{k}:{a:.3f}" for k, a in top3])
                self.logger.debug(f"  簇{c}内权重 α (top3): {alpha_str}")

        # ---- JSON 结构化记录 ----
        record = {
            "round":      round_idx,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":  elapsed,
            "losses": {
                "L_G":    round(loss_g,    6),
                "L_conf": round(loss_conf, 6),
                "L_div":  round(loss_div,  6),
                "L_adv":  round(loss_adv,  6),
                "L_S":    round(loss_s,    6),
                "L_D":    round(loss_d,    6),
            },
            "accuracy": round(accuracy, 6),
            "agg_weights": {
                "inter": {
                    str(c): round(b, 6)
                    for c, b in agg_info.get('inter', {}).items()
                },
                "intra": {
                    str(c): {
                        str(k): round(a, 6)
                        for k, a in alpha_d.items()
                    }
                    for c, alpha_d in agg_info.get('intra', {}).items()
                },
            },
        }
        self._json_records.append(record)

        # 每轮刷盘（保证训练中断时日志不丢失）
        self._flush_json()

        # 更新 latest 软链接（覆盖写）
        self._write_latest()

    # ------------------------------------------------------------------ #
    #  信息/警告/错误日志（供其他模块调用）
    # ------------------------------------------------------------------ #

    def info(self, msg: str):
        """记录 INFO 级日志。"""
        self.logger.info(msg)

    def debug(self, msg: str):
        """记录 DEBUG 级日志（仅写文件，不输出控制台）。"""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """记录 WARNING 级日志。"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """记录 ERROR 级日志。"""
        self.logger.error(msg)

    # ------------------------------------------------------------------ #
    #  JSON 刷盘
    # ------------------------------------------------------------------ #

    def _flush_json(self):
        """将内存中的 JSON 记录写入磁盘（全量覆盖写）。"""
        try:
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(self._json_records, f, ensure_ascii=False, indent=2)
        except IOError as e:
            self.logger.error(f"JSON 日志写入失败: {e}")

    def _write_latest(self):
        """将当前 JSON 记录覆盖写入 latest.json（方便实时监控）。"""
        try:
            with open(self.latest_json, "w", encoding="utf-8") as f:
                json.dump(self._json_records, f, ensure_ascii=False, indent=2)
        except IOError:
            pass  # latest 写失败不影响主流程

    # ------------------------------------------------------------------ #
    #  训练结束日志
    # ------------------------------------------------------------------ #

    def finalize(self, final_accuracy: float, total_rounds: int):
        """
        记录训练结束摘要。

        Args:
            final_accuracy: 最终测试集 accuracy
            total_rounds:   总训练轮数
        """
        self.logger.info("=" * 60)
        self.logger.info(f"训练完成！总轮数={total_rounds} | 最终 Accuracy={final_accuracy:.4f}")
        self.logger.info(f"JSON 日志已保存: {self.json_file}")
        self.logger.info("=" * 60)

        # 关闭所有 handler，释放文件句柄
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)