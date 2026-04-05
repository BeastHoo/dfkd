# utils/metrics.py
# 功能：训练指标追踪，记录 accuracy/loss 曲线，支持导出 CSV 与绘图
# ============================================================

import os
import csv
import json
from typing import Dict, List, Optional


class MetricsTracker:
    """
    训练指标追踪器。

    记录内容：
      - 每轮损失：L_G / L_S / L_D
      - 每 eval_every 轮的测试集 top-1 accuracy
      - 支持导出 CSV（便于 Excel/pandas 分析）
      - 支持调用 matplotlib 绘制损失/accuracy 曲线（可选依赖）
    """

    def __init__(self):
        """初始化空的指标容器。"""

        # ---- 损失记录：每轮均记录 ----
        # {round_idx: {'loss_g': float, 'loss_s': float, 'loss_d': float}}
        self.loss_records: Dict[int, Dict[str, float]] = {}

        # ---- Accuracy 记录：仅评估轮记录 ----
        # {round_idx: accuracy_float}
        self.accuracy_records: Dict[int, float] = {}

        # ---- 有序轮次列表（用于曲线绘制）----
        self.loss_rounds:     List[int] = []
        self.accuracy_rounds: List[int] = []

        # ---- 最优 accuracy 追踪 ----
        self.best_accuracy: float = 0.0
        self.best_round:    int   = 0

    # ------------------------------------------------------------------ #
    #  记录接口
    # ------------------------------------------------------------------ #

    def record_losses(
        self,
        round_idx: int,
        loss_g: float,
        loss_s: float,
        loss_d: float,
    ):
        """
        记录单轮损失值。由 server.py 每轮调用。

        Args:
            round_idx: 当前轮次
            loss_g:    生成器总损失 L_G
            loss_s:    中心模型蒸馏损失 L_S
            loss_d:    判别器平均损失 L_D
        """
        self.loss_records[round_idx] = {
            'loss_g': loss_g,
            'loss_s': loss_s,
            'loss_d': loss_d,
        }
        self.loss_rounds.append(round_idx)

    def record_accuracy(self, round_idx: int, accuracy: float):
        """
        记录评估轮的测试集 top-1 accuracy。
        每 eval_every 轮由 server.py 调用。

        Args:
            round_idx: 当前轮次
            accuracy:  top-1 accuracy，范围 [0, 1]
        """
        self.accuracy_records[round_idx] = accuracy
        self.accuracy_rounds.append(round_idx)

        # 更新最优记录
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_round = round_idx
            print(
                f"[metrics] 新最优 Accuracy: {accuracy:.4f} "
                f"(Round {round_idx})"
            )

    # ------------------------------------------------------------------ #
    #  统计摘要
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """
        返回训练统计摘要。

        Returns:
            dict：包含最优 accuracy、最终损失等关键指标
        """
        if not self.loss_records:
            return {}

        last_round = self.loss_rounds[-1]
        last_losses = self.loss_records[last_round]

        summary = {
            'total_rounds':   len(self.loss_rounds),
            'best_accuracy':  self.best_accuracy,
            'best_round':     self.best_round,
            'final_loss_g':   last_losses.get('loss_g', 0.),
            'final_loss_s':   last_losses.get('loss_s', 0.),
            'final_loss_d':   last_losses.get('loss_d', 0.),
        }

        # 最近 10 轮平均损失（衡量收敛状态）
        recent_rounds = self.loss_rounds[-10:]
        if recent_rounds:
            recent_g = [self.loss_records[r]['loss_g'] for r in recent_rounds]
            recent_s = [self.loss_records[r]['loss_s'] for r in recent_rounds]
            summary['recent10_avg_loss_g'] = sum(recent_g) / len(recent_g)
            summary['recent10_avg_loss_s'] = sum(recent_s) / len(recent_s)

        return summary

    def print_summary(self):
        """打印训练统计摘要至控制台。"""
        s = self.summary()
        if not s:
            print("[metrics] 暂无记录")
            return

        print("\n" + "=" * 50)
        print("  训练统计摘要")
        print("=" * 50)
        print(f"  总轮数:          {s['total_rounds']}")
        print(f"  最优 Accuracy:   {s['best_accuracy']:.4f}  (Round {s['best_round']})")
        print(f"  最终 L_G:        {s['final_loss_g']:.4f}")
        print(f"  最终 L_S:        {s['final_loss_s']:.4f}")
        print(f"  最终 L_D:        {s['final_loss_d']:.4f}")
        if 'recent10_avg_loss_g' in s:
            print(f"  近10轮平均 L_G:  {s['recent10_avg_loss_g']:.4f}")
            print(f"  近10轮平均 L_S:  {s['recent10_avg_loss_s']:.4f}")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------ #
    #  导出 CSV
    # ------------------------------------------------------------------ #

    def save_csv(self, save_dir: str):
        """
        将损失曲线和 accuracy 曲线分别导出为 CSV 文件。

        生成文件：
          - losses.csv：每轮损失，列：round, loss_g, loss_s, loss_d
          - accuracy.csv：评估轮 accuracy，列：round, accuracy

        Args:
            save_dir: 保存目录（通常为 cfg.log_dir）
        """
        os.makedirs(save_dir, exist_ok=True)

        # ---- 损失曲线 CSV ----
        loss_csv = os.path.join(save_dir, "losses.csv")
        with open(loss_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "loss_g", "loss_s", "loss_d"]
            )
            writer.writeheader()
            for r in self.loss_rounds:
                row = {"round": r}
                row.update(self.loss_records[r])
                writer.writerow(row)
        print(f"[metrics] 损失曲线已保存: {loss_csv}")

        # ---- Accuracy 曲线 CSV ----
        acc_csv = os.path.join(save_dir, "accuracy.csv")
        with open(acc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "accuracy"])
            writer.writeheader()
            for r in self.accuracy_rounds:
                writer.writerow({
                    "round":    r,
                    "accuracy": self.accuracy_records[r],
                })
        print(f"[metrics] Accuracy 曲线已保存: {acc_csv}")

    def save_json(self, save_dir: str):
        """
        将所有指标导出为单个 JSON 文件（metrics.json）。
        结构与 logger.py 的 JSON 格式互补，便于对照分析。

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        metrics_json = os.path.join(save_dir, "metrics.json")

        data = {
            "summary":   self.summary(),
            "losses":    [
                {"round": r, **self.loss_records[r]}
                for r in self.loss_rounds
            ],
            "accuracy":  [
                {"round": r, "accuracy": self.accuracy_records[r]}
                for r in self.accuracy_rounds
            ],
        }

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[metrics] 指标 JSON 已保存: {metrics_json}")

    def save(self, save_dir: str):
        """
        统一保存接口：同时导出 CSV 和 JSON，并打印摘要。
        由 server.py 在训练结束时调用。

        Args:
            save_dir: 保存目录（cfg.log_dir）
        """
        self.print_summary()
        self.save_csv(save_dir)
        self.save_json(save_dir)

    # ------------------------------------------------------------------ #
    #  可视化（可选，需要 matplotlib）
    # ------------------------------------------------------------------ #

    def plot(
        self,
        save_dir: str,
        show: bool = False,
    ):
        """
        绘制损失曲线与 accuracy 曲线并保存为 PNG。
        matplotlib 为可选依赖，未安装时打印提示而不报错。

        生成文件：
          - loss_curves.png：L_G / L_S / L_D 三条曲线
          - accuracy_curve.png：测试集 top-1 accuracy 曲线

        Args:
            save_dir: 图片保存目录
            show:     是否调用 plt.show() 显示交互窗口
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # 非交互后端，服务器环境无需显示器
            import matplotlib.pyplot as plt
        except ImportError:
            print("[metrics] matplotlib 未安装，跳过绘图。"
                  "可通过 pip install matplotlib 安装。")
            return

        os.makedirs(save_dir, exist_ok=True)

        # ---- 损失曲线 ----
        if self.loss_rounds:
            fig, ax = plt.subplots(figsize=(10, 5))
            rounds = self.loss_rounds
            ax.plot(rounds,
                    [self.loss_records[r]['loss_g'] for r in rounds],
                    label="L_G（生成器）", linewidth=1.5)
            ax.plot(rounds,
                    [self.loss_records[r]['loss_s'] for r in rounds],
                    label="L_S（中心模型）", linewidth=1.5)
            ax.plot(rounds,
                    [self.loss_records[r]['loss_d'] for r in rounds],
                    label="L_D（判别器均值）", linewidth=1.5, linestyle="--")
            ax.set_xlabel("训练轮次 (Round)")
            ax.set_ylabel("损失值")
            ax.set_title("联邦蒸馏训练损失曲线")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            loss_png = os.path.join(save_dir, "loss_curves.png")
            fig.savefig(loss_png, dpi=150)
            plt.close(fig)
            print(f"[metrics] 损失曲线图已保存: {loss_png}")

        # ---- Accuracy 曲线 ----
        if self.accuracy_rounds:
            fig, ax = plt.subplots(figsize=(8, 4))
            rounds = self.accuracy_rounds
            accs   = [self.accuracy_records[r] for r in rounds]
            ax.plot(rounds, accs,
                    label="Top-1 Accuracy", color="steelblue",
                    linewidth=2, marker="o", markersize=4)

            # 标注最优点
            ax.axhline(
                y=self.best_accuracy, color="red",
                linestyle="--", linewidth=1,
                label=f"最优 {self.best_accuracy:.4f} (Round {self.best_round})"
            )
            ax.set_xlabel("训练轮次 (Round)")
            ax.set_ylabel("Top-1 Accuracy")
            ax.set_title("中心模型测试集 Accuracy 曲线")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            acc_png = os.path.join(save_dir, "accuracy_curve.png")
            fig.savefig(acc_png, dpi=150)
            plt.close(fig)
            print(f"[metrics] Accuracy 曲线图已保存: {acc_png}")

        if show:
            plt.show()