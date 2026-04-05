# pretrain/trainer.py
# 功能：单客户端预训练器，封装完整的训练/评估/best-checkpoint逻辑
# 复用模块：
#   - models/task_model.py  （模型结构，直接 import，不复制代码）
#   - utils/logger.py       （日志记录）
# ============================================================

import copy
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ClientPretrainer:
    """
    单客户端本地任务模型预训练器。

    职责：
      - 在客户端本地数据上用标准交叉熵训练 T_k
      - 支持 CosineAnnealingLR / StepLR 学习率调度
      - 维护 best checkpoint（最优测试集 accuracy 对应的权重）
      - 每轮记录 train_loss 和 test_accuracy
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cfg,
        logger,
    ):
        """
        初始化预训练器。

        Args:
            client_id:    客户端编号 k
            model:        已初始化的任务模型 T_k（未训练）
            train_loader: 客户端本地训练数据 DataLoader
            test_loader:  全局测试集 DataLoader（用于评估）
            cfg:          OmegaConf 配置对象
            logger:       utils/logger.py 的 Logger 实例
        """
        self.client_id   = client_id
        self.model       = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.cfg         = cfg
        self.logger      = logger

        # 设备
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # ---- 设置客户端专属随机种子 ----
        # torch.manual_seed(seed + client_id) 保证不同客户端有不同但可复现的随机性
        torch.manual_seed(cfg.seed + client_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed + client_id)

        # ---- 损失函数 ----
        self.criterion = nn.CrossEntropyLoss()

        # ---- SGD 优化器 ----
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.pretrain_lr,
            momentum=cfg.pretrain_momentum,
            weight_decay=cfg.pretrain_weight_decay,
            nesterov=True,   # Nesterov momentum 通常比标准 momentum 收敛更快
        )

        # ---- 学习率调度器 ----
        scheduler_type = getattr(cfg, 'pretrain_lr_scheduler', 'cosine').lower()
        if scheduler_type == 'cosine':
            # CosineAnnealingLR：从 pretrain_lr 余弦衰减至接近 0
            # T_max = pretrain_epochs，覆盖完整训练周期
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.pretrain_epochs,
                eta_min=1e-4,   # 最小学习率不降至 0，保留微调能力
            )
        elif scheduler_type == 'step':
            # StepLR：每 pretrain_lr_step 轮乘以 pretrain_lr_gamma
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.pretrain_lr_step,
                gamma=cfg.pretrain_lr_gamma,
            )
        else:
            raise ValueError(
                f"不支持的学习率调度器: {scheduler_type}，"
                f"请在 config.yaml 中设置 pretrain_lr_scheduler: cosine 或 step"
            )

        # ---- Best checkpoint 状态 ----
        self.best_accuracy: float = 0.0
        self.best_epoch:    int   = 0
        self.best_state:    dict  = {}   # 深拷贝的最优 state_dict

    # ------------------------------------------------------------------ #
    #  单轮训练
    # ------------------------------------------------------------------ #

    def train_one_epoch(self, epoch: int) -> float:
        """
        在本地训练集上训练一轮，返回平均 train_loss。

        使用标准交叉熵损失，SGD + momentum 更新参数。

        Args:
            epoch: 当前轮次编号（仅用于 tqdm 描述）

        Returns:
            avg_loss: 本轮平均训练损失（float）
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        # ---- tqdm 进度条（每个 epoch 内的 batch 进度）----
        desc = f"Client {self.client_id} | Epoch {epoch}/{self.cfg.pretrain_epochs}"
        if TQDM_AVAILABLE:
            loader = tqdm(self.train_loader, desc=desc, leave=False, ncols=90)
        else:
            loader = self.train_loader

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)                    # (B, num_classes)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # 梯度裁剪，防止深层网络（Large/ResNet）梯度爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size   = images.size(0)
            total_loss   += loss.item() * batch_size
            total_samples += batch_size

            # 实时更新 tqdm 后缀
            if TQDM_AVAILABLE:
                loader.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / total_samples if total_samples > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  评估
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        在全局测试集上计算 top-1 accuracy。

        使用 torch.no_grad() 节省显存，模型切换为 eval 模式
        关闭 BatchNorm/Dropout 的训练行为。

        Returns:
            accuracy: float，范围 [0, 1]
        """
        self.model.eval()
        correct = 0
        total   = 0

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  完整预训练流程
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """
        执行完整 pretrain_epochs 轮训练，维护 best checkpoint。

        流程：
          for epoch in 1..pretrain_epochs:
            1. train_one_epoch → train_loss
            2. scheduler.step()
            3. evaluate → acc
            4. 若 acc > best_accuracy → 更新 best checkpoint（深拷贝 state_dict）
            5. 记录日志

        Best Checkpoint 策略：
          只保存测试集 accuracy 最高时刻的 state_dict（内存中），
          训练结束后由 pretrain_main.py 写入磁盘，
          避免每轮都写磁盘造成 IO 瓶颈。

        Returns:
            dict：
              {
                'best_accuracy': float,       最优测试集 accuracy
                'best_epoch':    int,         最优 accuracy 出现的轮次
                'best_state':    OrderedDict, 最优模型 state_dict（深拷贝）
                'acc_curve':     List[float], 每轮 accuracy 列表
                'loss_curve':    List[float], 每轮 train_loss 列表
              }
        """
        acc_curve:  List[float] = []
        loss_curve: List[float] = []

        self.logger.info(
            f"[预训练] Client {self.client_id} 开始训练，"
            f"共 {self.cfg.pretrain_epochs} 轮，"
            f"设备: {self.device}"
        )

        # ---- 外层 epoch 进度条 ----
        epoch_range = range(1, self.cfg.pretrain_epochs + 1)
        if TQDM_AVAILABLE:
            epoch_iter = tqdm(
                epoch_range,
                desc=f"Client {self.client_id:2d} 预训练",
                ncols=90,
                unit="epoch",
            )
        else:
            epoch_iter = epoch_range

        for epoch in epoch_iter:
            # Step 1：训练一轮
            train_loss = self.train_one_epoch(epoch)

            # Step 2：更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Step 3：评估
            acc = self.evaluate()

            acc_curve.append(acc)
            loss_curve.append(train_loss)

            # Step 4：Best checkpoint 更新
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_epoch    = epoch
                # 深拷贝 state_dict，避免后续训练覆盖
                self.best_state = copy.deepcopy(self.model.state_dict())

            # Step 5：日志记录
            log_msg = (
                f"[Client {self.client_id}] "
                f"Epoch {epoch:3d}/{self.cfg.pretrain_epochs} | "
                f"Loss={train_loss:.4f} | "
                f"Acc={acc:.4f} | "
                f"Best={self.best_accuracy:.4f}(ep{self.best_epoch}) | "
                f"LR={current_lr:.6f}"
            )
            self.logger.debug(log_msg)

            # tqdm 后缀实时更新
            if TQDM_AVAILABLE:
                epoch_iter.set_postfix(
                    acc=f"{acc:.4f}",
                    best=f"{self.best_accuracy:.4f}",
                    loss=f"{train_loss:.4f}",
                )

        self.logger.info(
            f"[预训练] Client {self.client_id} 训练完成 | "
            f"最优 Accuracy={self.best_accuracy:.4f} (Epoch {self.best_epoch})"
        )

        return {
            'best_accuracy': self.best_accuracy,
            'best_epoch':    self.best_epoch,
            'best_state':    self.best_state,
            'acc_curve':     acc_curve,
            'loss_curve':    loss_curve,
        }