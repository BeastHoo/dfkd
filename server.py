# server.py
# 功能：FedDistillServer 主循环，统筹协调四个训练阶段
# 对应设计文档：训练流程（每轮4个阶段）+ Checkpoint 机制
# ============================================================
#
# 执行顺序：
#   【前置，仅一次】
#     1. 加载数据集划分与 DataLoader
#     2. 加载所有客户端教师模型 T_k
#     3. 计算 Sinkhorn 距离矩阵 → K-means 聚类 → 选簇头
#     4. 初始化 G、S、D_k、优化器
#
#   【每轮 fed_rounds 次】
#     阶段1：G 采样生成伪数据 x
#     阶段2：ThreadPoolExecutor 并行客户端 local_compute（Step 2-A + 2-B）
#     阶段3：hierarchical_aggregate → 全局软标签 Z
#     阶段4：update_generator（Step 4-A）→ update_central_model（Step 4-B）
#     评估：每 eval_every 轮在测试集评估 S 的 top-1 accuracy
#     保存：每 save_ckpt_every 轮保存 checkpoint
# ============================================================

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from data.data_utils import prepare_data
from models.generator import Generator
from models.task_model import CentralModel
from models.discriminator import Discriminator
from clustering.sinkhorn import compute_sinkhorn_matrix
from clustering.clustering import (
    cluster_clients,
    get_cluster_sizes,
    pack_cluster_result,
    unpack_cluster_result,
)
from client.client import Client
from aggregation.aggregation import hierarchical_aggregate
from trainer.trainer import update_generator, update_central_model, build_optimizers
from utils.logger import Logger
from utils.metrics import MetricsTracker


class FedDistillServer:
    """
    联邦蒸馏服务器，统筹协调完整训练流程。

    职责：
      - 持有生成器 G 和中心模型 S
      - 管理所有 Client 对象（含 T_k 和 D_k）
      - 执行聚类（仅一次）、每轮数据生成、并行推理、聚合、更新
      - 管理 checkpoint 保存与恢复
    """

    def __init__(self, cfg):
        """
        初始化服务器：加载数据、模型、执行聚类。

        Args:
            cfg: OmegaConf 配置对象
        """
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[server] 运行设备: {self.device}")

        # ---- 工具模块 ----
        self.logger = Logger(log_dir=cfg.log_dir)
        self.metrics = MetricsTracker()

        # ---- 当前轮次（恢复训练时从 checkpoint 更新）----
        self.current_round = 0

        # ---- Step 1：数据准备 ----
        self._prepare_data()

        # ---- Step 2：初始化客户端（含加载 T_k）----
        self._init_clients()

        # ---- Step 3：聚类（仅执行一次）----
        self._run_clustering()

        # ---- Step 4：初始化服务器端模型与优化器 ----
        self._init_server_models()

    # ================================================================== #
    #  初始化子流程
    # ================================================================== #

    def _prepare_data(self):
        """加载数据集划分，构建 DataLoader，计算 p_k 和 π_k。"""
        print("[server] 准备数据集...")
        (
            self.client_loaders,
            self.test_loader,
            self.p_k_list,
            self.pi_k_list,
        ) = prepare_data(self.cfg)

        # π_k 字典：{client_id: π_k}，供 trainer 使用
        self.pi_k_dict: Dict[int, float] = {
            k: self.pi_k_list[k] for k in range(self.cfg.num_clients)
        }

    def _init_clients(self):
        """初始化所有客户端，加载教师模型 T_k 和判别器 D_k。"""
        print("[server] 初始化客户端...")
        self.clients: List[Client] = []
        for k in range(self.cfg.num_clients):
            client = Client(
                client_id=k,
                pi_k=self.pi_k_list[k],
                data_loader=self.client_loaders[k],
                cfg=self.cfg,
                device=self.device,
            )
            self.clients.append(client)
        print(f"[server] 已初始化 {len(self.clients)} 个客户端")

    def _run_clustering(self):
        """
        计算 Sinkhorn 距离矩阵 → K-means 聚类 → 选取簇头。
        聚类结果固定，整个训练过程不再重新聚类。
        """
        print("[server] 计算 Sinkhorn 距离矩阵并聚类...")

        # 计算 K×K Sinkhorn 距离矩阵
        dist_matrix = compute_sinkhorn_matrix(
            p_k_list=self.p_k_list,
            eps=self.cfg.sinkhorn_eps,
            num_iters=self.cfg.sinkhorn_iters,
            device=self.device,
        )

        # K-means 聚类 + 簇头选取
        self.cluster_assignments, self.cluster_heads = cluster_clients(
            dist_matrix=dist_matrix,
            num_clusters=self.cfg.num_clusters,
            seed=self.cfg.seed,
        )

        # 各簇客户端数量 N_c
        self.cluster_sizes = get_cluster_sizes(
            self.cluster_assignments, self.cfg.num_clusters
        )

        print(f"[server] 聚类完成：{self.cfg.num_clusters} 个簇，"
              f"簇头={self.cluster_heads}")

    def _init_server_models(self):
        """初始化生成器 G、中心模型 S 及其优化器。"""
        print("[server] 初始化服务器端模型...")

        # 生成器 G
        self.generator = Generator(
            latent_dim=self.cfg.latent_dim,
            image_channels=self.cfg.image_channels,
        ).to(self.device)

        # 中心模型 S（ResNet-18，与 Large 同构）
        self.central_model = CentralModel(
            num_classes=self.cfg.num_classes,
            image_channels=self.cfg.image_channels,
        ).to(self.device)

        # 优化器
        self.optimizer_g, self.optimizer_s = build_optimizers(
            generator=self.generator,
            central_model=self.central_model,
            cfg=self.cfg,
        )

        # 创建 checkpoint 目录
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

    # ================================================================== #
    #  阶段 1：生成伪数据
    # ================================================================== #

    def _generate_fake_data(self) -> torch.Tensor:
        """
        阶段 1：G 从 ω ~ N(0,I) 采样，生成 batch 伪数据。

        Returns:
            x_fake: shape=(gen_batch_size, C, H, W)，值域[-1,1]
        """
        self.generator.eval()
        with torch.no_grad():
            z_noise = torch.randn(
                self.cfg.gen_batch_size, self.cfg.latent_dim, device=self.device
            )
            x_fake = self.generator(z_noise)
        return x_fake

    # ================================================================== #
    #  阶段 2-A：联合更新所有判别器（对齐 FedIOD update_netDS_batch）
    # ================================================================== #

    def _train_discriminators_jointly(
        self,
        x_fake: torch.Tensor,
        num_steps: int,
    ) -> float:
        """
        联合更新所有客户端判别器，对齐 FedIOD 的 update_netDS_batch。

        FedIOD：所有判别器损失累加平均 → 一次 backward → 各 D_k 用同一
                optim_d step。
        本实现：各 D_k 独立 optimizer_d，联合 backward 后各自 step，
                梯度来源完全相同，效果等价。

        Args:
            x_fake:    阶段1生成的伪数据，shape=(B, C, H, W)
            num_steps: 判别器更新步数（config: local_gan_steps）

        Returns:
            avg_d_loss: 所有客户端平均判别器损失
        """
        x_fake_detach = x_fake.detach()
        total_loss_val = 0.0

        for _ in range(num_steps):
            # 累加所有客户端判别器损失
            joint_loss = torch.tensor(0.0, device=self.device)
            cnt = len(self.clients)
            for client in self.clients:
                loss_d = client.compute_discriminator_loss(x_fake_detach)
                joint_loss = joint_loss + loss_d
            joint_loss = joint_loss / cnt   # 平均，对齐 FedIOD /cnt_disc

            # 联合 backward
            for client in self.clients:
                client.optimizer_d.zero_grad()
            joint_loss.backward()

            # 各 D_k 独立 clip + step
            for client in self.clients:
                nn.utils.clip_grad_norm_(
                    client.discriminator.parameters(), max_norm=1.0
                )
                client.optimizer_d.step()
                client.discriminator.eval()

            total_loss_val += joint_loss.item()

        return total_loss_val / num_steps

    # ================================================================== #
    #  阶段 2-B：客户端 T_k 推理
    # ================================================================== #

    def _parallel_local_compute(
        self, x_fake: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        阶段 2-B：串行执行所有客户端 T_k 推理，收集 logits 和软概率。
        判别器更新（Step 2-A）已由 _train_discriminators_jointly 单独处理。

        Args:
            x_fake: 生成数据，shape=(B, C, H, W)

        Returns:
            logits_dict:  {client_id: z_k}
            softmax_dict: {client_id: q_k}
        """
        logits_dict:  Dict[int, torch.Tensor] = {}
        softmax_dict: Dict[int, torch.Tensor] = {}

        for client in self.clients:
            try:
                z_k, q_k = client.infer(
                    x=x_fake,
                    temperature=self.cfg.temperature,
                )
                logits_dict[client.client_id]  = z_k.detach()
                softmax_dict[client.client_id] = q_k.detach()
            except Exception as e:
                import traceback
                print(f"[server] 警告：客户端 {client.client_id} 推理失败:")
                traceback.print_exc()

        return logits_dict, softmax_dict

    # ================================================================== #
    #  阶段 3：分层聚合
    # ================================================================== #

    def _aggregate(
        self, logits_dict: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        阶段 3：执行两阶段分层聚合，生成全局软标签 Z。

        Args:
            logits_dict: {client_id: z_k}

        Returns:
            Z:    全局软标签 logits，shape=(B, num_classes)
            info: 聚合权重信息（用于日志）
        """
        Z, info = hierarchical_aggregate(
            cluster_assignments=self.cluster_assignments,
            logits_dict=logits_dict,
            cluster_sizes=self.cluster_sizes,
            num_clusters=self.cfg.num_clusters,
            temperature=self.cfg.temperature,
            gamma=self.cfg.gamma,
            agg_temperature=self.cfg.agg_temperature,
        )
        return Z, info

    # ================================================================== #
    #  评估：测试集 top-1 accuracy
    # ================================================================== #

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        在全局测试集上评估中心模型 S 的 top-1 accuracy。
        每 eval_every 轮调用一次。

        Returns:
            accuracy: float，[0, 1]
        """
        self.central_model.eval()
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.central_model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    # ================================================================== #
    #  Checkpoint 保存与恢复
    # ================================================================== #

    def save_checkpoint(self, round_idx: int, accuracy: float):
        """
        保存当前训练状态至 checkpoint 文件。

        保存内容（对应设计文档第九节）：
          round, accuracy, generator, central_model,
          discriminators, optim_g, optim_s, optim_d, cluster_result

        Args:
            round_idx: 当前轮次
            accuracy:  当前测试集 accuracy
        """
        state = {
            'round':         round_idx,
            'accuracy':      accuracy,
            'generator':     self.generator.state_dict(),
            'central_model': self.central_model.state_dict(),
            'discriminators': {
                k: self.clients[k].discriminator.state_dict()
                for k in range(self.cfg.num_clients)
            },
            'optim_g': self.optimizer_g.state_dict(),
            'optim_s': self.optimizer_s.state_dict(),
            'optim_d': {
                k: self.clients[k].optimizer_d.state_dict()
                for k in range(self.cfg.num_clients)
            },
            'cluster_result': pack_cluster_result(
                self.cluster_assignments, self.cluster_heads
            ),
        }

        # 保存 round_XXXX.pt
        round_path = os.path.join(
            self.cfg.ckpt_dir, f"round_{round_idx:04d}.pt"
        )
        torch.save(state, round_path)

        # 更新 latest.pt
        latest_path = os.path.join(self.cfg.ckpt_dir, "latest.pt")
        torch.save(state, latest_path)

        print(f"[server] Checkpoint 已保存: {round_path} | accuracy={accuracy:.4f}")

    def load_checkpoint(self, ckpt_path: str):
        """
        从指定 checkpoint 文件恢复训练状态。
        对应 main.py 的 --resume 参数。

        Args:
            ckpt_path: checkpoint 文件路径
        """
        print(f"[server] 从 checkpoint 恢复: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)

        # 恢复轮次（下一轮从 current_round + 1 开始）
        self.current_round = state['round']

        # 恢复模型参数
        self.generator.load_state_dict(state['generator'])
        self.central_model.load_state_dict(state['central_model'])

        # 恢复判别器参数
        for k in range(self.cfg.num_clients):
            if k in state['discriminators']:
                self.clients[k].discriminator.load_state_dict(
                    state['discriminators'][k]
                )

        # 恢复优化器状态
        self.optimizer_g.load_state_dict(state['optim_g'])
        self.optimizer_s.load_state_dict(state['optim_s'])
        for k in range(self.cfg.num_clients):
            if k in state['optim_d']:
                self.clients[k].optimizer_d.load_state_dict(
                    state['optim_d'][k]
                )

        # 恢复聚类结果（覆盖重新聚类的结果）
        if 'cluster_result' in state:
            self.cluster_assignments, self.cluster_heads = unpack_cluster_result(
                state['cluster_result']
            )
            self.cluster_sizes = get_cluster_sizes(
                self.cluster_assignments, self.cfg.num_clusters
            )
            print(f"[server] 聚类结果已从 checkpoint 恢复")

        print(f"[server] 从第 {self.current_round + 1} 轮继续训练")

    # ================================================================== #
    #  主训练循环
    # ================================================================== #

    def train(self, resume_path: Optional[str] = None):
        """
        执行完整的联邦蒸馏训练主循环。

        Args:
            resume_path: 若指定，则从该 checkpoint 恢复后继续训练
        """
        # 恢复训练
        if resume_path is not None:
            self.load_checkpoint(resume_path)

        start_round = self.current_round + 1
        total_rounds = self.cfg.fed_rounds

        print(f"\n[server] 开始联邦蒸馏训练：第 {start_round} 轮 → 第 {total_rounds} 轮")
        print("=" * 60)

        # 初始 Z（第一轮聚合前用零张量占位）
        Z = torch.zeros(
            self.cfg.gen_batch_size, self.cfg.num_classes, device=self.device
        )

        for round_idx in range(start_round, total_rounds + 1):
            self.current_round = round_idx
            self.logger.start_round(round_idx)

            # ---------------------------------------------------------- #
            #  阶段 1：生成伪数据
            # ---------------------------------------------------------- #
            x_fake = self._generate_fake_data()

            # ---------------------------------------------------------- #
            #  阶段 2-A：联合更新所有判别器（对齐 FedIOD update_netDS_batch）
            # ---------------------------------------------------------- #
            avg_d_loss = self._train_discriminators_jointly(
                x_fake=x_fake,
                num_steps=self.cfg.local_gan_steps,
            )

            # ---------------------------------------------------------- #
            #  阶段 2-B：T_k 推理，收集各客户端 logits
            # ---------------------------------------------------------- #
            logits_dict, softmax_dict = self._parallel_local_compute(x_fake)

            # ---------------------------------------------------------- #
            #  阶段 3：分层聚合 → 全局软标签 Z
            # ---------------------------------------------------------- #
            if len(logits_dict) > 0:
                Z, agg_info = self._aggregate(logits_dict)
            else:
                print(f"[server] 警告：第 {round_idx} 轮无有效客户端 logits，跳过聚合")
                agg_info = {}

            # ---------------------------------------------------------- #
            #  阶段 4-A：更新生成器 G
            # ---------------------------------------------------------- #
            g_loss_info = update_generator(
                generator=self.generator,
                optimizer_g=self.optimizer_g,
                clients=self.clients,
                central_model=self.central_model,
                Z=Z,
                pi_k_list=self.pi_k_list,
                pi_k_dict=self.pi_k_dict,
                cfg=self.cfg,
                device=self.device,
                num_steps=self.cfg.generator_steps,
            )

            # ---------------------------------------------------------- #
            #  阶段 4-B：更新中心模型 S
            # ---------------------------------------------------------- #
            # 注意：传入阶段1生成的同一批 x_fake，与 Z 严格对应
            # 保证"S(x_fake) 对标 Z=Aggregate({T_k(x_fake)})"的标签-输入一致性
            loss_s = update_central_model(
                central_model=self.central_model,
                optimizer_s=self.optimizer_s,
                x_fake=x_fake,        # 阶段1生成、阶段2/3使用的同一批数据
                Z=Z,
                cfg=self.cfg,
                num_steps=self.cfg.central_steps,
            )

            # ---------------------------------------------------------- #
            #  评估：每 eval_every 轮测试 S 的 top-1 accuracy
            # ---------------------------------------------------------- #
            accuracy = 0.0
            if round_idx % self.cfg.eval_every == 0:
                accuracy = self.evaluate()
                self.metrics.record_accuracy(round_idx, accuracy)
                print(
                    f"[server] Round {round_idx:4d}/{total_rounds} | "
                    f"Acc={accuracy:.4f} | "
                    f"L_G={g_loss_info['loss_g']:.4f} | "
                    f"L_conf={g_loss_info['loss_conf']:.4f} | "
                    f"L_bal={g_loss_info['loss_balance']:.4f} | "
                    f"L_div={g_loss_info['loss_div']:.4f} | "
                    f"L_adv={g_loss_info['loss_adv']:.4f} | "
                    f"L_S={loss_s:.4f} | "
                    f"L_D={avg_d_loss:.4f}"
                )
            else:
                print(
                    f"[server] Round {round_idx:4d}/{total_rounds} | "
                    f"L_G={g_loss_info['loss_g']:.4f} | "
                    f"L_S={loss_s:.4f} | "
                    f"L_D={avg_d_loss:.4f}"
                )

            # ---------------------------------------------------------- #
            #  记录日志
            # ---------------------------------------------------------- #
            self.logger.log_round(
                round_idx=round_idx,
                loss_g=g_loss_info['loss_g'],
                loss_conf=g_loss_info['loss_conf'],
                loss_div=g_loss_info['loss_div'],
                loss_adv=g_loss_info['loss_adv'],
                loss_balance=g_loss_info['loss_balance'],
                loss_s=loss_s,
                loss_d=avg_d_loss,
                accuracy=accuracy,
                agg_info=agg_info,
            )
            self.metrics.record_losses(
                round_idx=round_idx,
                loss_g=g_loss_info['loss_g'],
                loss_s=loss_s,
                loss_d=avg_d_loss,
            )

            # ---------------------------------------------------------- #
            #  保存 Checkpoint
            # ---------------------------------------------------------- #
            if round_idx % self.cfg.save_ckpt_every == 0:
                self.save_checkpoint(round_idx, accuracy)

        # 训练结束，保存最终 checkpoint
        final_accuracy = self.evaluate()
        self.save_checkpoint(total_rounds, final_accuracy)
        self.metrics.save(self.cfg.log_dir)

        print("=" * 60)
        print(f"[server] 训练完成！最终测试集 accuracy = {final_accuracy:.4f}")
        print(f"[server] 日志已保存至: {self.cfg.log_dir}")