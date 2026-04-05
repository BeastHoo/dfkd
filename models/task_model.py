# models/task_model.py
# 功能：定义异构任务模型 T_k（Small/Medium/Large）及中心模型 S（CentralModel）
# 客户端模型从外部 .pt 文件加载，加载后全程 eval() + requires_grad_(False)
# 中心模型 S 与 Large（ResNet-18）结构相同，参数正常训练
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# ================================================================== #
#  Small：2层 Conv + 2层 FC
# ================================================================== #

class SmallModel(nn.Module):
    """
    小型任务模型（Small）。
    结构：Conv×2 + FC×2
    适用于资源受限客户端。
    """

    def __init__(self, num_classes: int, image_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B, C, 32, 32) → (B, 32, 15, 15)
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32→16

            # Block 2: (B, 32, 16, 16) → (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ================================================================== #
#  Medium：3层 Conv块（含 BN）+ 2层 FC
# ================================================================== #

class MediumModel(nn.Module):
    """
    中型任务模型（Medium）。
    结构：Conv块×3（含 BN + MaxPool）+ FC×2
    """

    def __init__(self, num_classes: int, image_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: → (B, 64, 16, 16)
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16

            # Block 2: → (B, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8

            # Block 3: → (B, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ================================================================== #
#  Large：标准 ResNet-18（输出层适配 num_classes）
# ================================================================== #

class BasicBlock(nn.Module):
    """ResNet-18 基本残差块。"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样快捷连接（当 stride≠1 或通道数变化时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class LargeModel(nn.Module):
    """
    大型任务模型（Large）：标准 ResNet-18，适配 CIFAR 32×32 输入。

    与 torchvision ResNet-18 的区别：
      - 第一层 Conv 改为 3×3，stride=1，padding=1（原为 7×7 stride=2）
      - 去掉第一个 MaxPool（原版针对 224×224，CIFAR 不需要）
    这是 CIFAR ResNet 的标准改动，保持特征图不过早缩小。
    """

    def __init__(self, num_classes: int, image_channels: int = 3):
        super().__init__()

        # 初始卷积（CIFAR 适配版）
        self.stem = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 4 个残差层，通道数：64 → 128 → 256 → 512
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ================================================================== #
#  CentralModel：与 Large 相同结构，参数正常训练
# ================================================================== #

class CentralModel(LargeModel):
    """
    中心任务模型 S，结构与 LargeModel（ResNet-18）完全相同。
    参数全程参与梯度更新，不冻结。
    用于接收全局软标签 Z 进行蒸馏训练（Step 4-B，公式 L_S）。
    """

    def __init__(self, num_classes: int, image_channels: int = 3):
        super().__init__(num_classes=num_classes, image_channels=image_channels)


# ================================================================== #
#  模型工厂：根据类型字符串创建模型实例
# ================================================================== #

ModelType = Literal["small", "medium", "large"]


def build_task_model(
        model_type: ModelType,
        num_classes: int,
        image_channels: int = 3,
) -> nn.Module:
    """
    根据类型字符串创建任务模型实例（不加载权重）。

    Args:
        model_type:     模型类型，'small' | 'medium' | 'large'
        num_classes:    输出类别数
        image_channels: 输入图像通道数

    Returns:
        nn.Module 实例
    """
    model_map = {
        "small": SmallModel,
        "medium": MediumModel,
        "large": LargeModel,
    }
    if model_type not in model_map:
        raise ValueError(f"不支持的模型类型: {model_type}，请使用 small/medium/large")
    return model_map[model_type](num_classes=num_classes, image_channels=image_channels)


# ================================================================== #
#  教师模型加载：从 .pt 文件加载，全程冻结
# ================================================================== #

def load_teacher_model(
        client_id: int,
        teacher_ckpt_dir: str,
        num_classes: int,
        image_channels: int,
        device: torch.device,
) -> nn.Module:
    """
    从指定目录加载客户端 k 的教师模型 T_k。

    文件命名规则：client_{k}.pt（k 从 0 开始）
    .pt 文件格式：
      - 方式 A（推荐）：{'model_type': 'small'|'medium'|'large', 'state_dict': OrderedDict}
      - 方式 B（兼容）：完整模型对象（torch.save(model, path)）

    加载后：
      - model.eval()               → 关闭 BN/Dropout 的训练模式
      - model.requires_grad_(False) → 冻结所有参数，全程不参与梯度更新

    Args:
        client_id:        客户端编号 k（从 0 开始）
        teacher_ckpt_dir: 教师模型目录路径
        num_classes:      类别数
        image_channels:   图像通道数
        device:           目标设备

    Returns:
        冻结后的教师模型 T_k（已移至 device）

    Raises:
        FileNotFoundError: 若对应 .pt 文件不存在，抛出明确错误（不静默跳过）
    """
    ckpt_path = os.path.join(teacher_ckpt_dir, f"client_{client_id}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[task_model] 客户端 {client_id} 的教师模型文件不存在: {ckpt_path}\n"
            f"请将预训练模型保存为 {ckpt_path}，"
            f"格式为 dict: {{'model_type': 'small'|'medium'|'large', 'state_dict': ...}}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- 方式 A：dict 格式（推荐）----
    if isinstance(ckpt, dict) and 'model_type' in ckpt and 'state_dict' in ckpt:
        model_type = ckpt['model_type']
        model = build_task_model(model_type, num_classes, image_channels)
        model.load_state_dict(ckpt['state_dict'])
        print(f"[task_model] Client {client_id}: 加载 {model_type} 模型 ← {ckpt_path}")

    # ---- 方式 B：完整模型对象（兼容旧格式）----
    elif isinstance(ckpt, nn.Module):
        model = ckpt
        print(f"[task_model] Client {client_id}: 加载完整模型对象 ← {ckpt_path}")

    # ---- 方式 C：仅 state_dict，默认 large ----
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model = build_task_model("large", num_classes, image_channels)
        model.load_state_dict(ckpt['state_dict'])
        print(f"[task_model] Client {client_id}: 加载 large 模型（无 model_type 字段） ← {ckpt_path}")

    else:
        raise ValueError(
            f"[task_model] 无法识别 .pt 文件格式: {ckpt_path}\n"
            f"支持格式：\n"
            f"  A. dict with keys 'model_type' + 'state_dict'\n"
            f"  B. 完整 nn.Module 对象\n"
            f"  C. dict with key 'state_dict'（默认 large 结构）"
        )

    # 移至目标设备，冻结参数，切换推理模式
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    return model


def save_teacher_model(
        model: nn.Module,
        model_type: ModelType,
        save_path: str,
) -> None:
    """
    将教师模型以推荐格式（方式 A）保存为 .pt 文件。
    供预训练脚本使用，保证 load_teacher_model 可正确加载。

    Args:
        model:      已训练好的模型实例
        model_type: 模型类型字符串
        save_path:  保存路径（含文件名）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {'model_type': model_type, 'state_dict': model.state_dict()},
        save_path,
    )
    print(f"[task_model] 教师模型已保存: {save_path}")
