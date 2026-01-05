"""
简化配置文件（仅保留本项目训练所需的核心参数）
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    # ===== 数据 =====
    data_root: str = "data_root"
    train_dir: str = "train"
    val_dir: str = "val"

    # 预处理/缓存（提升 I/O）
    use_cached_dataset: bool = True           # 启用预处理后的随机访问格式
    cache_dir: str = "cache_masked"           # 预处理产物目录（data_root 下）
    cache_format: str = "npy"                 # 目前实现 npy
    prefer_uncompressed_nii: bool = True      # 仍读取 NIfTI 时优先 .nii

    # 文件命名（原始 NIfTI 流程）
    te_files: Optional[List[str]] = None
    mask_file: str = "mask.nii.gz"

    # ===== 协议 =====
    channels_per_te: int = 126   # 每个 TE 的通道数
    te_count: int = 5
    input_channels: int = 630    # 5 * 126
    directions: int = 30

    # 与原项目一致：不在前向模型中显式使用 T0/B0 做无量纲化（仅保留以兼容旧接口），
    # 实际对 t、b 的无量纲化通过常数 T_SCALE/B_SCALE 在 dataset/physics 流程中完成
    T0: Optional[float] = None
    B0: Optional[float] = None

    # TE 列表（毫秒）
    tes: Optional[List[float]] = None

    # ===== 归一化 =====
    t_ref_idx: int = 0           # 参考 TE 索引（默认最短 TE）
    use_s0_normalization: bool = False      # 原始域训练，保留选项作兼容

    # ===== 模型 =====
    in_channels: int = 630
    mid_channels: int = 64
    out_global: int = 4          # ln_s0, r, sigma20, sigma30
    out_per_dir: int = 6         # D, sigma11, sigma02, sigma21, sigma12, sigma03
    out_channels: int = 184      # 4 + 30*6

    # ===== 损失 =====
    exclude_ref_b0_in_loss: bool = False    # 原始域默认不排除参考TE的b0
    # Huber（clipped L1）配置
    use_huber: bool = False
    huber_delta: float = 0.3

    # ===== 训练 =====
    device: str = "cuda:0"       # 已弃用：实际使用 gpu_id
    gpu_id: Optional[int] = 0
    batch_size: int = 4
    num_workers: int = 1
    prefetch_factor: int = 2
    use_amp: bool = True
    epochs: int = 200
    stage1_epochs: int = 50      # Stage1 最多训练轮数（b0 阶段的上限）
    stage1_min_epochs: int = 10  # Stage1 至少训练轮数，避免过早切换
    stage1_patience: int = 5     # 在验证集上无明显提升的耐心轮数
    stage1_improve_tol: float = 1e-4  # 验证 L1_numden 的提升阈值
    stage2_min_epochs: int = 10  # Stage2(DW-only) 至少训练轮数
    stage2_patience: int = 5     # Stage2 在验证集上无明显提升的耐心轮数
    stage2_improve_tol: float = 1e-4  # Stage2 验证 L1_numden 的提升阈值
    stage1_global_warm_epochs: int = 10  # Stage1 前若干轮使用较大学习率
    stage1_global_warm_lr: float = 1e-4  # Stage1 global_head 预热学习率
    stage2_dir_lr: float = 2e-4          # Stage2 中 dir_head 的学习率（原来 5e-5，现翻倍）
    use_two_stage_b0_dw: bool = True
    first_order_only: bool = False
    freeze_global_after_stage1: bool = True
    b0_threshold: float = 1e-6
    # 更小的基础学习率，提升初期稳定性
    lr: float = 5e-5
    # r 以及高阶系数的更小学习率（通过参数组设置）
    lr_r: float = 2e-5
    lr_high_order: float = 5e-5
    weight_decay: float = 1e-4
    dir_weight_decay: float = 0.0

    # 梯度
    accum_steps: int = 1
    # 更严格的梯度裁剪
    clip_grad_norm: float = 0.3

    # 日志
    log_interval: int = 20
    save_dir: str = "checkpoints"
    # 忽略错误观测开关（训练时）
    ignore_invalid_signal: bool = False
    signal_min_valid: float = 0.0

    # ===== LS warm-start =====
    use_ls_init: bool = False            # use LS parameter maps to warm-start training
    ls_init_dir: str = "ls_init"         # directory under data_root containing LS maps
    lambda_param_init: float = 0.1       # initial weight for param alignment loss
    param_warmup_epochs: int = 5         # epochs to keep constant weight
    param_decay_epochs: int = 15         # epochs to linearly decay weight to 0

    # ===== Resume training =====
    resume_from_best: bool = True       # 是否从 best_model_A.pt 继续训练
    resume_ckpt_name: str = "best_model_A.pt"  # 续训用的 checkpoint 文件名

    def __post_init__(self):
        if self.te_files is None:
            self.te_files = [f"te{i}.nii.gz" for i in range(1, 6)]
        if self.tes is None:
            self.tes = [22, 37, 52, 67, 82]


# 默认配置
cfg = Config()
