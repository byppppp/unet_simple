"""
Curriculum + Correct Logging Training (Scheme A + E)

- Scheme A: curriculum on physics orders
  * Early epochs: only 0/1-order (ln_s0, r, D)
  * Middle: enable 2nd-order with small scale
  * Later: enable 3rd-order with small scale (then full)

- Scheme E: epoch logging prints both
  * mean of batch L1
  * and aggregated L1 = (sum num) / (sum den)

Run: python rdmri_unet_simple/train_A.py
"""

from pathlib import Path
from typing import Tuple
import math
import time
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import cfg
from models import UNetWithHeads
from physics import TaylorForward
from loss import L1PhysicsLoss
from dataset import CachedMRIDataset


# -------- Curriculum schedule (can adjust here) --------
# Stage boundaries (inclusive):
STAGE1_EPOCHS = 5   # 1..5: only 0/1-order
STAGE2_EPOCHS = 15  # 6..15: enable 2nd-order with small scale
STAGE3_EPOCHS = 25  # 16..25: enable 3rd-order with small scale

# Scales applied in each stage
SCALE2_STAGE2 = 0.1
SCALE3_STAGE3 = 0.1

T_SCALE = 100.0
B_SCALE = 2.5


def select_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，使用 CPU")
        return torch.device("cpu")
    gid = getattr(cfg, "gpu_id", 0) or 0
    gid = min(gid, torch.cuda.device_count() - 1)
    torch.cuda.set_device(gid)
    print(f"✓ 使用 GPU {gid}: {torch.cuda.get_device_name(gid)}")
    return torch.device(f"cuda:{gid}")


def curriculum_scale(epoch: int) -> Tuple[float, float]:
    """Return (scale2, scale3) according to epoch."""
    # 2nd-order scale: 0 -> 1 linearly between (STAGE1_EPOCHS, STAGE3_EPOCHS]
    if epoch <= STAGE1_EPOCHS:
        scale2 = 0.0
    elif epoch <= STAGE3_EPOCHS:
        span2 = max(STAGE3_EPOCHS - STAGE1_EPOCHS, 1)
        frac2 = (epoch - STAGE1_EPOCHS) / span2
        scale2 = float(min(max(frac2, 0.0), 1.0))
    else:
        scale2 = 1.0

    # 3rd-order scale: 0 -> 1 linearly between (STAGE2_EPOCHS, 40]
    s3_full_epoch = 40
    if epoch <= STAGE2_EPOCHS:
        scale3 = 0.0
    elif epoch <= s3_full_epoch:
        span3 = max(s3_full_epoch - STAGE2_EPOCHS, 1)
        frac3 = (epoch - STAGE2_EPOCHS) / span3
        scale3 = float(min(max(frac3, 0.0), 1.0))
    else:
        scale3 = 1.0

    return scale2, scale3


def apply_curriculum(theta_global: torch.Tensor, theta_dir: torch.Tensor, epoch: int):
    """Return copies of theta with curriculum scaling applied.

    theta_global: [B, 4, H, W] -> (ln_s0, r, sigma20, sigma30)
    theta_dir:    [B, 30, 6, H, W] -> (D, sigma11, sigma02, sigma21, sigma12, sigma03)
    """
    s2, s3 = curriculum_scale(epoch)
    # clone to avoid in-place affecting backprop graph multiple uses
    tg = theta_global.clone()
    td = theta_dir.clone()
    # Global: 2nd, 3rd
    tg[:, 2:3, :, :] = tg[:, 2:3, :, :] * 1.0
    tg[:, 3:4, :, :] = tg[:, 3:4, :, :] * 1.0
    # Directional: 2nd (sigma11, sigma02) and 3rd (sigma21, sigma12, sigma03)
    td[:, :, 1, :, :] = td[:, :, 1, :, :] * s2
    td[:, :, 2, :, :] = td[:, :, 2, :, :] * s2
    td[:, :, 3, :, :] = td[:, :, 3, :, :] * s3
    td[:, :, 4, :, :] = td[:, :, 4, :, :] * s3
    td[:, :, 5, :, :] = td[:, :, 5, :, :] * s3

    # Optional: force first-order-only model (ln_s0, r, D) by zeroing higher-order sigmas
    if getattr(cfg, 'first_order_only', False):
        tg[:, 2:4, :, :] = 0.0
        td[:, :, 1:, :, :] = 0.0

    return tg, td, s2, s3


def masked_mean_l1(S_hat: torch.Tensor, signal: torch.Tensor, mask: torch.Tensor, channel_mask: torch.Tensor = None):
    S_scale = 1000.0
    l1_map = torch.abs(S_hat - signal) / S_scale
    if channel_mask is not None:
        l1_map = l1_map[:, channel_mask, :, :]
    if mask is None:
        num = l1_map.sum()
        den = torch.tensor(l1_map.numel(), dtype=l1_map.dtype, device=l1_map.device)
        return num / den, num, den
    if mask.shape[1] == 1:
        mask_exp = mask.expand(-1, l1_map.shape[1], -1, -1)
    else:
        mask_exp = mask
    num = (l1_map * mask_exp).sum()
    den = mask_exp.sum().clamp_min(1e-8)
    return num / den, num, den


def normalize_signal_for_unet(signal: torch.Tensor) -> torch.Tensor:
    """对输入 UNet 的信号做 log+线性缩放归一化，范围约为 [0, 1]。

    仅用于网络输入；物理损失仍然使用原始信号。
    """
    # 避免 log(0)
    x = signal.clamp(min=1.0)
    x = torch.log(x)

    # 依据经验信号范围 [100, 20000] 设定 log 区间
    log_min = math.log(100.0)
    log_max = math.log(20000.0)
    scale = (log_max - log_min)
    x = (x - log_min) / scale
    return x.clamp(0.0, 1.0)


def _tensor_stats(x: torch.Tensor):
    return {
        'mean': float(x.mean().item()),
        'std': float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }


def _masked_stats_4d(x: torch.Tensor, mask: torch.Tensor):
    if mask is not None and mask.ndim == 4:
        if x.ndim == 4 and mask.shape[1] == 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        vals = x[mask > 0.5]
    else:
        vals = x.reshape(-1)
    if vals.numel() == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return _tensor_stats(vals)


@torch.no_grad()
def _physics_breakdown(theta_global: torch.Tensor, theta_dir: torch.Tensor,
                       t_vec: torch.Tensor, b_vec: torch.Tensor, P: torch.Tensor,
                       mask: torch.Tensor = None):
    C = 630
    # 确保协议向量和 P 与模型参数处于相同 dtype/device，避免 AMP 下的 Half/Float 冲突
    dtype = theta_global.dtype
    device = theta_global.device

    ln_s0 = theta_global[:, 0:1]
    r = theta_global[:, 1:2]
    sigma20 = theta_global[:, 2:3]
    sigma30 = theta_global[:, 3:4]
    D = theta_dir[:, :, 0, :, :]
    sigma11 = theta_dir[:, :, 1, :, :]
    sigma02 = theta_dir[:, :, 2, :, :]
    sigma21 = theta_dir[:, :, 3, :, :]
    sigma12 = theta_dir[:, :, 4, :, :]
    sigma03 = theta_dir[:, :, 5, :, :]

    t = t_vec.to(device=device, dtype=dtype).view(1, -1, 1, 1)
    b = b_vec.to(device=device, dtype=dtype).view(1, -1, 1, 1)
    P_cast = P.to(device=device, dtype=dtype)

    D_ch = torch.einsum('cd,bdhw->bchw', P_cast, D)
    sigma11_ch = torch.einsum('cd,bdhw->bchw', P_cast, sigma11)
    sigma02_ch = torch.einsum('cd,bdhw->bchw', P_cast, sigma02)
    sigma21_ch = torch.einsum('cd,bdhw->bchw', P_cast, sigma21)
    sigma12_ch = torch.einsum('cd,bdhw->bchw', P_cast, sigma12)
    sigma03_ch = torch.einsum('cd,bdhw->bchw', P_cast, sigma03)
    term0 = ln_s0.expand(-1, C, -1, -1)
    term_r = -r.expand(-1, C, -1, -1) * t
    term_D = -D_ch * b
    t2 = t ** 2; b2 = b ** 2; tb = t * b
    term2 = 0.5 * (sigma20.expand(-1, C, -1, -1) * t2 + 2.0 * sigma11_ch * tb + sigma02_ch * b2)
    t3 = t ** 3; b3 = b ** 3; t2b = t2 * b; tb2 = t * b2
    term3 = -(1.0/6.0) * (sigma30.expand(-1, C, -1, -1) * t3 + 3.0 * sigma21_ch * t2b + 3.0 * sigma12_ch * tb2 + sigma03_ch * b3)
    log_raw = term0 + term_r + term_D + term2 + term3
    if mask is not None and mask.ndim == 4:
        mexp = mask.expand(-1, C, -1, -1) if mask.shape[1] == 1 else mask
        sel = mexp > 0.5
    else:
        sel = torch.ones_like(log_raw, dtype=torch.bool)
    total = float(sel.sum().item()) if sel.numel() > 0 else 1.0
    clamp_hits = ((log_raw <= -15.0) | (log_raw >= 15.0)) & sel
    return {'clamp_rate': float(clamp_hits.sum().item()) / total}


def _param_weight(epoch: int, stage: int, stage2_epoch: int = 0) -> float:
    """Return LS alignment loss weight for given epoch and stage.

    Stage1: use global epoch schedule as before.
    Stage2: restart a similar schedule based on stage2_epoch so that
    LS supervision is also active when entering Stage2.
    """
    w0 = getattr(cfg, 'lambda_param_init', 0.0)
    warm = getattr(cfg, 'param_warmup_epochs', 0)
    decay = getattr(cfg, 'param_decay_epochs', 0)
    if w0 <= 0 or (warm <= 0 and decay <= 0):
        return float(w0)

    if stage == 1:
        # Original behaviour: schedule based on global epoch
        if epoch <= warm:
            return float(w0)
        if decay <= 0:
            return 0.0
        if epoch <= warm + decay:
            frac = (epoch - warm) / max(decay, 1)
            return float(w0) * max(0.0, 1.0 - frac)
        return 0.0

    # Stage2: schedule based on stage2_epoch (1-based inside Stage2)
    e2 = max(int(stage2_epoch), 1)
    if e2 <= warm:
        return float(w0)
    if decay <= 0:
        return 0.0
    if e2 <= warm + decay:
        frac = (e2 - warm) / max(decay, 1)
        return float(w0) * max(0.0, 1.0 - frac)
    return 0.0


def _build_ls_targets(batch, ls_root: Path, device):
    # Prefer cached LS tensors if provided by dataset
    tg_batch = batch.get('theta_global_ls', None)
    td_batch = batch.get('theta_dir_ls', None)
    if tg_batch is not None and td_batch is not None:
        return tg_batch.to(device, non_blocking=True), td_batch.to(device, non_blocking=True)

    case_names = batch.get('case_name', None)
    slice_idxs = batch.get('slice_idx', None)
    if case_names is None or slice_idxs is None or ls_root is None:
        return None, None
    if isinstance(case_names, list):
        names = case_names
    else:
        names = [str(x) for x in case_names]
    if torch.is_tensor(slice_idxs):
        zs = slice_idxs.detach().cpu().tolist()
    elif isinstance(slice_idxs, list):
        zs = [int(x) for x in slice_idxs]
    else:
        zs = [int(slice_idxs)]
    tg_list = []
    td_list = []
    for name, z in zip(names, zs):
        try:
            case_path = ls_root / name
            s0 = nib.load(str(case_path / 's0.nii.gz')).get_fdata(dtype=np.float32)
            r = nib.load(str(case_path / 'r.nii.gz')).get_fdata(dtype=np.float32)
            s20 = nib.load(str(case_path / 'sigma20.nii.gz')).get_fdata(dtype=np.float32)
            s30 = nib.load(str(case_path / 'sigma30.nii.gz')).get_fdata(dtype=np.float32)
            def _ld(n):
                return nib.load(str(case_path / f'{n}.nii.gz')).get_fdata(dtype=np.float32)
            D = _ld('D'); s11 = _ld('sigma11'); s02 = _ld('sigma02'); s21 = _ld('sigma21'); s12 = _ld('sigma12'); s03 = _ld('sigma03')
            # LS NIfTI maps under ls_root are stored in physical units
            # (same convention as predict.py). Convert them back to
            # model-space parameters before using as warm-start targets.
            # Globals: r_model = r_phys * T_SCALE, etc.
            s0_phys = s0[:, :, z]
            r_phys = r[:, :, z]
            s20_phys = s20[:, :, z]
            s30_phys = s30[:, :, z]

            lns0 = np.log(np.clip(s0_phys, 1e-6, None)).astype(np.float32)
            r_model = (r_phys * T_SCALE).astype(np.float32)
            s20_model = (s20_phys * (T_SCALE * T_SCALE)).astype(np.float32)
            s30_model = (s30_phys * (T_SCALE * T_SCALE * T_SCALE)).astype(np.float32)

            tg_np = np.stack([lns0, r_model, s20_model, s30_model], axis=0)

            def _dir_slice_model(arr4d: np.ndarray, scale: float):
                """Take one directional 4D map in physical units and
                convert to model-space [30,H,W] for this slice."""
                hw31_phys = arr4d[:, :, z, :].astype(np.float32)  # [H,W,31]
                hw30_phys = hw31_phys[:, :, 1:1+cfg.directions]
                hw30_model = hw30_phys * float(scale)
                return np.transpose(hw30_model, (2, 0, 1))  # [30,H,W]

            # Directional phys -> model scaling (must match predict.py):
            #   D_model       = D_phys * B_SCALE
            #   sigma11_model = sigma11_phys * (T_SCALE * B_SCALE)
            #   sigma02_model = sigma02_phys * (B_SCALE^2)
            #   sigma21_model = sigma21_phys * (T_SCALE^2 * B_SCALE)
            #   sigma12_model = sigma12_phys * (T_SCALE * B_SCALE^2)
            #   sigma03_model = sigma03_phys * (B_SCALE^3)
            td_np = np.stack([
                _dir_slice_model(D, B_SCALE),
                _dir_slice_model(s11, T_SCALE * B_SCALE),
                _dir_slice_model(s02, B_SCALE * B_SCALE),
                _dir_slice_model(s21, T_SCALE * T_SCALE * B_SCALE),
                _dir_slice_model(s12, T_SCALE * B_SCALE * B_SCALE),
                _dir_slice_model(s03, B_SCALE * B_SCALE * B_SCALE),
            ], axis=1)
            tg_list.append(torch.from_numpy(tg_np).float())
            td_list.append(torch.from_numpy(td_np).float())
        except Exception as e:
            print(f"[ls-init] 加载 {name} slice {z} 失败: {e}")
            return None, None
    tg = torch.stack(tg_list, dim=0).to(device, non_blocking=True)
    td = torch.stack(td_list, dim=0).to(device, non_blocking=True)
    return tg, td


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred is None or target is None:
        return torch.tensor(0.0, device=mask.device)
    if pred.dim() == 4:
        if mask.shape[1] == 1:
            mask_exp = mask.expand(-1, pred.shape[1], -1, -1)
        else:
            mask_exp = mask
        diff2 = (pred - target) ** 2
        num = (diff2 * mask_exp).sum()
        den = mask_exp.sum().clamp_min(1e-8)
        return num / den
    elif pred.dim() == 5:
        # pred: [B,C1,C2,H,W]; mask: [B,1,H,W] (voxel-level)
        # First lift mask to [B,1,1,H,W], then expand to [B,C1,C2,H,W].
        if mask.dim() == 4:
            mask_exp = mask.unsqueeze(1)  # [B,1,1,H,W]
        else:
            mask_exp = mask
        if mask_exp.shape[1] == 1 and mask_exp.shape[2] == 1:
            mask_exp = mask_exp.expand(-1, pred.shape[1], pred.shape[2], -1, -1)
        diff2 = (pred - target) ** 2
        num = (diff2 * mask_exp).sum()
        den = mask_exp.sum().clamp_min(1e-8)
        return num / den
    else:
        return ((pred - target) ** 2).mean()


def train_epoch(model, forward_model, criterion, loader, optimizer, device,
                t_vec, b_vec, P, epoch, scaler=None, use_amp=False,
                ls_root: Path = None, stage: int = 1, stage2_epoch: int = 0):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    # Stats (Scheme E)
    l1_sum = 0.0
    num_total = 0.0
    den_total = 0.0
    n_batches = 0
    # Diagnostics accumulators
    diag = {
        'fg_ratio_sum': 0.0,
        'S_hat_mean_sum': 0.0,
        'S_hat_std_sum': 0.0,
        'signal_mean_sum': 0.0,
        'signal_std_sum': 0.0,
        'signal_stage_mean_sum': 0.0,
        'signal_stage_std_sum': 0.0,
        'lns0_mean_sum': 0.0,
        'r_mean_sum': 0.0,
        'Dmean_mean_sum': 0.0,
        's0_mean_sum': 0.0,
        'r_true_mean_sum': 0.0,
        'D_true_mean_sum': 0.0,
        'sigma20_true_mean_sum': 0.0,
        'sigma30_true_mean_sum': 0.0,
        'sigma11_true_mean_sum': 0.0,
        'sigma02_true_mean_sum': 0.0,
        'sigma21_true_mean_sum': 0.0,
        'sigma12_true_mean_sum': 0.0,
        'sigma03_true_mean_sum': 0.0,
        's0_min': float('inf'),
        's0_max': -float('inf'),
        's0_var_sum': 0.0,
        'r_true_min': float('inf'),
        'r_true_max': -float('inf'),
        'r_true_var_sum': 0.0,
        'D_true_min': float('inf'),
        'D_true_max': -float('inf'),
        'D_true_var_sum': 0.0,
        'sigma20_true_min': float('inf'),
        'sigma20_true_max': -float('inf'),
        'sigma20_true_var_sum': 0.0,
        'sigma30_true_min': float('inf'),
        'sigma30_true_max': -float('inf'),
        'sigma30_true_var_sum': 0.0,
        'sigma11_true_min': float('inf'),
        'sigma11_true_max': -float('inf'),
        'sigma11_true_var_sum': 0.0,
        'sigma02_true_min': float('inf'),
        'sigma02_true_max': -float('inf'),
        'sigma02_true_var_sum': 0.0,
        'sigma21_true_min': float('inf'),
        'sigma21_true_max': -float('inf'),
        'sigma21_true_var_sum': 0.0,
        'sigma12_true_min': float('inf'),
        'sigma12_true_max': -float('inf'),
        'sigma12_true_var_sum': 0.0,
        'sigma03_true_min': float('inf'),
        'sigma03_true_max': -float('inf'),
        'sigma03_true_var_sum': 0.0,
    }

    # Two-stage (b0 vs DW) channel masks
    use_two_stage = getattr(cfg, 'use_two_stage_b0_dw', False)
    b0_ch_1d = None
    dw_ch_1d = None
    if use_two_stage:
        b0_th = getattr(cfg, 'b0_threshold', 1e-6)
        b0_ch_1d = (b_vec.abs() < b0_th)
        dw_ch_1d = ~b0_ch_1d

    for batch_idx, batch in enumerate(pbar):
        signal = batch['signal'].to(device, non_blocking=True)          # [B,630,H,W] 原始信号
        signal_net = normalize_signal_for_unet(signal)                  # 仅供 UNet 输入
        mask_voxel = batch['mask'].to(device, non_blocking=True)        # [B,1,H,W] 体素级最终 mask

        # 可选：观测级 630 通道 mask（来自预处理 temp_mask_630）
        mask_temp630 = batch.get('mask_temp630', None)
        if mask_temp630 is not None:
            mask_temp630 = mask_temp630.to(device, non_blocking=True)   # [B,630,H,W]

        # 构造用于 L1 损失的通道级 mask：voxel_mask_final ∧ temp_mask_630
        # 形状: [B,630,H,W]
        voxel_mask_ch = mask_voxel.expand(-1, signal.shape[1], -1, -1)  # [B,630,H,W]
        if mask_temp630 is not None:
            loss_mask = (voxel_mask_ch > 0.5) & (mask_temp630 > 0.5)
        else:
            # 兼容旧 cache：若无 temp_mask_630，仅使用体素级 mask
            loss_mask = (voxel_mask_ch > 0.5)
        loss_mask = loss_mask.float()

        # Stage-aware channel selection: Stage1 -> b0 only, Stage2 -> DW only
        if use_two_stage and b0_ch_1d is not None and dw_ch_1d is not None:
            if stage == 1:
                ch_1d = b0_ch_1d
            else:
                ch_1d = dw_ch_1d
            ch_mask_4d = ch_1d.view(1, -1, 1, 1).to(loss_mask.device)
            loss_mask = loss_mask * ch_mask_4d.float()

        if use_amp and scaler is not None:
            with autocast():
                signal_net_amp = normalize_signal_for_unet(signal)
                theta_global, theta_dir = model(signal_net_amp)
                if stage == 1:
                    tg, td, s2, s3 = apply_curriculum(theta_global, theta_dir, epoch)
                    # Stage1: 方向高阶在课程上和物理上都完全关闭
                    if getattr(cfg, 'use_two_stage_b0_dw', False):
                        td[:, :, 1:, :, :] = 0.0
                        s2 = 0.0
                        s3 = 0.0
                else:
                    # Stage2: 关闭课程学习，直接使用完整高阶参数
                    tg = theta_global
                    td = theta_dir
                    s2 = 1.0
                    s3 = 1.0
                # 使用通道级 loss_mask 进行物理 L1 损失
                loss, loss_dict = criterion(tg, td, signal, t_vec, b_vec, P, loss_mask)
                if getattr(cfg, 'use_ls_init', False) and ls_root is not None:
                    tg_ls, td_ls = _build_ls_targets(batch, ls_root, device)
                    if tg_ls is not None and td_ls is not None:
                        w = _param_weight(epoch, stage, stage2_epoch)
                        if w > 0:
                            # 参数对齐损失仍基于体素级 mask
                            loss = loss + w * (_masked_mse(tg, tg_ls, mask_voxel) + _masked_mse(td, td_ls, mask_voxel))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            theta_global, theta_dir = model(signal_net)
            if stage == 1:
                tg, td, s2, s3 = apply_curriculum(theta_global, theta_dir, epoch)
                # Stage1: 方向高阶在课程上和物理上都完全关闭
                if getattr(cfg, 'use_two_stage_b0_dw', False):
                    td[:, :, 1:, :, :] = 0.0
                    s2 = 0.0
                    s3 = 0.0
            else:
                # Stage2: 关闭课程学习，直接使用完整高阶参数
                tg = theta_global
                td = theta_dir
                s2 = 1.0
                s3 = 1.0
            # 使用通道级 loss_mask 进行物理 L1 损失
            loss, loss_dict = criterion(tg, td, signal, t_vec, b_vec, P, loss_mask)
            if getattr(cfg, 'use_ls_init', False) and ls_root is not None:
                tg_ls, td_ls = _build_ls_targets(batch, ls_root, device)
                if tg_ls is not None and td_ls is not None:
                    w = _param_weight(epoch, stage, stage2_epoch)
                    if w > 0:
                        # 参数对齐损失仍基于体素级 mask
                        loss = loss + w * (_masked_mse(tg, tg_ls, mask_voxel) + _masked_mse(td, td_ls, mask_voxel))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Aggregated logging (recompute S_hat once for stats)
        with torch.no_grad():
            S_hat = forward_model(tg, td, t_vec, b_vec, P)
            # 统计时的 L1 与训练损失保持一致，使用通道级 loss_mask
            l1_now, num, den = masked_mean_l1(S_hat, signal, loss_mask, getattr(criterion, 'channel_mask', None))
            l1_sum += float(l1_now.item())
            num_total += float(num.item())
            den_total += float(den.item())
            n_batches += 1

            # Foreground ratio
            # 前景比例基于体素级 mask
            if epoch == 1:
                fg_ratio = (mask_voxel.sum(dim=(1, 2, 3)) / (mask_voxel.shape[2] * mask_voxel.shape[3])).mean().item()
                diag['fg_ratio_sum'] += fg_ratio

            # Signal stats
            # 使用与当前阶段相同的通道级 mask（Stage1: 仅 b0；Stage2: 仅 DW）统计 S_hat
            s_shat = _masked_stats_4d(S_hat, loss_mask)
            diag['S_hat_mean_sum'] += s_shat['mean']
            diag['S_hat_std_sum'] += s_shat['std']
            if epoch == 1:
                s_sig = _masked_stats_4d(signal, mask_voxel)
                diag['signal_mean_sum'] += s_sig['mean']
                diag['signal_std_sum'] += s_sig['std']
            # 当前阶段通道（基于 loss_mask）的原始信号统计
            s_sig_stage = _masked_stats_4d(signal, loss_mask)
            diag['signal_stage_mean_sum'] += s_sig_stage['mean']
            diag['signal_stage_std_sum'] += s_sig_stage['std']

            # Param stats
            lns0 = tg[:, 0:1]
            rpar = tg[:, 1:2]
            sigma20 = tg[:, 2:3]
            sigma30 = tg[:, 3:4]
            D_all = td[:, :, 0, :, :]
            sigma11_all = td[:, :, 1, :, :]
            sigma02_all = td[:, :, 2, :, :]
            sigma21_all = td[:, :, 3, :, :]
            sigma12_all = td[:, :, 4, :, :]
            sigma03_all = td[:, :, 5, :, :]
            Dmean = D_all.mean(dim=1, keepdim=True)
            sigma11_mean = sigma11_all.mean(dim=1, keepdim=True)
            sigma02_mean = sigma02_all.mean(dim=1, keepdim=True)
            sigma21_mean = sigma21_all.mean(dim=1, keepdim=True)
            sigma12_mean = sigma12_all.mean(dim=1, keepdim=True)
            sigma03_mean = sigma03_all.mean(dim=1, keepdim=True)
            diag['lns0_mean_sum'] += _masked_stats_4d(lns0, mask_voxel)['mean']
            diag['r_mean_sum'] += _masked_stats_4d(rpar, mask_voxel)['mean']
            diag['Dmean_mean_sum'] += _masked_stats_4d(Dmean, mask_voxel)['mean']

            # 在日志统计中使用 float32 计算 exp，避免 AMP 半精度下的溢出导致 inf/nan
            s0_true = torch.exp(lns0.float())
            r_true = rpar / T_SCALE
            D_true = Dmean / B_SCALE
            sigma20_true = sigma20 / (T_SCALE * T_SCALE)
            sigma30_true = sigma30 / (T_SCALE * T_SCALE * T_SCALE)
            sigma11_true = sigma11_mean / (T_SCALE * B_SCALE)
            sigma02_true = sigma02_mean / (B_SCALE * B_SCALE)
            sigma21_true = sigma21_mean / (T_SCALE * T_SCALE * B_SCALE)
            sigma12_true = sigma12_mean / (T_SCALE * B_SCALE * B_SCALE)
            sigma03_true = sigma03_mean / (B_SCALE * B_SCALE * B_SCALE)

            s0_stats = _masked_stats_4d(s0_true, mask_voxel)
            r_true_stats = _masked_stats_4d(r_true, mask_voxel)
            D_true_stats = _masked_stats_4d(D_true, mask_voxel)
            sigma20_stats = _masked_stats_4d(sigma20_true, mask_voxel)
            sigma30_stats = _masked_stats_4d(sigma30_true, mask_voxel)
            sigma11_stats = _masked_stats_4d(sigma11_true, mask_voxel)
            sigma02_stats = _masked_stats_4d(sigma02_true, mask_voxel)
            sigma21_stats = _masked_stats_4d(sigma21_true, mask_voxel)
            sigma12_stats = _masked_stats_4d(sigma12_true, mask_voxel)
            sigma03_stats = _masked_stats_4d(sigma03_true, mask_voxel)

            diag['s0_mean_sum'] += s0_stats['mean']
            diag['r_true_mean_sum'] += r_true_stats['mean']
            diag['D_true_mean_sum'] += D_true_stats['mean']
            diag['sigma20_true_mean_sum'] += sigma20_stats['mean']
            diag['sigma30_true_mean_sum'] += sigma30_stats['mean']
            diag['sigma11_true_mean_sum'] += sigma11_stats['mean']
            diag['sigma02_true_mean_sum'] += sigma02_stats['mean']
            diag['sigma21_true_mean_sum'] += sigma21_stats['mean']
            diag['sigma12_true_mean_sum'] += sigma12_stats['mean']
            diag['sigma03_true_mean_sum'] += sigma03_stats['mean']

            diag['s0_min'] = min(diag['s0_min'], s0_stats['min'])
            diag['s0_max'] = max(diag['s0_max'], s0_stats['max'])
            diag['s0_var_sum'] += s0_stats['std'] ** 2
            diag['r_true_min'] = min(diag['r_true_min'], r_true_stats['min'])
            diag['r_true_max'] = max(diag['r_true_max'], r_true_stats['max'])
            diag['r_true_var_sum'] += r_true_stats['std'] ** 2
            diag['D_true_min'] = min(diag['D_true_min'], D_true_stats['min'])
            diag['D_true_max'] = max(diag['D_true_max'], D_true_stats['max'])
            diag['D_true_var_sum'] += D_true_stats['std'] ** 2
            diag['sigma20_true_min'] = min(diag['sigma20_true_min'], sigma20_stats['min'])
            diag['sigma20_true_max'] = max(diag['sigma20_true_max'], sigma20_stats['max'])
            diag['sigma20_true_var_sum'] += sigma20_stats['std'] ** 2
            diag['sigma30_true_min'] = min(diag['sigma30_true_min'], sigma30_stats['min'])
            diag['sigma30_true_max'] = max(diag['sigma30_true_max'], sigma30_stats['max'])
            diag['sigma30_true_var_sum'] += sigma30_stats['std'] ** 2
            diag['sigma11_true_min'] = min(diag['sigma11_true_min'], sigma11_stats['min'])
            diag['sigma11_true_max'] = max(diag['sigma11_true_max'], sigma11_stats['max'])
            diag['sigma11_true_var_sum'] += sigma11_stats['std'] ** 2
            diag['sigma02_true_min'] = min(diag['sigma02_true_min'], sigma02_stats['min'])
            diag['sigma02_true_max'] = max(diag['sigma02_true_max'], sigma02_stats['max'])
            diag['sigma02_true_var_sum'] += sigma02_stats['std'] ** 2
            diag['sigma21_true_min'] = min(diag['sigma21_true_min'], sigma21_stats['min'])
            diag['sigma21_true_max'] = max(diag['sigma21_true_max'], sigma21_stats['max'])
            diag['sigma21_true_var_sum'] += sigma21_stats['std'] ** 2
            diag['sigma12_true_min'] = min(diag['sigma12_true_min'], sigma12_stats['min'])
            diag['sigma12_true_max'] = max(diag['sigma12_true_max'], sigma12_stats['max'])
            diag['sigma12_true_var_sum'] += sigma12_stats['std'] ** 2
            diag['sigma03_true_min'] = min(diag['sigma03_true_min'], sigma03_stats['min'])
            diag['sigma03_true_max'] = max(diag['sigma03_true_max'], sigma03_stats['max'])
            diag['sigma03_true_var_sum'] += sigma03_stats['std'] ** 2

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'L1(batch)': f"{l1_now.item():.4f}",
                'cur(s2,s3)': f"({s2:.3f},{s3:.3f})"
            })

    # Epoch results
    l1_mean = l1_sum / max(n_batches, 1)
    l1_numden = (num_total / max(den_total, 1e-8))
    # Diagnostics averages
    fg_ratio_avg = diag['fg_ratio_sum'] / max(n_batches, 1)
    S_hat_mean_avg = diag['S_hat_mean_sum'] / max(n_batches, 1)
    S_hat_std_avg = diag['S_hat_std_sum'] / max(n_batches, 1)
    signal_mean_avg = diag['signal_mean_sum'] / max(n_batches, 1)
    signal_std_avg = diag['signal_std_sum'] / max(n_batches, 1)
    signal_stage_mean_avg = diag['signal_stage_mean_sum'] / max(n_batches, 1)
    signal_stage_std_avg = diag['signal_stage_std_sum'] / max(n_batches, 1)
    lns0_mean_avg = diag['lns0_mean_sum'] / max(n_batches, 1)
    r_mean_avg = diag['r_mean_sum'] / max(n_batches, 1)
    Dmean_mean_avg = diag['Dmean_mean_sum'] / max(n_batches, 1)
    s0_mean_avg = diag['s0_mean_sum'] / max(n_batches, 1)
    r_true_mean_avg = diag['r_true_mean_sum'] / max(n_batches, 1)
    D_true_mean_avg = diag['D_true_mean_sum'] / max(n_batches, 1)
    sigma20_true_mean_avg = diag['sigma20_true_mean_sum'] / max(n_batches, 1)
    sigma30_true_mean_avg = diag['sigma30_true_mean_sum'] / max(n_batches, 1)
    sigma11_true_mean_avg = diag['sigma11_true_mean_sum'] / max(n_batches, 1)
    sigma02_true_mean_avg = diag['sigma02_true_mean_sum'] / max(n_batches, 1)
    sigma21_true_mean_avg = diag['sigma21_true_mean_sum'] / max(n_batches, 1)
    sigma12_true_mean_avg = diag['sigma12_true_mean_sum'] / max(n_batches, 1)
    sigma03_true_mean_avg = diag['sigma03_true_mean_sum'] / max(n_batches, 1)

    s0_var_avg = diag['s0_var_sum'] / max(n_batches, 1)
    r_true_var_avg = diag['r_true_var_sum'] / max(n_batches, 1)
    D_true_var_avg = diag['D_true_var_sum'] / max(n_batches, 1)
    sigma20_true_var_avg = diag['sigma20_true_var_sum'] / max(n_batches, 1)
    sigma30_true_var_avg = diag['sigma30_true_var_sum'] / max(n_batches, 1)
    sigma11_true_var_avg = diag['sigma11_true_var_sum'] / max(n_batches, 1)
    sigma02_true_var_avg = diag['sigma02_true_var_sum'] / max(n_batches, 1)
    sigma21_true_var_avg = diag['sigma21_true_var_sum'] / max(n_batches, 1)
    sigma12_true_var_avg = diag['sigma12_true_var_sum'] / max(n_batches, 1)
    sigma03_true_var_avg = diag['sigma03_true_var_sum'] / max(n_batches, 1)

    return {
        'l1_mean': l1_mean,
        'l1_numden': l1_numden,
        'num_total': num_total,
        'den_total': den_total,
        'fg_ratio': fg_ratio_avg,
        'S_hat_mean': S_hat_mean_avg,
        'S_hat_std': S_hat_std_avg,
        'signal_mean': signal_mean_avg,
        'signal_std': signal_std_avg,
        'signal_stage_mean': signal_stage_mean_avg,
        'signal_stage_std': signal_stage_std_avg,
        'lns0_mean': lns0_mean_avg,
        'r_mean': r_mean_avg,
        'Dmean_mean': Dmean_mean_avg,
        's0_mean': s0_mean_avg,
        'r_true_mean': r_true_mean_avg,
        'D_true_mean': D_true_mean_avg,
        'sigma20_true_mean': sigma20_true_mean_avg,
        'sigma30_true_mean': sigma30_true_mean_avg,
        'sigma11_true_mean': sigma11_true_mean_avg,
        'sigma02_true_mean': sigma02_true_mean_avg,
        'sigma21_true_mean': sigma21_true_mean_avg,
        'sigma12_true_mean': sigma12_true_mean_avg,
        'sigma03_true_mean': sigma03_true_mean_avg,
        's0_min': diag['s0_min'],
        's0_max': diag['s0_max'],
        's0_var': s0_var_avg,
        'r_true_min': diag['r_true_min'],
        'r_true_max': diag['r_true_max'],
        'r_true_var': r_true_var_avg,
        'D_true_min': diag['D_true_min'],
        'D_true_max': diag['D_true_max'],
        'D_true_var': D_true_var_avg,
        'sigma20_true_min': diag['sigma20_true_min'],
        'sigma20_true_max': diag['sigma20_true_max'],
        'sigma20_true_var': sigma20_true_var_avg,
        'sigma30_true_min': diag['sigma30_true_min'],
        'sigma30_true_max': diag['sigma30_true_max'],
        'sigma30_true_var': sigma30_true_var_avg,
        'sigma11_true_min': diag['sigma11_true_min'],
        'sigma11_true_max': diag['sigma11_true_max'],
        'sigma11_true_var': sigma11_true_var_avg,
        'sigma02_true_min': diag['sigma02_true_min'],
        'sigma02_true_max': diag['sigma02_true_max'],
        'sigma02_true_var': sigma02_true_var_avg,
        'sigma21_true_min': diag['sigma21_true_min'],
        'sigma21_true_max': diag['sigma21_true_max'],
        'sigma21_true_var': sigma21_true_var_avg,
        'sigma12_true_min': diag['sigma12_true_min'],
        'sigma12_true_max': diag['sigma12_true_max'],
        'sigma12_true_var': sigma12_true_var_avg,
        'sigma03_true_min': diag['sigma03_true_min'],
        'sigma03_true_max': diag['sigma03_true_max'],
        'sigma03_true_var': sigma03_true_var_avg,
    }


@torch.no_grad()
def validate(model, forward_model, criterion, loader, device, t_vec, b_vec, P, epoch, use_amp=False, ls_root: Path = None, stage: int = 1):
    model.eval()
    l1_sum = 0.0
    num_total = 0.0
    den_total = 0.0
    n_batches = 0

    # Two-stage (b0 vs DW) channel masks
    use_two_stage = getattr(cfg, 'use_two_stage_b0_dw', False)
    b0_ch_1d = None
    dw_ch_1d = None
    if use_two_stage:
        b0_th = getattr(cfg, 'b0_threshold', 1e-6)
        b0_ch_1d = (b_vec.abs() < b0_th)
        dw_ch_1d = ~b0_ch_1d

    for batch in tqdm(loader, desc="Validating"):
        signal = batch['signal'].to(device, non_blocking=True)          # [B,630,H,W] 原始信号
        signal_net = normalize_signal_for_unet(signal)                  # 仅供 UNet 输入
        mask_voxel = batch['mask'].to(device, non_blocking=True)        # [B,1,H,W]

        # 可选：观测级 630 通道 mask（来自预处理 temp_mask_630）
        mask_temp630 = batch.get('mask_temp630', None)
        if mask_temp630 is not None:
            mask_temp630 = mask_temp630.to(device, non_blocking=True)   # [B,630,H,W]

        # 构造与 train_epoch 相同的通道级 loss_mask
        voxel_mask_ch = mask_voxel.expand(-1, signal.shape[1], -1, -1)  # [B,630,H,W]
        if mask_temp630 is not None:
            loss_mask = (voxel_mask_ch > 0.5) & (mask_temp630 > 0.5)
        else:
            loss_mask = (voxel_mask_ch > 0.5)
        loss_mask = loss_mask.float()

        # Stage-aware channel selection: Stage1 -> b0 only, Stage2 -> DW only
        if use_two_stage and b0_ch_1d is not None and dw_ch_1d is not None:
            if stage == 1:
                ch_1d = b0_ch_1d
            else:
                ch_1d = dw_ch_1d
            ch_mask_4d = ch_1d.view(1, -1, 1, 1).to(loss_mask.device)
            loss_mask = loss_mask * ch_mask_4d.float()
        if use_amp:
            with autocast():
                signal_net_amp = normalize_signal_for_unet(signal)
                theta_global, theta_dir = model(signal_net_amp)
        else:
            theta_global, theta_dir = model(signal_net)
        if stage == 1:
            tg, td, s2, s3 = apply_curriculum(theta_global, theta_dir, epoch)
            # Stage1: 方向高阶在课程上和物理上都完全关闭，保持与 train_epoch 一致
            if getattr(cfg, 'use_two_stage_b0_dw', False):
                td[:, :, 1:, :, :] = 0.0
                s2 = 0.0
                s3 = 0.0
        else:
            # Stage2: 关闭课程学习，直接使用完整高阶参数
            tg = theta_global
            td = theta_dir
            s2 = 1.0
            s3 = 1.0
        S_hat = forward_model(tg, td, t_vec, b_vec, P)
        # 使用与训练阶段一致的通道级 loss_mask 计算验证 L1
        l1_now, num, den = masked_mean_l1(S_hat, signal, loss_mask, getattr(criterion, 'channel_mask', None))
        l1_sum += float(l1_now.item())
        num_total += float(num.item())
        den_total += float(den.item())
        n_batches += 1
    l1_mean = l1_sum / max(n_batches, 1)
    l1_numden = (num_total / max(den_total, 1e-8))
    return {
        'l1_mean': l1_mean,
        'l1_numden': l1_numden,
        'num_total': num_total,
        'den_total': den_total,
    }


def main():
    print("=" * 60)
    print("RDMRI UNet Simple - train_A (Curriculum + Correct Logging)")
    print("=" * 60)

    device = select_gpu()
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)
    # 每次运行重新创建日志文件，用于按 epoch 追加训练日志
    epoch_log_path = save_dir / "train_log.txt"
    with epoch_log_path.open("w", encoding="utf-8") as f:
        f.write("train_A epoch logs\n")
    writer = SummaryWriter(log_dir='runs/train_A')

    # 1) Datasets（仅使用 CachedMRIDataset 预处理缓存）
    print("\n加载数据（CachedMRIDataset）...")
    grad_csv = Path(cfg.data_root) / "grad_126.csv"
    train_dataset = CachedMRIDataset(
        cfg.data_root, split=cfg.train_dir,
        cache_dir=cfg.cache_dir,
        grad_csv_path=grad_csv, tes=cfg.tes,
        t_ref_idx=getattr(cfg, 't_ref_idx', 0),
    )
    val_dataset = CachedMRIDataset(
        cfg.data_root, split=cfg.val_dir,
        cache_dir=cfg.cache_dir,
        grad_csv_path=grad_csv, tes=cfg.tes,
        t_ref_idx=getattr(cfg, 't_ref_idx', 0),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=getattr(cfg, 'prefetch_factor', 4),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=getattr(cfg, 'prefetch_factor', 4),
        persistent_workers=True,
    )

    # 2) Protocol vectors
    print("\n获取协议向量...")
    t_vec = train_dataset.t_vec.to(device, non_blocking=True)
    b_vec = train_dataset.b_vec.to(device, non_blocking=True)
    P = train_dataset.P.to(device, non_blocking=True)
    print(f"✓ 协议: t_vec {tuple(t_vec.shape)}, b_vec {tuple(b_vec.shape)}, P {tuple(P.shape)}")

    # 3) Model
    print("\n创建模型与物理前向...")
    model = UNetWithHeads(
        in_channels=cfg.in_channels,
        mid_channels=cfg.mid_channels,
        n_dirs=cfg.directions,
    ).to(device)
    forward_model = TaylorForward().to(device)
    criterion = L1PhysicsLoss(
        forward_model=forward_model,
        exclude_ref_b0=getattr(cfg, 'exclude_ref_b0_in_loss', True),
        t_vec=t_vec, b_vec=b_vec,
        t_ref_idx=getattr(cfg, 't_ref_idx', 0),
        use_huber=getattr(cfg, 'use_huber', False),
        huber_delta=getattr(cfg, 'huber_delta', 0.3),
        ignore_invalid_signal=getattr(cfg, 'ignore_invalid_signal', True),
        signal_min_valid=getattr(cfg, 'signal_min_valid', 0.0),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.unet.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": model.heads.global_head.parameters(), "lr": cfg.lr_r, "weight_decay": 0.0},
            {"params": model.heads.dir_head.parameters(), "lr": cfg.lr_high_order, "weight_decay": getattr(cfg, 'dir_weight_decay', 0.0)},
        ],
    )

    # Optional: resume from best checkpoint (only weights/optimizer states, LR 仍由后续调度逻辑控制)
    start_epoch = 1
    if getattr(cfg, 'resume_from_best', False):
        ckpt_path = save_dir / getattr(cfg, 'resume_ckpt_name', 'best_model_A.pt')
        if ckpt_path.is_file():
            print(f"\n[Resume] Loading checkpoint from {ckpt_path} ...")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            print(f"[Resume] Resume start epoch: {start_epoch}")
        else:
            print(f"[Resume] Checkpoint {ckpt_path} not found, start from scratch.")

    # 4) Train loop
    print("\n开始训练（方案A+E + Two-Stage b0/DW）")
    print(f"Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"Curriculum schedule: Stage1 1..{STAGE1_EPOCHS} (0/1阶) | Stage2 {STAGE1_EPOCHS+1}..{STAGE2_EPOCHS} (2阶×{SCALE2_STAGE2}) | Stage3 {STAGE2_EPOCHS+1}..{STAGE3_EPOCHS} (3阶×{SCALE3_STAGE3}) | >{STAGE3_EPOCHS} 全开启")
    print("=" * 60)

    best_val = float('inf')
    best_stage1_val = float('inf')
    best_stage2_val = float('inf')
    stage1_no_improve = 0
    stage2_no_improve = 0
    stage2_epoch = 0
    in_stage2 = False  # False: Stage1 (b0), True: Stage2 (DW)
    scaler = GradScaler(enabled=getattr(cfg, 'use_amp', False))
    train_start_time = time.time()
    base_fg_ratio = None
    base_signal_mean = None
    base_signal_std = None
    stage1_signal_stage_mean = None
    stage1_signal_stage_std = None
    stage2_signal_stage_mean = None
    stage2_signal_stage_std = None
    stage1_min_epochs = getattr(cfg, 'stage1_min_epochs', 0)
    stage1_max_epochs = getattr(cfg, 'stage1_epochs', cfg.epochs)
    stage1_patience = getattr(cfg, 'stage1_patience', 0)
    stage1_improve_tol = getattr(cfg, 'stage1_improve_tol', 0.0)
    stage1_global_warm_epochs = getattr(cfg, 'stage1_global_warm_epochs', 10)
    stage1_global_warm_lr = getattr(cfg, 'stage1_global_warm_lr', 1e-4)
    stage2_dir_lr = getattr(cfg, 'stage2_dir_lr', cfg.lr_high_order)
    stage2_min_epochs = getattr(cfg, 'stage2_min_epochs', 0)
    stage2_patience = getattr(cfg, 'stage2_patience', 0)
    stage2_improve_tol = getattr(cfg, 'stage2_improve_tol', 0.0)

    # 如果从已经超过 Stage1 上限的 epoch 恢复，并且启用了两阶段训练，
    # 则直接从 Stage2 开始，避免把方向高阶当成 Stage1 又清零一次。
    if getattr(cfg, 'use_two_stage_b0_dw', False) and start_epoch > stage1_max_epochs:
        in_stage2 = True
        stage2_epoch = max(0, start_epoch - stage1_max_epochs)

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Current stage for this epoch
        stage = 2 if in_stage2 and getattr(cfg, 'use_two_stage_b0_dw', False) else 1

        # ----- Per-stage / per-epoch learning rate schedule -----
        if stage == 1:
            # Stage1 (b0-only): global_head warmup + no dir_head learning
            if epoch <= stage1_global_warm_epochs:
                # Warmup: larger LR for global_head
                optimizer.param_groups[1]['lr'] = stage1_global_warm_lr
            else:
                # After warmup: use base lr_r
                optimizer.param_groups[1]['lr'] = cfg.lr_r
            # Stage1 不学习方向参数
            optimizer.param_groups[2]['lr'] = 0.0
            # UNet 使用基础 lr
            optimizer.param_groups[0]['lr'] = cfg.lr
        else:
            # Stage2 (DW-only): freeze UNet + global_head, train dir_head only
            optimizer.param_groups[0]['lr'] = 0.0
            optimizer.param_groups[1]['lr'] = 0.0
            optimizer.param_groups[2]['lr'] = stage2_dir_lr
        epoch_start_time = time.time()
        train_metrics = train_epoch(
            model, forward_model, criterion, train_loader, optimizer,
            device, t_vec, b_vec, P, epoch,
            scaler=scaler if getattr(cfg, 'use_amp', False) else None,
            use_amp=getattr(cfg, 'use_amp', False),
            ls_root=(Path(cfg.data_root) / getattr(cfg, 'ls_init_dir', 'ls_init') / cfg.train_dir) if getattr(cfg, 'use_ls_init', False) else None,
            stage=stage,
            stage2_epoch=(stage2_epoch + 1) if stage == 2 else 0,
        )
        val_metrics = validate(
            model, forward_model, criterion, val_loader,
            device, t_vec, b_vec, P, epoch,
            use_amp=getattr(cfg, 'use_amp', False),
            ls_root=(Path(cfg.data_root) / getattr(cfg, 'ls_init_dir', 'ls_init') / cfg.val_dir) if getattr(cfg, 'use_ls_init', False) else None,
            stage=stage,
        )

        if epoch == 1:
            base_fg_ratio = train_metrics['fg_ratio']
            base_signal_mean = train_metrics['signal_mean']
            base_signal_std = train_metrics['signal_std']

        # 仅在每个阶段开始时记录一次当前阶段通道的信号 mean/std
        if stage == 1 and stage1_signal_stage_mean is None:
            stage1_signal_stage_mean = train_metrics['signal_stage_mean']
            stage1_signal_stage_std = train_metrics['signal_stage_std']
        if stage == 2 and stage2_signal_stage_mean is None:
            stage2_signal_stage_mean = train_metrics['signal_stage_mean']
            stage2_signal_stage_std = train_metrics['signal_stage_std']

        fg_disp = base_fg_ratio if base_fg_ratio is not None else train_metrics['fg_ratio']
        signal_mean_disp = base_signal_mean if base_signal_mean is not None else train_metrics['signal_mean']
        signal_std_disp = base_signal_std if base_signal_std is not None else train_metrics['signal_std']
        if stage == 1 and stage1_signal_stage_mean is not None:
            stage_signal_mean_disp = stage1_signal_stage_mean
            stage_signal_std_disp = stage1_signal_stage_std
        elif stage == 2 and stage2_signal_stage_mean is not None:
            stage_signal_mean_disp = stage2_signal_stage_mean
            stage_signal_std_disp = stage2_signal_stage_std
        else:
            stage_signal_mean_disp = train_metrics['signal_stage_mean']
            stage_signal_std_disp = train_metrics['signal_stage_std']

        # Logs：统一打印到控制台并追加到文本文件
        epoch_lines = [
            f"\nEpoch {epoch}/{cfg.epochs}",
            f"  Train - L1_mean: {train_metrics['l1_mean']:.6f} | L1_numden: {train_metrics['l1_numden']:.6f}",
            f"          fg_ratio: {fg_disp:.4f} | signal mean/std: {signal_mean_disp:.4f}/{signal_std_disp:.4f} | stage-ch mean/std: {stage_signal_mean_disp:.4f}/{stage_signal_std_disp:.4f}",
            f"          S_hat mean/std: {train_metrics['S_hat_mean']:.4f}/{train_metrics['S_hat_std']:.4f}",
            f"          params true stats s0: mean={train_metrics['s0_mean']:.4f}, min={train_metrics['s0_min']:.4f}, max={train_metrics['s0_max']:.4f}, var={train_metrics['s0_var']:.4e}",
            f"                             r: mean={train_metrics['r_true_mean']:.6f}, min={train_metrics['r_true_min']:.6f}, max={train_metrics['r_true_max']:.6f}, var={train_metrics['r_true_var']:.4e}",
            f"                             D: mean={train_metrics['D_true_mean']:.6f}, min={train_metrics['D_true_min']:.6f}, max={train_metrics['D_true_max']:.6f}, var={train_metrics['D_true_var']:.4e}",
            f"          2nd true stats  sigma20: mean={train_metrics['sigma20_true_mean']:.6e}, min={train_metrics['sigma20_true_min']:.6e}, max={train_metrics['sigma20_true_max']:.6e}, var={train_metrics['sigma20_true_var']:.4e}",
            f"                             sigma11: mean={train_metrics['sigma11_true_mean']:.6e}, min={train_metrics['sigma11_true_min']:.6e}, max={train_metrics['sigma11_true_max']:.6e}, var={train_metrics['sigma11_true_var']:.4e}",
            f"                             sigma02: mean={train_metrics['sigma02_true_mean']:.6e}, min={train_metrics['sigma02_true_min']:.6e}, max={train_metrics['sigma02_true_max']:.6e}, var={train_metrics['sigma02_true_var']:.4e}",
            f"          3rd true stats  sigma30: mean={train_metrics['sigma30_true_mean']:.6e}, min={train_metrics['sigma30_true_min']:.6e}, max={train_metrics['sigma30_true_max']:.6e}, var={train_metrics['sigma30_true_var']:.4e}",
            f"                             sigma21: mean={train_metrics['sigma21_true_mean']:.6e}, min={train_metrics['sigma21_true_min']:.6e}, max={train_metrics['sigma21_true_max']:.6e}, var={train_metrics['sigma21_true_var']:.4e}",
            f"                             sigma12: mean={train_metrics['sigma12_true_mean']:.6e}, min={train_metrics['sigma12_true_min']:.6e}, max={train_metrics['sigma12_true_max']:.6e}, var={train_metrics['sigma12_true_var']:.4e}",
            f"                             sigma03: mean={train_metrics['sigma03_true_mean']:.6e}, min={train_metrics['sigma03_true_min']:.6e}, max={train_metrics['sigma03_true_max']:.6e}, var={train_metrics['sigma03_true_var']:.4e}",
            f"  Val   - L1_mean: {val_metrics['l1_mean']:.6f} | L1_numden: {val_metrics['l1_numden']:.6f}",
        ]
        for line in epoch_lines:
            print(line)
        with epoch_log_path.open("a", encoding="utf-8") as f:
            for line in epoch_lines:
                f.write(line + "\n")

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        epoch_min, epoch_sec = divmod(int(epoch_elapsed), 60)
        total_min, total_sec = divmod(int(total_elapsed), 60)
        print(f"  Time - This epoch: {epoch_min}m {epoch_sec}s | Total: {total_min}m {total_sec}s")

        writer.add_scalar('A/train_l1_mean', train_metrics['l1_mean'], epoch)
        writer.add_scalar('A/train_l1_numden', train_metrics['l1_numden'], epoch)
        writer.add_scalar('A/val_l1_mean', val_metrics['l1_mean'], epoch)
        writer.add_scalar('A/val_l1_numden', val_metrics['l1_numden'], epoch)

        # Save best on val aggregated L1
        if val_metrics['l1_numden'] < best_val:
            best_val = val_metrics['l1_numden']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l1_numden': best_val,
            }, save_dir / 'best_model_A.pt')
            print(f"  ✓ 保存最佳模型 (val L1_numden: {best_val:.6f})")

        # Stage1 convergence / max-epoch check: decide whether to switch to Stage2 from next epoch
        if stage == 1 and getattr(cfg, 'use_two_stage_b0_dw', False):
            # Hard cap: if reached max Stage1 epochs, force switch
            if epoch >= stage1_max_epochs:
                if not in_stage2:
                    in_stage2 = True
                    if getattr(cfg, 'freeze_global_after_stage1', False):
                        optimizer.param_groups[0]['lr'] = 0.0
                        optimizer.param_groups[1]['lr'] = 0.0
                    print(f"[TwoStage] Reached Stage1 max epochs ({stage1_max_epochs}), switching to Stage2 from next epoch.")
            else:
                # Early-stop style convergence based on val L1_numden
                if val_metrics['l1_numden'] + stage1_improve_tol < best_stage1_val:
                    best_stage1_val = val_metrics['l1_numden']
                    stage1_no_improve = 0
                else:
                    stage1_no_improve += 1

                if (epoch >= stage1_min_epochs) and (stage1_no_improve >= stage1_patience) and (not in_stage2):
                    in_stage2 = True
                    if getattr(cfg, 'freeze_global_after_stage1', False):
                        optimizer.param_groups[0]['lr'] = 0.0
                        optimizer.param_groups[1]['lr'] = 0.0
                    print(f"[TwoStage] Stage1 converged at epoch {epoch} (no improvement for {stage1_no_improve} epochs), switching to Stage2 from next epoch.")

        # Stage2 convergence check (DW-only): early stop entire training when converged
        if stage == 2 and getattr(cfg, 'use_two_stage_b0_dw', False):
            stage2_epoch += 1
            if val_metrics['l1_numden'] + stage2_improve_tol < best_stage2_val:
                best_stage2_val = val_metrics['l1_numden']
                stage2_no_improve = 0
            else:
                stage2_no_improve += 1

            if (stage2_epoch >= stage2_min_epochs) and (stage2_no_improve >= stage2_patience):
                print(f"[TwoStage] Stage2 converged at epoch {epoch} (Stage2 epochs: {stage2_epoch}, no improvement for {stage2_no_improve} epochs). Early stop training.")
                break

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'checkpoint_A_epoch{epoch}.pt')

    print("\n训练完成 (train_A)")
    writer.close()


if __name__ == '__main__':
    main()
