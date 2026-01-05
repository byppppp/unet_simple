"""
损失函数 - 简化版

仅使用基于物理前向模型的 L1 损失
"""

import torch
import torch.nn as nn
from typing import Optional


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    计算带空间 mask 的通道平均：mean(sum(x*mask)/sum(mask))。

    约定：
    - x: [B, C, H, W]
    - mask: [B, 1, H, W] 或 [B, C, H, W]
    """
    if mask is None:
        return x.mean()

    if x.dim() != 4 or mask.dim() != 4:
        raise ValueError("x 与 mask 需为 4D 张量 [B,C,H,W]")
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("mask 的 batch/空间尺寸需与 x 一致")

    if mask.shape[1] == 1:
        mask_expanded = mask.expand(-1, x.shape[1], -1, -1)
    elif mask.shape[1] == x.shape[1]:
        mask_expanded = mask
    else:
        raise ValueError("mask 的通道数与 x 不匹配")

    num = (x * mask_expanded).sum()
    den = mask_expanded.sum().clamp_min(1e-8)
    return num / den


class L1PhysicsLoss(nn.Module):
    """
    基于物理一致性的 L1 损失。

    思路：
    1) 通过物理前向模型从参数合成信号 S_hat；
    2) 计算 S_hat 与观测信号 S 的 L1 距离；
    3) 可选：在损失中排除指定通道。
    """

    def __init__(self, forward_model, exclude_ref_b0: bool = False,
                 t_vec: Optional[torch.Tensor] = None, b_vec: Optional[torch.Tensor] = None,
                 t_ref_idx: int = 0, use_huber: bool = False, huber_delta: float = 200.0,
                 ignore_invalid_signal: bool = True, signal_min_valid: float = 0.0):
        super().__init__()
        self.forward_model = forward_model
        self.exclude_ref_b0 = exclude_ref_b0
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.ignore_invalid_signal = ignore_invalid_signal
        self.signal_min_valid = float(signal_min_valid)

        # 创建通道 mask（如启用）
        if exclude_ref_b0 and (t_vec is not None) and (b_vec is not None):
            self.register_buffer('channel_mask', self._create_channel_mask(t_vec, b_vec, t_ref_idx))
        else:
            self.channel_mask = None

    def _create_channel_mask(self, t_vec: torch.Tensor, b_vec: torch.Tensor, t_ref_idx: int) -> torch.Tensor:
        """创建通道 mask，排除参考 TE 的 b0 通道（保留旧功能兼容性）。"""
        device = t_vec.device

        channels_per_te = 126
        t_ref = t_vec[t_ref_idx * channels_per_te].item()

        channel_mask = torch.ones(len(t_vec), dtype=torch.bool, device=device)

        for c in range(len(t_vec)):
            t_val = float(t_vec[c].item())
            b_val = float(b_vec[c].item())
            # 容差：|t - t_ref| < 0.1 ms 且 |b| < 0.01 ms/μm²
            if abs(t_val - t_ref) < 0.1 and abs(b_val) < 0.01:
                channel_mask[c] = False

        n_excluded = int((~channel_mask).sum().item())
        n_total = int(len(t_vec))
        print(f"损失函数: 排除 {n_excluded}/{n_total} 个参考TE的b0通道")
        return channel_mask

    def forward(self, theta_global, theta_dir, signal, t_vec, b_vec, P, mask=None):
        """
        参数：
            theta_global: [B, 4, H, W]
            theta_dir:    [B, 30, 6, H, W]
            signal:       [B, 630, H, W]
            t_vec, b_vec, P: 协议向量
            mask:         [B, 1, H, W] 或 [B, C, H, W]
        返回:
            total_loss, loss_dict
        """
        # 1) 物理前向（log 域）
        if hasattr(self.forward_model, 'forward_log'):
            log_S_hat = self.forward_model.forward_log(theta_global, theta_dir, t_vec, b_vec, P)
        else:
            # 兼容无 forward_log 的前向模型：先在原始域合成，再取 log
            with torch.amp.autocast('cuda', enabled=False):
                S_lin = self.forward_model(theta_global, theta_dir, t_vec, b_vec, P)
                S_lin = S_lin.float().clamp_min(1.0)
                log_S_hat = torch.log(S_lin)

        # 2) 观测信号转换到 log 域
        # 使用不低于 signal_min_valid 的下界，避免 log(0) 及极低信号导致的数值不稳定
        eps = max(self.signal_min_valid, 1.0)
        signal_clamped = signal.clamp_min(eps).to(dtype=log_S_hat.dtype)
        log_S = torch.log(signal_clamped)

        # 3) log 域 L1 误差
        err = (log_S_hat - log_S).abs()  # [B,C,H,W]

        # 构造观测有效性掩膜（按通道）
        mask_combined = mask
        if self.ignore_invalid_signal:
            obs_valid = torch.isfinite(signal) & (signal > self.signal_min_valid)
            # 同步通道筛选
            if self.channel_mask is not None:
                obs_valid = obs_valid[:, self.channel_mask, :, :]
            # 与用户 mask 合并
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError("mask 需为4D [B,1,H,W] 或 [B,C,H,W]")
                if mask.shape[1] == 1:
                    mask_exp = mask.expand(-1, obs_valid.shape[1], -1, -1)
                elif mask.shape[1] == obs_valid.shape[1]:
                    mask_exp = mask
                else:
                    raise ValueError("mask 的通道数与误差通道数不匹配")
                mask_combined = (mask_exp > 0.5) & obs_valid
            else:
                mask_combined = obs_valid
            # 最终使用 float 掩膜
            mask_combined = mask_combined.float()

        # 通道排除（参考b0）需在构造 err 后进行，同时保证 mask_combined 也同样裁剪
        if self.channel_mask is not None:
            err = err[:, self.channel_mask, :, :]

        if self.use_huber:
            delta = self.huber_delta
            quad = 0.5 * (err ** 2) / delta
            lin = err - 0.5 * delta
            robust = torch.where(err <= delta, quad, lin)
            l1_loss = _masked_mean(robust, mask_combined if self.ignore_invalid_signal else mask)
        else:
            l1_loss = _masked_mean(err, mask_combined if self.ignore_invalid_signal else mask)
        total_loss = l1_loss
        loss_dict = {
            'total': total_loss,
            'l1': l1_loss,
        }
        return total_loss, loss_dict


class SimpleL1Loss(nn.Module):
    """最简单的 L1 损失（用于消融）"""

    def __init__(self, forward_model, ignore_invalid_signal: bool = True, signal_min_valid: float = 0.0):
        super().__init__()
        self.forward_model = forward_model
        self.ignore_invalid_signal = ignore_invalid_signal
        self.signal_min_valid = float(signal_min_valid)

    def forward(self, theta_global, theta_dir, signal, t_vec, b_vec, P, mask=None):
        # 物理前向
        S_hat = self.forward_model(theta_global, theta_dir, t_vec, b_vec, P)

        # L1（统一 masked mean），加入固定尺度因子 S_scale
        S_scale = 1000.0
        l1_map = torch.abs(S_hat - signal) / S_scale
        mask_combined = mask
        if self.ignore_invalid_signal:
            obs_valid = torch.isfinite(signal) & (signal > self.signal_min_valid)
            if mask is not None:
                if mask.dim() != 4:
                    raise ValueError("mask 需为4D [B,1,H,W] 或 [B,C,H,W]")
                if mask.shape[1] == 1:
                    mask_exp = mask.expand(-1, l1_map.shape[1], -1, -1)
                elif mask.shape[1] == l1_map.shape[1]:
                    mask_exp = mask
                else:
                    raise ValueError("mask 的通道数与误差通道数不匹配")
                mask_combined = (mask_exp > 0.5) & obs_valid
            else:
                mask_combined = obs_valid
            mask_combined = mask_combined.float()
        l1_loss = _masked_mean(l1_map, mask_combined if self.ignore_invalid_signal else mask)

        loss_dict = {'total': l1_loss, 'l1': l1_loss}
        return l1_loss, loss_dict
