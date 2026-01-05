"""
物理前向模型 - 简化版（仅保留核心功能）

从参数 θ 合成 MRI 信号
"""

import torch
import torch.nn as nn


class TaylorForward(nn.Module):
    """
    泰勒展开前向模型（使用缩放后的 TE/b 协议向量，相当于对 t、b 做无量纲化）
    
    理想的物理公式（以物理量表示，t 以毫秒，b 以 ms/μm²）：
    log(S) = ln_s0 - r*t - D*b 
                 + 0.5*(sigma20*t² + 2*sigma11*t*b + sigma02*b²)
                 - 1/6*(sigma30*t³ + 3*sigma21*t²*b + 3*sigma12*t*b² + sigma03*b³)
    
    其中（物理意义）：
    - t: TE值（单位：毫秒）
    - b: b值（单位：ms/μm²）
      - ln_s0: 原始信号的对数幅度
    
    实现细节：
    - 在 dataset.create_protocol_vectors 中，首先对物理量做常数缩放：
        t_tilde = t_ms / T_SCALE,  b_tilde = b_ms_μm2 / B_SCALE
      这相当于对 t、b 做无量纲化以改善数值稳定性；
    - 本前向模型直接使用缩放后的 t_vec, b_vec（即 t_tilde, b_tilde），
      不在此处显式维护 T0/B0，只保留 T0/B0 参数以兼容旧接口。
    """
    
    def __init__(self, T0=None, B0=None, t_ref=None):
        """
        Args:
            T0, B0: 保留参数以兼容旧代码，但不使用（与code项目一致）
            t_ref: 参考TE值（毫秒），保留以兼容旧接口（当前原始域实现不使用）
        """
        super().__init__()
        # 保留但不使用，以兼容旧代码
        if T0 is not None:
            self.register_buffer('T0', torch.tensor(T0))
        if B0 is not None:
            self.register_buffer('B0', torch.tensor(B0))
        # ⭐ 保留参考TE，当前原始域实现不再使用
        if t_ref is not None:
            self.register_buffer('t_ref', torch.tensor(t_ref, dtype=torch.float32))
        else:
            self.t_ref = None
    
    def _compute_log_s(self, theta_global, theta_dir, t_vec, b_vec, P):
        """在 log 域计算合成信号 log(S_hat)。"""
        # 始终在 float32 中执行物理前向，避免混合精度导致的溢出
        theta_global = theta_global.float()
        theta_dir = theta_dir.float()
        t_vec = t_vec.float()
        b_vec = b_vec.float()
        P = P.float()

        # 提取全局参数
        ln_s0 = theta_global[:, 0:1, :, :]    # [B, 1, H, W]
        r = theta_global[:, 1:2, :, :]
        sigma20 = theta_global[:, 2:3, :, :]
        sigma30 = theta_global[:, 3:4, :, :]

        # 提取方向参数
        D = theta_dir[:, :, 0, :, :]          # [B, 30, H, W]
        sigma11 = theta_dir[:, :, 1, :, :]
        sigma02 = theta_dir[:, :, 2, :, :]
        sigma21 = theta_dir[:, :, 3, :, :]
        sigma12 = theta_dir[:, :, 4, :, :]
        sigma03 = theta_dir[:, :, 5, :, :]

        # t_vec / b_vec 已在 dataset 中按 T_SCALE / B_SCALE 缩放，相当于无量纲变量 t_tilde / b_tilde
        t = t_vec.view(1, -1, 1, 1)  # [1, 630, 1, 1] - 缩放后的 TE（无量纲 t_tilde）
        b = b_vec.view(1, -1, 1, 1)  # [1, 630, 1, 1] - 缩放后的 b 值（无量纲 b_tilde）

        # 通过 P 矩阵将 30 方向参数映射到 630 通道
        D_ch = torch.einsum('cd,bdhw->bchw', P, D)           # [B, 630, H, W]
        sigma11_ch = torch.einsum('cd,bdhw->bchw', P, sigma11)
        sigma02_ch = torch.einsum('cd,bdhw->bchw', P, sigma02)
        sigma21_ch = torch.einsum('cd,bdhw->bchw', P, sigma21)
        sigma12_ch = torch.einsum('cd,bdhw->bchw', P, sigma12)
        sigma03_ch = torch.einsum('cd,bdhw->bchw', P, sigma03)

        # 计算 log(S_hat)（原始信号域）
        # 零阶项
        log_s = ln_s0

        # 一阶项: -r * t - D * b
        log_s = log_s - r * t - D_ch * b

        # 二阶项: +0.5*(sigma20*t² + 2*sigma11*t*b + sigma02*b²)
        t2 = t ** 2
        b2 = b ** 2
        tb = t * b
        second_order = sigma20 * t2 + 2.0 * sigma11_ch * tb + sigma02_ch * b2
        log_s = log_s + 0.5 * second_order

        # 三阶项: -1/6*(...)
        t3 = t ** 3
        b3 = b ** 3
        t2b = t2 * b
        tb2 = t * b2
        third_order = (sigma30 * t3 + 
                      3.0 * sigma21_ch * t2b + 
                      3.0 * sigma12_ch * tb2 + 
                      sigma03_ch * b3)
        log_s = log_s - (1.0/6.0) * third_order

        return log_s

    def forward_log(self, theta_global, theta_dir, t_vec, b_vec, P):
        """返回合成信号的 log(S_hat) [B, 630, H, W]。"""
        with torch.amp.autocast('cuda', enabled=False):
            log_s = self._compute_log_s(theta_global, theta_dir, t_vec, b_vec, P)
        return log_s

    def forward(self, theta_global, theta_dir, t_vec, b_vec, P):
        """返回合成信号 S_hat [B, 630, H, W]（原始域）。"""
        with torch.amp.autocast('cuda', enabled=False):
            log_s = self._compute_log_s(theta_global, theta_dir, t_vec, b_vec, P)
            S_hat = torch.exp(log_s)
        return S_hat

