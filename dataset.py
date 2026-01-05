"""
数据加载器 - 简化版

基于预处理缓存（npy）的 MRI 数据集与协议向量构建工具
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import cfg


def create_protocol_vectors(grad_csv_path, tes):
    """
    创建协议向量：t_vec, b_vec, P
    
    Args:
        grad_csv_path: 梯度表CSV路径
        tes: TE列表（5个值，单位：毫秒）
    
    Returns:
        t_vec: [630] TE 协议向量；先以毫秒构造，再除以常数 T_SCALE（t_ms / T_SCALE），
               因此在前向模型中作为无量纲变量 t_tilde 使用
        b_vec: [630] b-value 协议向量；先从 s/mm² 转为 ms/μm²，再除以常数 B_SCALE，
               因此在前向模型中作为无量纲变量 b_tilde 使用
        P: [630, 30] 方向映射矩阵
    """
    import pandas as pd
    
    # 读取梯度表
    grad_df = pd.read_csv(grad_csv_path)
    
    # 假设每个TE有126个观测（30个方向×4个b值 + 6个b0）
    n_per_te = 126
    
    t_vec = []
    b_vec = []
    P_rows = []

    T_SCALE = 100.0
    B_SCALE = 2.5
    
    for te_idx, te_val_ms in enumerate(tes):
        # ⭐ 与 code 项目一致：先在物理单位上构造协议，然后对 t、b 做常数缩放
        for subchan_idx in range(n_per_te):
            row = grad_df.iloc[subchan_idx]
            
            # TE: 先使用毫秒，再除以 T_SCALE → 无量纲 t_tilde
            t_vec.append(te_val_ms / T_SCALE)
            # b: 从 s/mm² 转换为 ms/μm²，再除以 B_SCALE → 无量纲 b_tilde
            b_val_ms_um2 = row['b'] / 1000.0  # s/mm² → ms/μm²
            b_vec.append(b_val_ms_um2 / B_SCALE)
            
            # P矩阵：one-hot编码方向
            p_row = np.zeros(30)
            if not row['is_b0']:  # 非b0才有方向
                dir_idx = int(row['dir_idx'])
                p_row[dir_idx] = 1.0
            P_rows.append(p_row)
    
    t_vec = torch.tensor(t_vec, dtype=torch.float32)
    b_vec = torch.tensor(b_vec, dtype=torch.float32)
    P = torch.tensor(np.array(P_rows), dtype=torch.float32)
    
    return t_vec, b_vec, P


class CachedMRIDataset(Dataset):
    """
    读取离线预处理后的缓存：
      - 每个切片的 原始信号 存为 float32 npy: [H, W, 630]
      - 每个切片的多级 mask 存为 npy:
        * voxel_mask_final_XXX.npy: [H, W] 最终训练用体素级 mask
        * temp_mask_630_XXX.npy: [H, W, 630] 观测级 mask
        * dir_mask_30_XXX.npy: [H, W, 30] 方向级 mask
    目录结构（示例）：
      cache_masked/
        train/case001/slice_000.npy, voxel_mask_final_000.npy, temp_mask_630_000.npy, dir_mask_30_000.npy, ...
        val/  /case002/...
    """

    def __init__(self, root_dir, split='train', cache_dir='cache', grad_csv_path=None, tes=None, t_ref_idx=0):
        self.root_dir = Path(root_dir)
        self.split_dir = Path(cache_dir) if Path(cache_dir).is_absolute() else (self.root_dir / cache_dir)
        self.split_dir = self.split_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"缓存目录不存在: {self.split_dir}")

        # 协议向量（与原数据集一致）
        if grad_csv_path is None or tes is None:
            raise ValueError("grad_csv_path 和 tes 必须提供（用于构建协议向量）")
        t_vec, b_vec, P = create_protocol_vectors(grad_csv_path, tes)
        self.t_vec = t_vec
        self.b_vec = b_vec
        self.P = P
        self.t_ref_idx = t_ref_idx
        self.t_ref = tes[t_ref_idx] if tes else None

        # 构建样本列表
        self.samples = []
        for case_dir in sorted([d for d in self.split_dir.iterdir() if d.is_dir()]):
            sig_files = sorted(case_dir.glob('slice_*.npy'))
            for sig_file in sig_files:
                idx = sig_file.stem.split('_')[-1]
                
                # 查找新的多级 mask 文件
                voxel_mask_file = case_dir / f'voxel_mask_final_{idx}.npy'
                temp_mask_file = case_dir / f'temp_mask_630_{idx}.npy'
                dir_mask_file = case_dir / f'dir_mask_30_{idx}.npy'
                
                # 兼容旧 cache（如果新 mask 不存在，尝试使用旧的 mask_XXX.npy）
                if voxel_mask_file.exists():
                    # 新版 cache，使用多级 mask
                    self.samples.append({
                        'signal': sig_file,
                        'mask_voxel': voxel_mask_file,
                        'mask_temp630': temp_mask_file if temp_mask_file.exists() else None,
                        'mask_dir30': dir_mask_file if dir_mask_file.exists() else None,
                        'case_name': case_dir.name,
                        'slice_idx': int(idx)
                    })
                else:
                    # 旧版 cache，使用单一 mask
                    old_mask_file = case_dir / f'mask_{idx}.npy'
                    if old_mask_file.exists():
                        self.samples.append({
                            'signal': sig_file,
                            'mask_voxel': old_mask_file,
                            'mask_temp630': None,
                            'mask_dir30': None,
                            'case_name': case_dir.name,
                            'slice_idx': int(idx)
                        })

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {self.split_dir} 未找到切片缓存（slice_*.npy）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 加载信号（不做任何修改，包括不 clip）
        signal = np.load(item['signal'])  # [H, W, 630] float32
        signal = torch.from_numpy(signal).float().permute(2, 0, 1)  # [630, H, W]
        
        # 加载体素级 mask（最终训练用 mask）
        mask_voxel_np = np.load(item['mask_voxel'])  # [H, W]
        mask_voxel = torch.from_numpy(mask_voxel_np).float().unsqueeze(0)  # [1, H, W]
        
        # 加载观测级 mask（630 通道）
        if item.get('mask_temp630') is not None and item['mask_temp630'].exists():
            temp_mask_np = np.load(item['mask_temp630'])  # [H, W, 630]
            temp_mask = torch.from_numpy(temp_mask_np).float().permute(2, 0, 1)  # [630, H, W]
        else:
            # 如果没有观测级 mask，使用体素级 mask 扩展到 630 通道作为 fallback
            temp_mask = None
        
        # 加载方向级 mask（30 个方向）
        if item.get('mask_dir30') is not None and item['mask_dir30'].exists():
            dir_mask_np = np.load(item['mask_dir30'])  # [H, W, 30]
            dir_mask = torch.from_numpy(dir_mask_np).float().permute(2, 0, 1)  # [30, H, W]
        else:
            dir_mask = None

        out = {
            'signal': signal,                    # [630, H, W]
            'mask': mask_voxel,                 # [1, H, W] - 最终训练用体素级 mask
            'mask_temp630': temp_mask,          # [630, H, W] or None - 观测级 mask
            'mask_dir30': dir_mask,             # [30, H, W] or None - 方向级 mask
            't_vec': self.t_vec,
            'b_vec': self.b_vec,
            'P': self.P,
            'case_name': item.get('case_name', None),
            'slice_idx': item.get('slice_idx', None),
        }
        # Optional: attach LS init if cached npy present in same folder
        try:
            if getattr(cfg, 'use_ls_init', False):
                idx_str = f"{int(item.get('slice_idx', 0)):03d}"
                base = Path(item['signal']).parent
                tg_path = base / f"theta_global_{idx_str}.npy"
                td_path = base / f"theta_dir_{idx_str}.npy"
                if tg_path.exists() and td_path.exists():
                    tg = np.load(tg_path).astype(np.float32)  # [4,H,W]
                    td = np.load(td_path).astype(np.float32)  # [30,6,H,W]
                    out['theta_global_ls'] = torch.from_numpy(tg)
                    out['theta_dir_ls'] = torch.from_numpy(td)
        except Exception as e:
            print(f"[Cached LS init] load failed for {item.get('case_name','?')} slice {item.get('slice_idx','?')}: {e}")
        return out
