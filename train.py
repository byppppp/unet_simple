"""
训练脚本 - 简化版（支持缓存数据集）

L1 损失的自监督训练
"""

import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp
from tqdm import tqdm

from config import cfg
from models import UNetWithHeads
from physics import TaylorForward
from loss import L1PhysicsLoss
from dataset import CachedMRIDataset


def select_gpu():
    """自动选择可用 GPU"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，使用 CPU")
        return torch.device('cpu')

    # 如果指定了 GPU
    if hasattr(cfg, 'gpu_id') and cfg.gpu_id is not None:
        gpu_id = cfg.gpu_id
        n_gpus = torch.cuda.device_count()
        if gpu_id >= n_gpus:
            print(f"⚠️ GPU {gpu_id} 不存在（系统只有 {n_gpus} 个），自动选择可用 GPU")
        else:
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✓ 使用指定 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            torch.cuda.set_device(gpu_id)
            return device

    # 自动选择显存最多的 GPU
    n_gpus = torch.cuda.device_count()
    print(f"\n检测到 {n_gpus} 个 GPU，自动选择显存最多的...")
    max_free_memory = 0
    best_gpu = 0
    for i in range(n_gpus):
        try:
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - reserved_memory
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} 可用: {free_memory/1024**3:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        except Exception as e:
            print(f"  ⚠️ 无法获取 GPU {i} 信息: {e}")
            continue
    device = torch.device(f'cuda:{best_gpu}')
    print(f"\n✓ 自动选择 GPU {best_gpu}")
    torch.cuda.set_device(best_gpu)
    return device


def train_epoch(model, criterion, loader, optimizer, device, t_vec, b_vec, P, epoch, scaler=None, use_amp=False):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_l1 = torch.tensor(0.0, device=device)

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        signal = batch['signal'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with amp.autocast('cuda'):
                theta_global, theta_dir = model(signal)
                loss, loss_dict = criterion(
                    theta_global, theta_dir, signal,
                    t_vec, b_vec, P, mask
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            theta_global, theta_dir = model(signal)
            loss, loss_dict = criterion(
                theta_global, theta_dir, signal,
                t_vec, b_vec, P, mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict['total']
        total_l1 += loss_dict['l1']

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'l1': f"{loss_dict['l1'].item():.4f}"
            })

    n_batches = len(loader)
    return {
        'loss': (total_loss / n_batches).item(),
        'l1': (total_l1 / n_batches).item(),
    }


def validate(model, criterion, loader, device, t_vec, b_vec, P, use_amp=False):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_l1 = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            signal = batch['signal'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            if use_amp:
                with amp.autocast('cuda'):
                    theta_global, theta_dir = model(signal)
                    loss, loss_dict = criterion(
                        theta_global, theta_dir, signal,
                        t_vec, b_vec, P, mask
                    )
            else:
                theta_global, theta_dir = model(signal)
                loss, loss_dict = criterion(
                    theta_global, theta_dir, signal,
                    t_vec, b_vec, P, mask
                )
            total_loss += loss_dict['total']
            total_l1 += loss_dict['l1']
    n_batches = len(loader)
    return {
        'loss': (total_loss / n_batches).item(),
        'l1': (total_l1 / n_batches).item(),
    }


def main():
    print("=" * 60)
    print("RDMRI UNet Simple - L1 自监督训练")
    print("=" * 60)

    device = select_gpu()
    print(f"使用设备: {device}")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir='runs/simple_l1')

    # 1. 数据集（仅使用预处理缓存 CachedMRIDataset）
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

    # 2. 协议向量（数据集缓存了一份，直接搬到 device）
    print("\n获取协议向量...")
    t_vec = train_dataset.t_vec.to(device, non_blocking=True)
    b_vec = train_dataset.b_vec.to(device, non_blocking=True)
    P = train_dataset.P.to(device, non_blocking=True)
    print(f"✓ 协议向量: t_vec {t_vec.shape}, b_vec {b_vec.shape}, P {P.shape}")

    # 3. 模型
    print("\n创建模型...")
    model = UNetWithHeads(
        in_channels=cfg.in_channels,
        mid_channels=cfg.mid_channels,
        n_dirs=cfg.directions,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 4. 前向与损失（原始信号域）
    forward_model = TaylorForward().to(device)

    criterion = L1PhysicsLoss(
        forward_model=forward_model,
        exclude_ref_b0=getattr(cfg, 'exclude_ref_b0_in_loss', True),
        t_vec=t_vec,
        b_vec=b_vec,
        t_ref_idx=getattr(cfg, 't_ref_idx', 0),
        ignore_invalid_signal=getattr(cfg, 'ignore_invalid_signal', True),
        signal_min_valid=getattr(cfg, 'signal_min_valid', 0.0),
    ).to(device)
    print("损失函数: L1")
    if getattr(cfg, 'exclude_ref_b0_in_loss', True):
        print("  ✓ 排除参考TE的 b0 通道（兼容原归一化流程）")

    # 5. 优化器
    # 参数组设置，降低 heads 最后一层的学习率
    param_groups = [
        {"params": model.unet.parameters(), "lr": cfg.lr},
        {"params": model.heads.global_pre.parameters(), "lr": cfg.lr},
        {"params": model.heads.dir_pre.parameters(), "lr": cfg.lr},
        {"params": model.heads.head_lns0.parameters(), "lr": cfg.lr},
        {"params": model.heads.head_r.parameters(), "lr": getattr(cfg, 'lr_r', cfg.lr)},
        {"params": model.heads.head_sigma20.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.head_sigma30.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.dir_D.parameters(), "lr": cfg.lr},
        {"params": model.heads.dir_sigma11.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.dir_sigma02.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.dir_sigma21.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.dir_sigma12.parameters(), "lr": cfg.lr_high_order},
        {"params": model.heads.dir_sigma03.parameters(), "lr": cfg.lr_high_order},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    # 6. 训练循环
    print("\n开始训练..")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Batch size: {cfg.batch_size}")
    print("=" * 60)

    best_val_loss = float('inf')
    scaler = amp.GradScaler('cuda', enabled=getattr(cfg, 'use_amp', True))
    train_start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start_time = time.time()
        train_metrics = train_epoch(
            model, criterion, train_loader, optimizer,
            device, t_vec, b_vec, P, epoch,
            scaler=scaler if getattr(cfg, 'use_amp', False) else None,
            use_amp=getattr(cfg, 'use_amp', False),
        )
        val_metrics = validate(
            model, criterion, val_loader,
            device, t_vec, b_vec, P,
            use_amp=getattr(cfg, 'use_amp', False),
        )

        print(f"\nEpoch {epoch}/{cfg.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, L1: {train_metrics['l1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, L1: {val_metrics['l1']:.4f}")

        epoch_elapsed = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        epoch_min, epoch_sec = divmod(int(epoch_elapsed), 60)
        total_min, total_sec = divmod(int(total_elapsed), 60)
        print(f"  Time - This epoch: {epoch_min}m {epoch_sec}s | Total: {total_min}m {total_sec}s")

        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('L1/train', train_metrics['l1'], epoch)
        writer.add_scalar('L1/val', val_metrics['l1'], epoch)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }, save_dir / 'best_model.pt')
            print(f"  ✓ 保存最佳模型 (val_loss: {best_val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'checkpoint_epoch{epoch}.pt')

    print("\n训练完成")
    writer.close()


if __name__ == '__main__':
    main()
