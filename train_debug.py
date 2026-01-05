"""
Debug training script for rdmri_unet_simple.

Goals:
- Train only a few epochs/batches
- Print detailed loss composition and parameter stats
- Focus on a specific layer (default: the 7th top-level module in UNet)

Run: python rdmri_unet_simple/train_debug.py
"""

from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from models import UNetWithHeads
from dataset import CachedMRIDataset
from physics import TaylorForward
from loss import L1PhysicsLoss


# Debug controls (no CLI required)
DEBUG_EPOCHS = 3
MAX_TRAIN_BATCHES = 2
MAX_VAL_BATCHES = 1
TARGET_LAYER_INDEX = 7  # 1-based index among top-level modules in UNet
OUT_DIR = Path("debug_out")


def select_gpu() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    gid = getattr(cfg, "gpu_id", 0) or 0
    gid = min(gid, torch.cuda.device_count() - 1)
    torch.cuda.set_device(gid)
    print(f"Using GPU {gid}: {torch.cuda.get_device_name(gid)}")
    return torch.device(f"cuda:{gid}")


def tensor_stats(x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, float]:
    with torch.no_grad():
        if mask is not None:
            x = x[mask]
        return {
            "min": float(x.min().item()),
            "p05": float(x.quantile(0.05).item()) if x.numel() > 10 else float(x.min().item()),
            "mean": float(x.mean().item()),
            "p95": float(x.quantile(0.95).item()) if x.numel() > 10 else float(x.max().item()),
            "max": float(x.max().item()),
            "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            "numel": int(x.numel()),
        }


def masked_stats_4d(x: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Compute stats over masked elements for 4D tensors [B, C, H, W] (or [B,1,H,W])."""
    if mask is not None and mask.ndim == 4:
        if x.ndim == 4 and mask.shape[1] == 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        sel = mask > 0.5
        vals = x[sel]
    else:
        vals = x.reshape(-1)
    if vals.numel() == 0:
        return {"min": 0.0, "p05": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0, "numel": 0}
    return {
        "min": float(vals.min().item()),
        "p05": float(vals.quantile(0.05).item()) if vals.numel() > 10 else float(vals.min().item()),
        "mean": float(vals.mean().item()),
        "p95": float(vals.quantile(0.95).item()) if vals.numel() > 10 else float(vals.max().item()),
        "max": float(vals.max().item()),
        "std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "numel": int(vals.numel()),
    }


def masked_mean_l1(S_hat: torch.Tensor, signal: torch.Tensor, mask: torch.Tensor, channel_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # S_hat, signal: [B, 630, H, W]
    l1_map = torch.abs(S_hat - signal)
    if channel_mask is not None:
        l1_map = l1_map[:, channel_mask, :, :]
    if mask is None:
        num = l1_map.sum()
        den = torch.tensor(l1_map.numel(), dtype=l1_map.dtype, device=l1_map.device)
        return num / den, num, den
    # expand mask to channel dim
    if mask.shape[1] == 1:
        mask_exp = mask.expand(-1, l1_map.shape[1], -1, -1)
    else:
        mask_exp = mask
    num = (l1_map * mask_exp).sum()
    den = mask_exp.sum().clamp_min(1e-8)
    return num / den, num, den


def get_unet_top_modules(unet: nn.Module) -> List[Tuple[str, nn.Module]]:
    # Ordered as defined in SimpleUNet2D.__init__ (uses _modules order)
    return list(unet._modules.items())


@torch.no_grad()
def physics_breakdown(theta_global: torch.Tensor,
                      theta_dir: torch.Tensor,
                      t_vec: torch.Tensor,
                      b_vec: torch.Tensor,
                      P: torch.Tensor,
                      mask: torch.Tensor = None) -> Dict[str, float]:
    """
    Recompute physics terms to report clamp rate and term means.
    Returns dict with means of each term within mask and clamp rate.
    """
    B, C, H, W = theta_dir.shape[0], 630, theta_dir.shape[-2], theta_dir.shape[-1]
    # Globals
    ln_s0 = theta_global[:, 0:1]
    r = theta_global[:, 1:2]
    sigma20 = theta_global[:, 2:3]
    sigma30 = theta_global[:, 3:4]

    # Directions
    D = theta_dir[:, :, 0, :, :]
    sigma11 = theta_dir[:, :, 1, :, :]
    sigma02 = theta_dir[:, :, 2, :, :]
    sigma21 = theta_dir[:, :, 3, :, :]
    sigma12 = theta_dir[:, :, 4, :, :]
    sigma03 = theta_dir[:, :, 5, :, :]

    # Vectors
    t = t_vec.view(1, -1, 1, 1)
    b = b_vec.view(1, -1, 1, 1)
    # Map 30->630
    D_ch = torch.einsum('cd,bdhw->bchw', P, D)
    sigma11_ch = torch.einsum('cd,bdhw->bchw', P, sigma11)
    sigma02_ch = torch.einsum('cd,bdhw->bchw', P, sigma02)
    sigma21_ch = torch.einsum('cd,bdhw->bchw', P, sigma21)
    sigma12_ch = torch.einsum('cd,bdhw->bchw', P, sigma12)
    sigma03_ch = torch.einsum('cd,bdhw->bchw', P, sigma03)

    # Terms
    term0 = ln_s0.expand(-1, C, -1, -1)
    term_r = -r.expand(-1, C, -1, -1) * t
    term_D = -D_ch * b

    t2 = t ** 2
    b2 = b ** 2
    tb = t * b
    term2 = 0.5 * (sigma20.expand(-1, C, -1, -1) * t2 + 2.0 * sigma11_ch * tb + sigma02_ch * b2)

    t3 = t ** 3
    b3 = b ** 3
    t2b = t2 * b
    tb2 = t * b2
    term3 = -(1.0 / 6.0) * (
        sigma30.expand(-1, C, -1, -1) * t3 + 3.0 * sigma21_ch * t2b + 3.0 * sigma12_ch * tb2 + sigma03_ch * b3
    )

    log_raw = term0 + term_r + term_D + term2 + term3
    log_clamped = torch.clamp(log_raw, -15.0, 15.0)

    # mask select
    if mask is not None and mask.ndim == 4:
        mexp = mask.expand(-1, C, -1, -1) if mask.shape[1] == 1 else mask
        sel = mexp > 0.5
    else:
        sel = torch.ones_like(log_raw, dtype=torch.bool)

    total = float(sel.sum().item()) if sel.numel() > 0 else 1.0
    clamp_hits = ((log_raw <= -15.0) | (log_raw >= 15.0)) & sel
    clamp_rate = float(clamp_hits.sum().item()) / total

    # Means within mask
    def mmean(t):
        return float(t[sel].mean().item()) if sel.any() else 0.0

    return {
        "clamp_rate": clamp_rate,
        "term0": mmean(term0),
        "term_r": mmean(term_r),
        "term_D": mmean(term_D),
        "term2": mmean(term2),
        "term3": mmean(term3),
        "log_raw": mmean(log_raw),
        "log_clamped": mmean(log_clamped),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Debug Training - small run with instrumentation")
    print("=" * 60)

    device = select_gpu()

    # 1) Dataset（仅使用 CachedMRIDataset 预处理缓存）
    print("\nLoading datasets (small debug run, CachedMRIDataset)...")
    grad_csv = Path(cfg.data_root) / "grad_126.csv"
    train_dataset = CachedMRIDataset(
        cfg.data_root, split=cfg.train_dir,
        cache_dir=cfg.cache_dir,
        grad_csv_path=grad_csv, tes=cfg.tes,
        t_ref_idx=getattr(cfg, "t_ref_idx", 0),
    )
    val_dataset = CachedMRIDataset(
        cfg.data_root, split=cfg.val_dir,
        cache_dir=cfg.cache_dir,
        grad_csv_path=grad_csv, tes=cfg.tes,
        t_ref_idx=getattr(cfg, "t_ref_idx", 0),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=min(cfg.batch_size, 2),
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, prefetch_factor=getattr(cfg, 'prefetch_factor', 2),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(cfg.batch_size, 2),
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, prefetch_factor=getattr(cfg, 'prefetch_factor', 2),
        persistent_workers=True,
    )

    # 2) Protocol vectors
    print("\nPreparing protocol vectors...")
    t_vec = train_dataset.t_vec.to(device)
    b_vec = train_dataset.b_vec.to(device)
    P = train_dataset.P.to(device)
    print(f"t_vec {tuple(t_vec.shape)}, b_vec {tuple(b_vec.shape)}, P {tuple(P.shape)}")

    # 3) Model and physics
    print("\nCreating model + physics...")
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
    ).to(device)

    # load checkpoint if exists (best)
    # Prefer a specific checkpoint if present, else fall back to best_model.pt
    ckpt_path = Path(cfg.save_dir) / 'checkpoint_epoch100.pt'
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.save_dir) / 'best_model.pt'
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Locate target layer
    top_modules = get_unet_top_modules(model.unet)
    if TARGET_LAYER_INDEX <= 0 or TARGET_LAYER_INDEX > len(top_modules):
        print(f"Target layer index {TARGET_LAYER_INDEX} out of range 1..{len(top_modules)}; listing:")
        for i, (n, m) in enumerate(top_modules, start=1):
            print(f"  {i:2d}: {n} -> {m.__class__.__name__}")
        return
    target_name, target_module = top_modules[TARGET_LAYER_INDEX - 1]
    print(f"\nTarget layer #{TARGET_LAYER_INDEX}: {target_name} ({target_module.__class__.__name__})")

    # Training small run
    for epoch in range(1, DEBUG_EPOCHS + 1):
        print("\n" + "-" * 60)
        print(f"Epoch {epoch}/{DEBUG_EPOCHS}")

        model.train()
        train_iter = iter(train_loader)
        train_batches = 0
        for _ in range(MAX_TRAIN_BATCHES):
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            train_batches += 1

            signal = batch['signal'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            # Mask info
            with torch.no_grad():
                mask_fg = mask.sum(dim=(1, 2, 3))
                total_px = torch.tensor(mask.shape[2] * mask.shape[3], device=mask.device)
                fg_ratio = (mask_fg / total_px).detach().cpu().numpy()
                print(f"  Batch{train_batches}: mask foreground ratio per sample: {fg_ratio}")

            # Snapshot target params for update ratio
            prev_params = [p.detach().clone() for p in target_module.parameters() if p.requires_grad]

            theta_global, theta_dir = model(signal)
            loss, loss_dict = criterion(theta_global, theta_dir, signal, t_vec, b_vec, P, mask)

            # Compute S_hat and detailed loss numbers (no grad)
            with torch.no_grad():
                S_hat = forward_model(theta_global, theta_dir, t_vec, b_vec, P)
                l1_now, num, den = masked_mean_l1(S_hat, signal, mask, getattr(criterion, 'channel_mask', None))
                print(f"    L1(batch)={l1_now.item():.6f} | num={num.item():.2e} den={den.item():.2e}")

                # Per-TE L1 (5 groups of 126)
                C = S_hat.shape[1]
                ch_per_te = 126
                te_losses = []
                for gi in range(C // ch_per_te):
                    idx = slice(gi * ch_per_te, (gi + 1) * ch_per_te)
                    l1_g, _, _ = masked_mean_l1(S_hat[:, idx], signal[:, idx], mask)
                    te_losses.append(float(l1_g.item()))
                print(f"    Per-TE L1: {[f'{x:.4f}' for x in te_losses]}")

                # Physics breakdown: clamp rate and term means
                pb = physics_breakdown(theta_global, theta_dir, t_vec, b_vec, P, mask)
                print(
                    "    Physics: clamp_rate={:.2f}% | terms(mean) zero={:.3f} r={:.3f} D={:.3f} 2nd={:.3f} 3rd={:.3f} log={:.3f}".format(
                        pb["clamp_rate"] * 100.0, pb["term0"], pb["term_r"], pb["term_D"], pb["term2"], pb["term3"], pb["log_raw"]
                    )
                )

                # Parameter stats (masked)
                ln_s0 = theta_global[:, 0:1]
                rpar = theta_global[:, 1:2]
                D = theta_dir[:, :, 0, :, :].mean(dim=1, keepdim=True)  # mean over 30 dirs
                s_lns0 = masked_stats_4d(ln_s0, mask)
                s_r = masked_stats_4d(rpar, mask)
                s_D = masked_stats_4d(D, mask)
                print(
                    f"    Params: ln_s0(mean={s_lns0['mean']:.3f}, p05={s_lns0['p05']:.3f}, p95={s_lns0['p95']:.3f}) | "
                    f"r(mean={s_r['mean']:.3f}, p05={s_r['p05']:.3f}, p95={s_r['p95']:.3f}) | "
                    f"D_mean(mean={s_D['mean']:.3f}, p05={s_D['p05']:.3f}, p95={s_D['p95']:.3f})"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Print target layer grad/weight stats before step
            for j, p in enumerate([p for p in target_module.parameters() if p.requires_grad]):
                gnorm = p.grad.norm().item() if p.grad is not None else 0.0
                pstat = tensor_stats(p.detach())
                print(f"    [{target_name} param{j}] grad_norm={gnorm:.3e} weight_mean={pstat['mean']:.3e} std={pstat['std']:.3e} min={pstat['min']:.3e} max={pstat['max']:.3e}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            # Update ratio for target layer
            with torch.no_grad():
                for j, (p, prev) in enumerate(zip([p for p in target_module.parameters() if p.requires_grad], prev_params)):
                    delta = (p - prev)
                    ratio = (delta.norm() / (p.norm() + 1e-12)).item()
                    print(f"    [{target_name} param{j}] update_ratio={ratio:.3e}")

        # Validation small run
        model.eval()
        val_iter = iter(val_loader)
        val_batches = 0
        with torch.no_grad():
            for _ in range(MAX_VAL_BATCHES):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break
                val_batches += 1
                signal = batch['signal'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                theta_global, theta_dir = model(signal)
                S_hat = forward_model(theta_global, theta_dir, t_vec, b_vec, P)
                l1_now, num, den = masked_mean_l1(S_hat, signal, mask, getattr(criterion, 'channel_mask', None))
                print(f"  [Val batch] L1={l1_now.item():.6f} | num={num.item():.2e} den={den.item():.2e}")

    print("\nDebug run finished.")


if __name__ == "__main__":
    main()
