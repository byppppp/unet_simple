"""
Inference script for rdmri_unet_simple

- Loads a trained UNetWithHeads checkpoint
- Runs inference on test/ (or val/) set
- Reconstructs signals with the TaylorForward physics model
- Computes simple metrics and saves parameter maps
"""

import os
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from config import cfg
from models import UNetWithHeads
from dataset import CachedMRIDataset
from physics import TaylorForward
from train_A import apply_curriculum
from metrics import (
    evaluate_signal_reconstruction,
    print_evaluation_metrics,
    compute_parameter_statistics,
    print_statistics,
)

T_SCALE = 100.0
B_SCALE = 2.5

def select_gpu() -> torch.device:
    """Pick a CUDA device if available, otherwise CPU.

    - If cfg.gpu_id is set and valid, use it
    - Else pick the GPU with the most free memory
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")

    if cfg.gpu_id is not None and cfg.gpu_id < torch.cuda.device_count():
        dev = torch.device(f"cuda:{cfg.gpu_id}")
        print(f"Using specified GPU {cfg.gpu_id}: {torch.cuda.get_device_name(cfg.gpu_id)}")
        torch.cuda.set_device(cfg.gpu_id)
        return dev

    # auto-pick by free memory (approx)
    n_gpus = torch.cuda.device_count()
    best_i, best_free = 0, -1
    for i in range(n_gpus):
        total = torch.cuda.get_device_properties(i).total_memory
        reserved = torch.cuda.memory_reserved(i)
        free = total - reserved
        if free > best_free:
            best_free = free
            best_i = i
    torch.cuda.set_device(best_i)
    print(f"Auto-selected GPU {best_i}: {torch.cuda.get_device_name(best_i)}")
    return torch.device(f"cuda:{best_i}")


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model to the requested device (fall back to CPU on OOM)."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = UNetWithHeads(
        in_channels=cfg.in_channels,
        mid_channels=cfg.mid_channels,
        n_dirs=cfg.directions,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", None)
    if epoch is not None:
        print(f"  Epoch: {epoch}")

    # 兼容不同训练脚本保存的验证指标字段名
    val_metric = ckpt.get("val_l1_numden", ckpt.get("val_loss", None))
    if val_metric is not None:
        print(f"  Val loss: {val_metric:.6f}")

    try:
        model = model.to(device)
        print(f"  Model device: {device}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU OOM while moving model. Falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
    return model, device, epoch


def normalize_signal_for_unet(signal: torch.Tensor) -> torch.Tensor:
    """Log+scale normalization for UNet input only.

    Physics model and saved outputs still use original signal/parameters.
    """
    import math

    x = signal.clamp(min=1.0)
    x = torch.log(x)
    log_min = math.log(100.0)
    log_max = math.log(20000.0)
    scale = (log_max - log_min)
    x = (x - log_min) / scale
    return x.clamp(0.0, 1.0)


@torch.inference_mode()
def predict_single_case(model: torch.nn.Module, signal: torch.Tensor, mask: torch.Tensor, device: torch.device):
    """Forward one slice.

    Inputs:
        signal: [630, H, W]
        mask:   [1, H, W]
    Returns:
        theta_global: [4, H, W]
        theta_dir:    [30, 6, H, W]
    """
    signal_b = signal.unsqueeze(0).to(device)
    signal_net = normalize_signal_for_unet(signal_b)
    theta_global, theta_dir = model(signal_net)
    theta_global = theta_global.squeeze(0).cpu()
    theta_dir = theta_dir.squeeze(0).cpu()

    if mask is not None:
        m = mask.cpu()
        theta_global = theta_global * m
        theta_dir = theta_dir * m.unsqueeze(1)  # [1,1,H,W] broadcasting over [30,6,H,W]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return theta_global, theta_dir


def _load_case_shape_and_affine(case_dir: Path):
    """Read first TE of a case to get (H, W, Z) and affine."""
    # prefer uncompressed .nii when available
    te1 = cfg.te_files[0] if getattr(cfg, "te_files", None) else "te1.nii.gz"
    te1_path = case_dir / te1
    if not te1_path.exists():
        te1_path = case_dir / te1.replace('.nii.gz', '.nii')
        if not te1_path.exists():
            te1_path = case_dir / 'te1.nii'
            if not te1_path.exists():
                te1_path = case_dir / 'te1.nii.gz'
    img = nib.load(str(te1_path))
    H, W, Z, _ = img.shape
    return (H, W, Z), img.affine


def _init_case_buffers(H: int, W: int, Z: int):
    """Allocate numpy buffers for one case.

    Returns dicts: globals3d, dirs4d
    globals3d keys: 's0','r','sigma20','sigma30' -> [H,W,Z]
    dirs4d keys: 'D','sigma11','sigma02','sigma21','sigma12','sigma03' -> [H,W,Z,31]
    """
    g3d = {
        's0': np.zeros((H, W, Z), dtype=np.float32),
        'r': np.zeros((H, W, Z), dtype=np.float32),
        'sigma20': np.zeros((H, W, Z), dtype=np.float32),
        'sigma30': np.zeros((H, W, Z), dtype=np.float32),
    }
    d4d = {name: np.zeros((H, W, Z, 31), dtype=np.float32)
           for name in ["D", "sigma11", "sigma02", "sigma21", "sigma12", "sigma03"]}
    return g3d, d4d


def _save_case_volumes(case_dir: Path, affine, globals3d: dict, dirs4d: dict):
    case_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(globals3d['s0'], affine), case_dir / 's0.nii.gz')
    nib.save(nib.Nifti1Image(globals3d['r'], affine), case_dir / 'r.nii.gz')
    nib.save(nib.Nifti1Image(globals3d['sigma20'], affine), case_dir / 'sigma20.nii.gz')
    nib.save(nib.Nifti1Image(globals3d['sigma30'], affine), case_dir / 'sigma30.nii.gz')
    # direction 4D outputs only (vol0 = mean, vol1..30 = per-direction)
    for name, arr4d in dirs4d.items():
        nib.save(nib.Nifti1Image(arr4d, affine), case_dir / f'{name}.nii.gz')


def main():
    print("=" * 60)
    print("Predict Script (rdmri_unet_simple)")
    print("=" * 60)

    ckpt = Path(cfg.save_dir) / "best_model_A.pt"
    out_dir = Path("predict_out")
    out_dir.mkdir(exist_ok=True)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return

    # device + model
    device = select_gpu()
    model, device, epoch_trained = load_model(ckpt, device)

    # physics model
    try:
        forward_model = TaylorForward().to(device)
        forward_device = device  # ensure physics runs on the selected device (GPU preferred)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU OOM for physics model, using CPU")
            forward_model = TaylorForward().to("cpu")
            forward_device = torch.device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    # dataset（仅使用 CachedMRIDataset 预处理缓存进行推理）
    print("\nLoading dataset (CachedMRIDataset)...")
    test_dir = Path(cfg.data_root) / "test"
    split_name = "test"
    if not test_dir.exists():
        print(f"Test dir not found: {test_dir}; use val/ instead")
        test_dir = Path(cfg.data_root) / cfg.val_dir
        split_name = cfg.val_dir

    grad_csv = Path(cfg.data_root) / "grad_126.csv"
    print("Using CachedMRIDataset (npy cache) for inference")
    ds = CachedMRIDataset(
        cfg.data_root,
        split=split_name,
        cache_dir=cfg.cache_dir,
        grad_csv_path=grad_csv,
        tes=cfg.tes,
        t_ref_idx=getattr(cfg, "t_ref_idx", 0),
    )
    print(f"Samples: {len(ds)}")

    # protocol vectors on physics device
    t_vec = ds.t_vec.to(forward_device)
    b_vec = ds.b_vec.to(forward_device)
    P = ds.P.to(forward_device)

    # run (group by case, assemble 3D/4D volumes)
    all_metrics = []
    all_param_stats = []
    current_case = None
    case_buffers = None
    case_affine = None
    case_used_slices = []
    case_metrics = []
    case_param_stats = []
    H = W = Z = None
    # 若 checkpoint 中未包含 epoch 信息，则回退为 cfg.epochs
    epoch_for_curriculum = epoch_trained if epoch_trained is not None else cfg.epochs
    for i in tqdm(range(len(ds)), desc="inference"):
        sample = ds[i]
        signal = sample["signal"]  # [630, H, W]
        mask = sample["mask"]      # [1, H, W]
        case_name = sample.get("case_name", sample.get("case_id", f"case_{i:03d}"))
        slice_idx = int(sample.get("slice_idx", i))
        affine = sample.get("affine", None)
        # normalization removed: no S0_ref available

        # new case: finalize previous and init buffers for this one
        if current_case != case_name:
            # save previous case volumes
            if current_case is not None and case_buffers is not None and case_affine is not None:
                out_case_dir = out_dir / current_case
                _save_case_volumes(out_case_dir, case_affine, case_buffers[0], case_buffers[1])
            # init new
            current_case = case_name
            case_dir = Path(cfg.data_root) / split_name / case_name
            (H, W, Z), case_affine = _load_case_shape_and_affine(case_dir)
            case_buffers = _init_case_buffers(H, W, Z)
            case_used_slices = []
            case_metrics = []
            case_param_stats = []

        print("\n" + "-" * 60)
        print(f"Case: {case_name}  Slice: {slice_idx}  Signal: {tuple(signal.shape)}")

        theta_global, theta_dir = predict_single_case(model, signal, mask, device)

        # 使用与训练阶段一致的 curriculum/first_order 处理后再重建
        with torch.inference_mode():
            tg = theta_global.unsqueeze(0).to(forward_device)
            td = theta_dir.unsqueeze(0).to(forward_device)
            tg, td, s2, s3 = apply_curriculum(tg, td, epoch_for_curriculum)
            recon = forward_model(tg, td, t_vec, b_vec, P).squeeze(0).cpu()
            # 后续参数统计与保存也基于经过 curriculum 处理后的参数
            theta_global_eff = tg.squeeze(0).cpu()   # [4,H,W]
            theta_dir_eff = td.squeeze(0).cpu()      # [30,6,H,W]

        m = evaluate_signal_reconstruction(recon, signal, mask)
        all_metrics.append(m)
        case_metrics.append(m)
        print_evaluation_metrics(m, title=f"Reconstruction quality - {case_name}_slice{slice_idx:03d}")

        # parameter stats (convert model-space parameters to physical units)
        ln_s0 = theta_global_eff[0]
        r_model = theta_global_eff[1]
        sigma20_model = theta_global_eff[2]
        sigma30_model = theta_global_eff[3]

        s0 = torch.exp(ln_s0)
        r_phys = r_model / T_SCALE
        sigma20_phys = sigma20_model / (T_SCALE * T_SCALE)
        sigma30_phys = sigma30_model / (T_SCALE * T_SCALE * T_SCALE)

        D_model = theta_dir_eff[:, 0, :, :]
        sigma11_model = theta_dir_eff[:, 1, :, :]
        sigma02_model = theta_dir_eff[:, 2, :, :]
        sigma21_model = theta_dir_eff[:, 3, :, :]
        sigma12_model = theta_dir_eff[:, 4, :, :]
        sigma03_model = theta_dir_eff[:, 5, :, :]

        D_phys = D_model / B_SCALE
        sigma11_phys = sigma11_model / (T_SCALE * B_SCALE)
        sigma02_phys = sigma02_model / (B_SCALE * B_SCALE)
        sigma21_phys = sigma21_model / (T_SCALE * T_SCALE * B_SCALE)
        sigma12_phys = sigma12_model / (T_SCALE * B_SCALE * B_SCALE)
        sigma03_phys = sigma03_model / (B_SCALE * B_SCALE * B_SCALE)

        theta_dir_phys = torch.empty_like(theta_dir_eff)
        theta_dir_phys[:, 0, :, :] = D_phys
        theta_dir_phys[:, 1, :, :] = sigma11_phys
        theta_dir_phys[:, 2, :, :] = sigma02_phys
        theta_dir_phys[:, 3, :, :] = sigma21_phys
        theta_dir_phys[:, 4, :, :] = sigma12_phys
        theta_dir_phys[:, 5, :, :] = sigma03_phys

        stats_s0 = compute_parameter_statistics(s0, mask, "s0")
        stats_r = compute_parameter_statistics(r_phys, mask, "r")
        D_mean_phys = D_phys.mean(dim=0)
        stats_D = compute_parameter_statistics(D_mean_phys, mask, "D_mean")
        print_statistics(stats_s0)
        print_statistics(stats_r)
        print_statistics(stats_D)
        row = {
            "case_name": case_name,
            "slice_idx": slice_idx,
            "s0": stats_s0,
            "r": stats_r,
            "D_mean": stats_D,
        }
        all_param_stats.append(row)
        case_param_stats.append(row)

        # write into case buffers
        if affine is not None:
            case_affine = affine  # prefer dataset-provided affine
        g3d, d4d = case_buffers
        z = slice_idx
        if z < 0 or z >= Z:
            print(f"Warning: slice_idx {z} out of range [0,{Z-1}] for case {case_name}; skipping slice.")
            continue
        # globals 3D
        g3d['s0'][:, :, z] = s0.cpu().numpy().astype(np.float32)
        g3d['r'][:, :, z] = r_phys.cpu().numpy().astype(np.float32)
        g3d['sigma20'][:, :, z] = sigma20_phys.cpu().numpy().astype(np.float32)
        g3d['sigma30'][:, :, z] = sigma30_phys.cpu().numpy().astype(np.float32)
        # directions 4D (vol0 = mean, vol1..30 = per-direction)
        dir_names = ["D", "sigma11", "sigma02", "sigma21", "sigma12", "sigma03"]
        for pi, pname in enumerate(dir_names):
            arr_dir = theta_dir_phys[:, pi, :, :].cpu().numpy()  # [30,H,W] in physical units
            dir_mean_np = arr_dir.mean(axis=0).astype(np.float32)  # [H,W]
            d4d[pname][:, :, z, 0] = dir_mean_np
            # move direction axis to last to match [H,W,30]
            arr_dir_hw30 = np.moveaxis(arr_dir, 0, -1).astype(np.float32)  # [H,W,30]
            d4d[pname][:, :, z, 1:] = arr_dir_hw30
        case_used_slices.append(z)

    # summary
    # save last case if any
    if current_case is not None and case_buffers is not None and case_affine is not None:
        out_case_dir = out_dir / current_case
        _save_case_volumes(out_case_dir, case_affine, case_buffers[0], case_buffers[1])

    print("\n" + "=" * 60)
    print("Inference done.")
    print(f"Results at: {out_dir.resolve()}")

    if len(all_metrics) > 0:
        keys = list(all_metrics[0].keys())
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
        print_evaluation_metrics(avg, title="Average reconstruction quality")

        # save report
        report = out_dir / "evaluation_metrics.txt"
        with report.open("w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("Averages:\n")
            for k in keys:
                f.write(f"  {k}: {avg[k]:.6f}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Per-sample details\n")
            f.write("=" * 60 + "\n\n")
            for m, s in zip(all_metrics, all_param_stats):
                case_name = s["case_name"]
                slice_idx = s["slice_idx"]
                f.write(f"\n{case_name}_slice{slice_idx:03d}:\n")
                f.write("  Reconstruction:\n")
                for k in keys:
                    f.write(f"    {k}: {m[k]:.6f}\n")
                f.write("  Parameters:\n")
                f.write(f"    s0: mean={s['s0']['mean']:.4f}, std={s['s0']['std']:.4f}\n")
                f.write(f"    r: mean={s['r']['mean']:.4f}, std={s['r']['std']:.4f}\n")
                f.write(f"    D_mean: mean={s['D_mean']['mean']:.6f}, std={s['D_mean']['std']:.6f}\n")
        print(f"Saved metrics to: {report}")


if __name__ == "__main__":
    main()
