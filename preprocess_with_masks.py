"""
Preprocess raw NIfTI into cached numpy slices with multiple mask levels.

Outputs per slice under `<cache_dir>/<split>/<case>/`:
  - slice_XXX.npy: float32, shape [H, W, 630], raw signal (no modification)
  - mask_original_XXX.npy: float32, shape [H, W], original binary mask
  - temp_mask_630_XXX.npy: float32, shape [H, W, 630], per-channel observation mask
  - dir_mask_30_XXX.npy: float32, shape [H, W, 30], per-direction validity mask
  - voxel_mask_final_XXX.npy: float32, shape [H, W], final training mask

Key features:
  - No signal modification; preserve raw values
  - Generate 630-channel observation mask based on value range [100, 20000]
  - Generate 30-direction mask based on valid observation count (>= 15 out of 20)
  - Generate final voxel mask by filtering voxels with:
    * >= 10 unusable directions, OR
    * > 10 invalid b0 observations
  - Skip slices with >10% invalid voxel ratio (no output files written)
  - Statistics: original mask count, affected voxels, final mask count, skipped slices

Usage example:
  python preprocess_with_masks.py \
      --data_root data_root \
      --split train val \
      --grad_csv data_root/grad_126.csv \
      --cache_dir cache_masked \
      --signal_min 100 \
      --signal_max 20000 \
      --dir_valid_threshold 15 \
      --max_invalid_dirs 10 \
      --max_invalid_b0 10 \
      --max_invalid_voxel_ratio 0.1
"""

from pathlib import Path
import argparse
import os
import threading
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Default valid signal range
SIGNAL_MIN = 100.0
SIGNAL_MAX = 20000.0

# Direction validity threshold (out of 20 observations per direction)
DIR_VALID_THRESHOLD = 15

# Voxel filtering thresholds
MAX_INVALID_DIRS = 10  # If >= 5 directions are invalid, mask out voxel
MAX_INVALID_B0 = 10    # If > 10 b0 observations are invalid, mask out voxel


def load_gradient_table(csv_path: str) -> pd.DataFrame:
    """
    Load gradient table CSV.
    Expected columns: vol_idx, b_value, gx, gy, gz (or similar).
    Returns DataFrame with at least 'b_value' column.
    """
    df = pd.read_csv(csv_path)
    # Ensure we have b_value column
    if 'b_value' not in df.columns and 'b' not in df.columns:
        raise ValueError(f"grad_csv must contain 'b_value' or 'b' column. Found: {df.columns.tolist()}")
    if 'b' in df.columns and 'b_value' not in df.columns:
        df['b_value'] = df['b']
    return df


def build_channel_mapping(grad_df: pd.DataFrame, num_tes: int = 5) -> Dict:
    """
    Build mapping from 630-channel index to (te_idx, vol_idx, b_value, dir_id).
    
    Args:
        grad_df: gradient table with 126 rows (one per vol_idx in a single TE)
        num_tes: number of TEs (default 5)
    
    Returns:
        Dict with keys:
            - 'b0_channels': list of channel indices corresponding to b0
            - 'nonb0_channels': dict mapping direction_id -> list of channel indices
            - 'channel_info': dict mapping channel_idx -> (te_idx, vol_idx, b_value, dir_id)
    """
    if len(grad_df) != 126:
        raise ValueError(f"grad_df should have 126 rows, got {len(grad_df)}")
    
    # Identify b0 and non-b0 volumes
    b0_mask = grad_df['b_value'] == 0
    b0_vol_indices = grad_df[b0_mask].index.tolist()
    nonb0_df = grad_df[~b0_mask].copy()
    
    # Assign direction IDs to non-b0 volumes
    # Assuming: 4 b-values * 30 directions = 120 volumes
    # Each direction appears in 4 b-values cyclically
    # Direction ID is assigned based on order in the gradient table
    nonb0_df = nonb0_df.reset_index(drop=True)
    # Assuming the 120 non-b0 volumes are organized as:
    # [dir1@b500, dir1@b1000, dir1@b1500, dir1@b2500, dir2@b500, ...]
    # OR cyclically: we need to extract direction ID from gradient vectors
    
    # For simplicity, we group by gradient direction (gx, gy, gz) if available
    if all(col in grad_df.columns for col in ['gx', 'gy', 'gz']):
        # Group by direction vector to get unique directions
        nonb0_df['dir_key'] = list(zip(nonb0_df['gx'], nonb0_df['gy'], nonb0_df['gz']))
        unique_dirs = nonb0_df['dir_key'].unique()
        dir_to_id = {d: i for i, d in enumerate(unique_dirs)}
        nonb0_df['dir_id'] = nonb0_df['dir_key'].map(dir_to_id)
    else:
        # Fallback: assume sequential grouping (4 b-values per direction)
        # 120 volumes / 4 b-values = 30 directions
        # Volumes are organized as: dir0@4b, dir1@4b, ..., dir29@4b
        nonb0_df['dir_id'] = (nonb0_df.index // 4) % 30
    
    # Build 630-channel mapping
    channel_info = {}
    b0_channels = []
    nonb0_channels = {i: [] for i in range(30)}  # 30 directions
    
    for te_idx in range(num_tes):
        for vol_idx in range(126):
            ch_idx = te_idx * 126 + vol_idx
            b_val = grad_df.loc[vol_idx, 'b_value']
            
            if vol_idx in b0_vol_indices:
                # b0 observation
                channel_info[ch_idx] = {
                    'te_idx': te_idx,
                    'vol_idx': vol_idx,
                    'b_value': b_val,
                    'dir_id': None,  # b0 has no direction
                }
                b0_channels.append(ch_idx)
            else:
                # non-b0 observation
                row_idx = nonb0_df[nonb0_df.index + len(b0_vol_indices) == vol_idx].index[0]
                dir_id = int(nonb0_df.loc[row_idx, 'dir_id'])
                channel_info[ch_idx] = {
                    'te_idx': te_idx,
                    'vol_idx': vol_idx,
                    'b_value': b_val,
                    'dir_id': dir_id,
                }
                nonb0_channels[dir_id].append(ch_idx)
    
    return {
        'b0_channels': b0_channels,
        'nonb0_channels': nonb0_channels,
        'channel_info': channel_info,
    }


def maybe_write_uncompressed_nii(src: Path, dst: Path):
    """Write uncompressed NIfTI if destination doesn't exist."""
    if dst.exists():
        return
    img = nib.load(str(src))
    nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine, img.header).to_filename(str(dst))


def preprocess_split(args, split: str, channel_mapping: Dict):
    """
    Preprocess one data split (train/val/test).
    
    Args:
        args: command-line arguments
        split: split name (e.g., 'train')
        channel_mapping: output from build_channel_mapping()
    """
    data_root = Path(args.data_root)
    split_dir = data_root / split
    cache_root = (Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
                  else data_root / args.cache_dir)
    out_split = cache_root / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Create log file in cache_root directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = cache_root / f"preprocess_{split}_{timestamp}.txt"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    log_lock = threading.Lock()

    def log_print(msg):
        """Print to console and write to log file"""
        print(msg)
        with log_lock:
            log_file.write(msg + '\n')
            log_file.flush()

    cases = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    signal_min = float(args.signal_min)
    signal_max = float(args.signal_max)
    dir_valid_threshold = int(args.dir_valid_threshold)
    max_invalid_dirs = int(args.max_invalid_dirs)
    max_invalid_b0 = int(args.max_invalid_b0)
    mask_threshold = float(args.mask_threshold)

    b0_channels = channel_mapping['b0_channels']
    nonb0_channels = channel_mapping['nonb0_channels']

    num_workers = getattr(args, "num_workers", 0)
    if num_workers is None or num_workers <= 0:
        slice_workers = os.cpu_count() or 1
    else:
        slice_workers = num_workers

    case_workers = getattr(args, "case_workers", 1)
    if case_workers is None or case_workers <= 0:
        case_workers = 1

    log_print(
        f"Using {case_workers} case workers and {slice_workers} slice workers per case"
    )

    def process_case(case_dir):
        log_print(f"\n{'='*80}")
        log_print(f"[{split}] Processing case: {case_dir.name}")
        log_print(f"{'='*80}")

        out_case = out_split / case_dir.name
        out_case.mkdir(exist_ok=True)

        vols = []
        affine = None
        for i in range(1, 6):
            fn_gz = case_dir / f"te{i}.nii.gz"
            fn = case_dir / f"te{i}.nii"
            src = fn if fn.exists() else fn_gz
            if args.write_nii and fn_gz.exists() and not fn.exists():
                maybe_write_uncompressed_nii(fn_gz, fn)
                src = fn
            img = nib.load(str(src))
            data = img.get_fdata(dtype=np.float32)
            if affine is None:
                affine = img.affine
            vols.append(data)

        vol4d = np.concatenate(vols, axis=-1)

        m_gz = case_dir / "mask.nii.gz"
        m = case_dir / "mask.nii"
        msrc = m if m.exists() else m_gz
        if args.write_nii and m_gz.exists() and not m.exists():
            maybe_write_uncompressed_nii(m_gz, m)
            msrc = m
        mask_all = nib.load(str(msrc)).get_fdata(dtype=np.float32)

        H, W, Z, C = vol4d.shape
        if C != 630:
            raise RuntimeError(f"Expected 630 channels, got {C}")

        # Build per-TE quantile-based signal thresholds
        brain_mask_bool = mask_all > mask_threshold
        n_tes = 5
        channels_per_te = 126

        # case-specific low quantile
        is_case15 = (split == "train" and case_dir.name == "case_15")
        q_low = 0.03 if is_case15 else 0.001
        q_high = 0.99999

        te_lows = np.zeros(n_tes, dtype=np.float32)
        te_highs = np.zeros(n_tes, dtype=np.float32)

        for te_idx in range(n_tes):
            te_vol = vols[te_idx]  # [H,W,Z,126]
            # Only consider values inside the anatomical mask and finite
            arr = te_vol[brain_mask_bool]
            arr = arr.reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                te_lows[te_idx] = signal_min
                te_highs[te_idx] = signal_max
                log_print(
                    f"  [WARN] TE{te_idx+1}: no finite values inside mask; "
                    f"using fallback [{signal_min:.1f}, {signal_max:.1f}]"
                )
            else:
                te_lows[te_idx] = float(np.quantile(arr, q_low))
                te_highs[te_idx] = float(np.quantile(arr, q_high))

        channel_low = np.zeros(C, dtype=np.float32)
        channel_high = np.zeros(C, dtype=np.float32)
        for te_idx in range(n_tes):
            start = te_idx * channels_per_te
            end = start + channels_per_te
            channel_low[start:end] = te_lows[te_idx]
            channel_high[start:end] = te_highs[te_idx]

        log_print(
            f"  Using quantile-based thresholds (q_low={q_low}, q_high={q_high}) per TE"
        )
        for te_idx in range(n_tes):
            log_print(
                f"    TE{te_idx+1}: low={te_lows[te_idx]:.2f}, high={te_highs[te_idx]:.2f}"
            )

        # Buffers for exporting NIfTI masks in original space
        channel_mask_4d = np.zeros((H, W, Z, C), dtype=np.uint8)
        voxel_mask_3d = np.zeros((H, W, Z), dtype=np.uint8)

        stats = {
            'total_slices': Z,
            'skipped_slices': 0,
            'original_fg_voxels': 0,
            'voxels_with_invalid_dirs': 0,
            'voxels_masked_by_dirs': 0,
            'voxels_masked_by_b0': 0,
            'final_fg_voxels': 0,
        }

        def process_slice(z: int):
            nonlocal log_print  # Access log_print from outer scope
            mask_raw = mask_all[:, :, z]
            mask_original = (mask_raw > mask_threshold).astype(np.float32)
            mask_bool = mask_original > 0.5
            original_fg = int(mask_original.sum())

            if original_fg == 0:
                return {
                    'skipped': False,
                    'original_fg_voxels': 0,
                    'voxels_with_invalid_dirs': 0,
                    'voxels_masked_by_dirs': 0,
                    'voxels_masked_by_b0': 0,
                    'final_fg_voxels': 0,
                }

            slice_data = vol4d[:, :, z, :].astype(np.float32)

            temp_mask_630 = np.repeat(mask_original[:, :, np.newaxis], 630, axis=2)

            # Per-channel validity based on per-TE quantile thresholds
            low = channel_low[None, None, :]
            high = channel_high[None, None, :]
            invalid_obs = (slice_data < low) | (slice_data > high)
            temp_mask_630[invalid_obs] = 0.0

            invalid_inside_mask = invalid_obs & mask_bool[:, :, np.newaxis]
            slice_data[invalid_inside_mask] = 100.0

            dir_mask_30 = np.zeros((H, W, 30), dtype=np.float32)

            for dir_id in range(30):
                ch_list = nonb0_channels[dir_id]
                if len(ch_list) == 0:
                    continue
                valid_count = temp_mask_630[:, :, ch_list].sum(axis=2)
                dir_mask_30[:, :, dir_id] = (valid_count >= dir_valid_threshold).astype(np.float32)

            voxel_mask_final = mask_original.copy()

            invalid_dirs_count = 30 - dir_mask_30.sum(axis=2)

            b0_valid_count = temp_mask_630[:, :, b0_channels].sum(axis=2)
            invalid_b0_count = len(b0_channels) - b0_valid_count

            mask_by_dirs = (invalid_dirs_count >= max_invalid_dirs)
            mask_by_b0 = (invalid_b0_count > max_invalid_b0)
            to_mask = mask_by_dirs | mask_by_b0

            voxel_mask_final[to_mask] = 0.0

            voxels_with_any_invalid_dir = (invalid_dirs_count > 0) & mask_bool
            
            # Calculate invalid voxel ratio
            final_fg = int(voxel_mask_final.sum())
            invalid_ratio = 1.0 - (final_fg / original_fg) if original_fg > 0 else 0.0
            max_invalid_voxel_ratio = getattr(args, 'max_invalid_voxel_ratio', 0.1)

            # Update NIfTI buffers (always reflect actual computed masks)
            channel_mask_4d[:, :, z, :] = temp_mask_630.astype(np.uint8)
            voxel_mask_3d[:, :, z] = voxel_mask_final.astype(np.uint8)
            
            # Skip this slice if invalid ratio exceeds threshold
            if invalid_ratio > max_invalid_voxel_ratio:
                log_print(
                    f"  Slice {z:03d}: "
                    f"orig_fg={original_fg:5d} | "
                    f"final_fg={final_fg:5d} | "
                    f"masked_by_dirs={int((mask_by_dirs & mask_bool).sum()):4d} | "
                    f"masked_by_b0={int((mask_by_b0 & mask_bool).sum()):4d} | "
                    f"SKIPPED (invalid ratio: {invalid_ratio*100:.1f}%)"
                )
                return {
                    'skipped': True,
                    'original_fg_voxels': original_fg,
                    'voxels_with_invalid_dirs': int(voxels_with_any_invalid_dir.sum()),
                    'voxels_masked_by_dirs': int((mask_by_dirs & mask_bool).sum()),
                    'voxels_masked_by_b0': int((mask_by_b0 & mask_bool).sum()),
                    'final_fg_voxels': final_fg,
                }
            
            # Write output files only if slice is valid
            np.save(out_case / f"slice_{z:03d}.npy", slice_data)
            # Store mask arrays as compact uint8 (0/1) to save space
            np.save(out_case / f"mask_original_{z:03d}.npy", mask_original.astype(np.uint8))
            np.save(out_case / f"temp_mask_630_{z:03d}.npy", temp_mask_630.astype(np.uint8))
            np.save(out_case / f"dir_mask_30_{z:03d}.npy", dir_mask_30.astype(np.uint8))
            np.save(out_case / f"voxel_mask_final_{z:03d}.npy", voxel_mask_final.astype(np.uint8))

            log_print(
                f"  Slice {z:03d}: "
                f"orig_fg={original_fg:5d} | "
                f"final_fg={final_fg:5d} | "
                f"masked_by_dirs={int((mask_by_dirs & mask_bool).sum()):4d} | "
                f"masked_by_b0={int((mask_by_b0 & mask_bool).sum()):4d}"
            )

            return {
                'skipped': False,
                'original_fg_voxels': original_fg,
                'voxels_with_invalid_dirs': int(voxels_with_any_invalid_dir.sum()),
                'voxels_masked_by_dirs': int((mask_by_dirs & mask_bool).sum()),
                'voxels_masked_by_b0': int((mask_by_b0 & mask_bool).sum()),
                'final_fg_voxels': final_fg,
            }

        if slice_workers <= 1:
            for z in range(Z):
                slice_stats = process_slice(z)
                if slice_stats.get('skipped', False):
                    stats['skipped_slices'] += 1
                for k in slice_stats:
                    if k != 'skipped':
                        stats[k] += slice_stats[k]
        else:
            with ThreadPoolExecutor(max_workers=slice_workers) as executor:
                for slice_stats in executor.map(process_slice, range(Z)):
                    if slice_stats.get('skipped', False):
                        stats['skipped_slices'] += 1
                    for k in slice_stats:
                        if k != 'skipped':
                            stats[k] += slice_stats[k]

        log_print(f"\n{'-'*80}")
        log_print(f"Case {case_dir.name} Summary:")
        log_print(f"  Total slices: {stats['total_slices']}")
        log_print(f"  Skipped slices: {stats['skipped_slices']}")
        log_print(f"  Valid slices: {stats['total_slices'] - stats['skipped_slices']}")
        log_print(f"  Original foreground voxels: {stats['original_fg_voxels']}")
        log_print(f"  Voxels with ≥1 invalid direction: {stats['voxels_with_invalid_dirs']}")
        log_print(f"  Voxels masked due to ≥{max_invalid_dirs} invalid dirs: {stats['voxels_masked_by_dirs']}")
        log_print(f"  Voxels masked due to >{max_invalid_b0} invalid b0: {stats['voxels_masked_by_b0']}")
        log_print(f"  Final foreground voxels: {stats['final_fg_voxels']}")
        log_print(f"  Retention rate: {100.0 * stats['final_fg_voxels'] / max(stats['original_fg_voxels'], 1):.2f}%")
        log_print(f"{'-'*80}")

        # Save NIfTI masks for this case in original space
        try:
            nib.save(
                nib.Nifti1Image(channel_mask_4d.astype(np.uint8), affine),
                case_dir / "channel_mask.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(voxel_mask_3d.astype(np.uint8), affine),
                case_dir / "voxel_mask_final.nii.gz",
            )
            log_print(
                f"  Saved NIfTI masks to: {case_dir / 'channel_mask.nii.gz'} "
                f"and {case_dir / 'voxel_mask_final.nii.gz'}"
            )
        except Exception as e:
            log_print(f"  [WARN] Failed to save NIfTI masks for {case_dir.name}: {e}")

    if case_workers <= 1:
        for case_dir in cases:
            process_case(case_dir)
    else:
        with ThreadPoolExecutor(max_workers=case_workers) as executor:
            list(executor.map(process_case, cases))

    # Close log file at the end of split processing
    log_file.close()
    print(f"\n[{split}] Log saved to: {log_file_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess dMRI data with multi-level masks")
    p.add_argument("--data_root", type=str, default="data_root",
                   help="Root directory containing split folders")
    p.add_argument("--split", nargs="+", default=["train", "val","test"],
                   help="Data splits to process")
    p.add_argument("--grad_csv", type=str, default=str(Path("data_root") / "grad_126.csv"),
                   help="Path to gradient table CSV (126 rows)")
    p.add_argument("--cache_dir", type=str, default="cache_masked",
                   help="Output cache directory")
    p.add_argument("--write_nii", action="store_true", default=True,
                   help="Write uncompressed .nii files for faster I/O")
    p.add_argument("--num_workers", type=int, default=5,
                   help="Number of parallel workers for slice processing (0=auto,1=single-thread)")
    p.add_argument("--case_workers", type=int, default=3,
                   help="Number of parallel workers for case-level processing per split (1=sequential)")
    
    # Signal validity range
    p.add_argument("--signal_min", type=float, default=SIGNAL_MIN,
                   help="Minimum valid signal value")
    p.add_argument("--signal_max", type=float, default=SIGNAL_MAX,
                   help="Maximum valid signal value")
    
    # Direction validity threshold
    p.add_argument("--dir_valid_threshold", type=int, default=DIR_VALID_THRESHOLD,
                   help="Minimum valid observations (out of 20) for a direction to be valid")
    
    # Voxel filtering thresholds
    p.add_argument("--max_invalid_dirs", type=int, default=MAX_INVALID_DIRS,
                   help="Mask voxel if >= this many directions are invalid")
    p.add_argument("--max_invalid_b0", type=int, default=MAX_INVALID_B0,
                   help="Mask voxel if > this many b0 observations are invalid")
    
    # Mask binarization
    p.add_argument("--mask_threshold", type=float, default=0.5,
                   help="Binarize original mask: >threshold is foreground")
    
    # Slice skipping threshold
    p.add_argument("--max_invalid_voxel_ratio", type=float, default=0.1,
                   help="Skip slice if invalid voxel ratio exceeds this threshold (default: 0.1 = 10%)")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create main summary log file
    data_root = Path(args.data_root)
    cache_root = (Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
                  else data_root / args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_log_path = cache_root / f"preprocess_summary_{timestamp}.txt"
    summary_log = open(summary_log_path, 'w', encoding='utf-8')
    
    def main_log_print(msg):
        """Print to console and write to summary log file"""
        print(msg)
        summary_log.write(msg + '\n')
        summary_log.flush()
    
    main_log_print("="*80)
    main_log_print("Loading gradient table...")
    main_log_print("="*80)
    grad_df = load_gradient_table(args.grad_csv)
    main_log_print(f"Loaded {len(grad_df)} gradient directions")
    main_log_print(f"B-values: {sorted(grad_df['b_value'].unique())}")
    
    main_log_print("\n" + "="*80)
    main_log_print("Building channel mapping...")
    main_log_print("="*80)
    channel_mapping = build_channel_mapping(grad_df, num_tes=5)
    main_log_print(f"Total b0 channels: {len(channel_mapping['b0_channels'])}")
    main_log_print(f"Total directions: {len(channel_mapping['nonb0_channels'])}")
    for dir_id in range(min(3, 30)):  # Print first 3 directions
        ch_count = len(channel_mapping['nonb0_channels'][dir_id])
        main_log_print(f"  Direction {dir_id}: {ch_count} observations")
    
    main_log_print("\n" + "="*80)
    main_log_print("Processing splits...")
    main_log_print("="*80)
    for sp in args.split:
        preprocess_split(args, sp, channel_mapping)
    
    main_log_print("\n" + "="*80)
    main_log_print("Preprocessing complete!")
    main_log_print("="*80)
    
    summary_log.close()
    print(f"\nSummary log saved to: {summary_log_path}")
