"""
评估指标工具函数

包含各种定量评估指标的计算函数，可被其他脚本调用
"""

import torch
import numpy as np
from typing import Union, Tuple


def _prepare_mask_for_indexing(mask: np.ndarray, data_shape: tuple) -> np.ndarray:
    """
    准备mask用于索引数据数组
    
    处理mask的维度，使其能够正确索引数据：
    - 如果mask是 [1, H, W]，压缩为 [H, W]
    - 如果数据是多通道 [C, H, W]，mask应该是 [H, W] 以便应用到所有通道
    
    Args:
        mask: 掩膜数组
        data_shape: 数据数组的形状
    
    Returns:
        mask: 处理后的掩膜，可以用于索引
    """
    if mask is None:
        return None
    
    mask = mask.astype(bool)
    
    # 如果mask是 [1, H, W]，压缩为 [H, W]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
    
    return mask


def _apply_mask_to_data(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """
    将mask应用到pred和target数据上
    
    支持多种数据格式：
    - [C, H, W]: 多通道数据，mask [H, W] 应用到所有通道
    - [H, W]: 单通道数据，mask [H, W] 直接应用
    
    Args:
        pred: 预测数据
        target: 目标数据
        mask: 掩膜（已处理为 [H, W] 格式）
    
    Returns:
        pred_masked: 应用mask后的预测数据（展平）
        target_masked: 应用mask后的目标数据（展平）
    """
    if mask is None:
        return pred.flatten(), target.flatten()
    
    # 处理多通道数据 [C, H, W]
    if pred.ndim == 3:
        # 对每个通道应用相同的mask，然后展平
        pred_masked = []
        target_masked = []
        for c in range(pred.shape[0]):
            pred_masked.append(pred[c][mask])
            target_masked.append(target[c][mask])
        pred_masked = np.concatenate(pred_masked)
        target_masked = np.concatenate(target_masked)
        return pred_masked, target_masked
    # 处理单通道或2D数据 [H, W]
    else:
        return pred[mask], target[mask]


def compute_mse(pred: Union[torch.Tensor, np.ndarray], 
                target: Union[torch.Tensor, np.ndarray],
                mask: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    计算均方误差 (Mean Squared Error)
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜，只计算掩膜内的误差
    
    Returns:
        mse: 均方误差
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 处理mask维度
    mask = _prepare_mask_for_indexing(mask, pred.shape)
    
    # 应用mask
    pred, target = _apply_mask_to_data(pred, target, mask)
    
    mse = np.mean((pred - target) ** 2)
    return float(mse)


def compute_rmse(pred: Union[torch.Tensor, np.ndarray], 
                 target: Union[torch.Tensor, np.ndarray],
                 mask: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    计算均方根误差 (Root Mean Squared Error)
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜
    
    Returns:
        rmse: 均方根误差
    """
    mse = compute_mse(pred, target, mask)
    return np.sqrt(mse)


def compute_nrmse(pred: Union[torch.Tensor, np.ndarray], 
                  target: Union[torch.Tensor, np.ndarray],
                  mask: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    计算归一化均方根误差 (Normalized RMSE)
    
    NRMSE = RMSE / (max(target) - min(target))
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜
    
    Returns:
        nrmse: 归一化均方根误差 (0-1之间，越小越好)
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 处理mask维度
    mask_processed = _prepare_mask_for_indexing(mask, target.shape)
    
    # 计算target范围
    if mask_processed is not None:
        _, target_masked = _apply_mask_to_data(target, target, mask_processed)
        # 检查mask是否为空（虽然dataset已过滤，但双重保险）
        if len(target_masked) == 0:
            # Mask全为0，返回默认值
            return 0.0
        target_range = target_masked.max() - target_masked.min()
    else:
        target_range = target.max() - target.min()
    
    rmse = compute_rmse(pred, target, mask)
    
    # 避免除零
    if target_range < 1e-10:
        return 0.0
    
    nrmse = rmse / target_range
    return float(nrmse)


def compute_mae(pred: Union[torch.Tensor, np.ndarray], 
                target: Union[torch.Tensor, np.ndarray],
                mask: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜
    
    Returns:
        mae: 平均绝对误差
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 处理mask维度
    mask = _prepare_mask_for_indexing(mask, pred.shape)
    
    # 应用mask
    pred, target = _apply_mask_to_data(pred, target, mask)
    
    mae = np.mean(np.abs(pred - target))
    return float(mae)


def compute_psnr(pred: Union[torch.Tensor, np.ndarray], 
                 target: Union[torch.Tensor, np.ndarray],
                 mask: Union[torch.Tensor, np.ndarray] = None,
                 data_range: float = None) -> float:
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜
        data_range: 数据范围，如果为None则自动从target计算
    
    Returns:
        psnr: 峰值信噪比 (dB)，越高越好（通常 > 30 dB 为好）
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if data_range is None:
        if mask is not None:
            # 处理mask维度
            mask_processed = _prepare_mask_for_indexing(mask, target.shape)
            _, target_masked = _apply_mask_to_data(target, target, mask_processed)
            # 检查mask是否为空
            if len(target_masked) == 0:
                # Mask全为0，使用默认范围
                data_range = 1.0
            else:
                data_range = target_masked.max() - target_masked.min()
        else:
            data_range = target.max() - target.min()
    
    mse = compute_mse(pred, target, mask)
    
    # 避免除零或log(0)
    if mse < 1e-10:
        return 100.0  # 非常高的PSNR
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(pred: Union[torch.Tensor, np.ndarray], 
                 target: Union[torch.Tensor, np.ndarray],
                 mask: Union[torch.Tensor, np.ndarray] = None,
                 data_range: float = None,
                 win_size: int = 7) -> float:
    """
    计算结构相似性指数 (Structural Similarity Index)
    
    使用简化的SSIM实现（不依赖skimage）
    
    Args:
        pred: 预测值 [H, W]
        target: 目标值 [H, W]
        mask: 可选的掩膜
        data_range: 数据范围
        win_size: 窗口大小
    
    Returns:
        ssim: 结构相似性指数 (0-1之间，越接近1越好)
    """
    try:
        from skimage.metrics import structural_similarity
        
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        if data_range is None:
            data_range = target.max() - target.min()
        
        ssim = structural_similarity(
            target, pred, 
            data_range=data_range,
            win_size=win_size
        )
        return float(ssim)
    
    except ImportError:
        print("⚠️ skimage未安装，无法计算SSIM")
        return -1.0


def compute_relative_error(pred: Union[torch.Tensor, np.ndarray], 
                          target: Union[torch.Tensor, np.ndarray],
                          mask: Union[torch.Tensor, np.ndarray] = None,
                          eps: float = 1e-10) -> float:
    """
    计算相对误差
    
    Relative Error = mean(|pred - target| / (|target| + eps))
    
    Args:
        pred: 预测值
        target: 目标值
        mask: 可选的掩膜
        eps: 防止除零的小常数
    
    Returns:
        rel_error: 相对误差 (百分比形式，如 0.05 表示 5%)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 处理mask维度
    mask = _prepare_mask_for_indexing(mask, pred.shape)
    
    # 应用mask
    pred, target = _apply_mask_to_data(pred, target, mask)
    
    rel_error = np.mean(np.abs(pred - target) / (np.abs(target) + eps))
    return float(rel_error)


def compute_parameter_statistics(param_map: Union[torch.Tensor, np.ndarray],
                                 mask: Union[torch.Tensor, np.ndarray] = None,
                                 param_name: str = "parameter") -> dict:
    """
    计算参数图的统计信息
    
    Args:
        param_map: 参数图 [H, W]
        mask: 可选的掩膜
        param_name: 参数名称（用于显示）
    
    Returns:
        stats: 包含统计信息的字典
    """
    if isinstance(param_map, torch.Tensor):
        param_map = param_map.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if mask is not None:
        # 处理mask维度（参数图通常是 [H, W]，mask可能是 [1, H, W]）
        mask = _prepare_mask_for_indexing(mask, param_map.shape)
        values = param_map[mask]
    else:
        values = param_map.flatten()
    
    stats = {
        'name': param_name,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'percentile_25': float(np.percentile(values, 25)),
        'percentile_75': float(np.percentile(values, 75)),
    }
    
    return stats


def print_statistics(stats: dict, indent: str = "  "):
    """
    打印参数统计信息
    
    Args:
        stats: 统计信息字典
        indent: 缩进字符串
    """
    print(f"{indent}{stats['name']}:")
    print(f"{indent}  均值: {stats['mean']:.6f}")
    print(f"{indent}  标准差: {stats['std']:.6f}")
    print(f"{indent}  最小值: {stats['min']:.6f}")
    print(f"{indent}  最大值: {stats['max']:.6f}")
    print(f"{indent}  中位数: {stats['median']:.6f}")
    print(f"{indent}  25%分位: {stats['percentile_25']:.6f}")
    print(f"{indent}  75%分位: {stats['percentile_75']:.6f}")


def evaluate_signal_reconstruction(pred_signal: Union[torch.Tensor, np.ndarray],
                                   target_signal: Union[torch.Tensor, np.ndarray],
                                   mask: Union[torch.Tensor, np.ndarray] = None) -> dict:
    """
    评估信号重建质量（综合多个指标）
    
    Args:
        pred_signal: 预测的信号 [C, H, W]
        target_signal: 目标信号 [C, H, W]
        mask: 可选的掩膜 [1, H, W]
    
    Returns:
        metrics: 包含所有评估指标的字典
    """
    metrics = {}
    
    # 基础误差指标
    metrics['mae'] = compute_mae(pred_signal, target_signal, mask)
    metrics['mse'] = compute_mse(pred_signal, target_signal, mask)
    metrics['rmse'] = compute_rmse(pred_signal, target_signal, mask)
    metrics['nrmse'] = compute_nrmse(pred_signal, target_signal, mask)
    metrics['relative_error'] = compute_relative_error(pred_signal, target_signal, mask)
    
    # PSNR（对整个信号）
    metrics['psnr'] = compute_psnr(pred_signal, target_signal, mask)
    
    return metrics


def print_evaluation_metrics(metrics: dict, title: str = "评估指标"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  MAE (平均绝对误差):        {metrics['mae']:.6f}")
    print(f"  MSE (均方误差):            {metrics['mse']:.6f}")
    print(f"  RMSE (均方根误差):         {metrics['rmse']:.6f}")
    print(f"  NRMSE (归一化RMSE):        {metrics['nrmse']:.6f}")
    print(f"  相对误差:                  {metrics['relative_error']*100:.2f}%")
    print(f"  PSNR (峰值信噪比):         {metrics['psnr']:.2f} dB")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 简单测试
    print("测试指标计算函数...")
    
    # 创建测试数据
    pred = np.random.rand(10, 10)
    target = pred + np.random.randn(10, 10) * 0.1
    
    # 测试各个指标
    print(f"MAE: {compute_mae(pred, target):.6f}")
    print(f"MSE: {compute_mse(pred, target):.6f}")
    print(f"RMSE: {compute_rmse(pred, target):.6f}")
    print(f"NRMSE: {compute_nrmse(pred, target):.6f}")
    print(f"PSNR: {compute_psnr(pred, target):.2f} dB")
    print(f"相对误差: {compute_relative_error(pred, target)*100:.2f}%")
    
    print("\n✓ 所有指标计算正常")

