"""
模型架构 - 简化版

UNet + 参数预测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet2D(nn.Module):
    """
    简化的2D UNet
    
    Args:
        in_channels: 输入通道数（630）
        mid_channels: 中间通道数（64）
    """
    
    def __init__(self, in_channels=630, mid_channels=64):
        super().__init__()
        
        # 1x1卷积降维：630 -> 64
        self.compress = nn.Conv2d(in_channels, mid_channels, 1)
        
        # 下采样路径
        self.enc1 = self._conv_block(mid_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # 底部
        self.bottleneck = self._conv_block(512, 512)
        
        # 上采样路径（注意：concat后的通道数 = up输出 + skip连接）
        self.up4 = self._upconv(512, 256)
        self.dec4 = self._conv_block(256 + 512, 256)  # 768 -> 256
        
        self.up3 = self._upconv(256, 128)
        self.dec3 = self._conv_block(128 + 256, 128)  # 384 -> 128
        
        self.up2 = self._upconv(128, 64)
        self.dec2 = self._conv_block(64 + 128, 64)    # 192 -> 64
        
        self.up1 = self._upconv(64, 64)
        self.dec1 = self._conv_block(64 + 64, 64)     # 128 -> 64
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        """卷积块：Conv - GN - ReLU - Conv - GN - ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _upconv(self, in_ch, out_ch):
        """上采样卷积"""
        return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    
    def _match_size_and_concat(self, x_up, x_skip):
        """
        匹配上采样和跳跃连接的尺寸，然后拼接
        
        处理奇数维度的情况（与原版rdmri一致）
        
        Args:
            x_up: 上采样后的张量 [B, C1, H, W]
            x_skip: 跳跃连接的张量 [B, C2, H', W']
            
        Returns:
            拼接后的张量 [B, C1+C2, H', W']
        """
        # 检查尺寸是否匹配
        if x_up.shape[-2:] != x_skip.shape[-2:]:
            # 需要padding
            padding = [0, 0, 0, 0]  # [left, right, top, bottom]
            
            # 检查宽度
            if x_up.shape[-1] != x_skip.shape[-1]:
                padding[1] = 1  # padding right
            
            # 检查高度
            if x_up.shape[-2] != x_skip.shape[-2]:
                padding[3] = 1  # padding bottom
            
            # 应用反射padding
            x_up = F.pad(x_up, padding, mode='reflect')
        
        # 拼接
        return torch.cat([x_up, x_skip], dim=1)
    
    def forward(self, x):
        """
        输入: [B, 630, H, W]
        输出: [B, 64, H, W] - 特征图
        
        通道流动：
        compress: 630 -> 64
        enc1: 64 -> 64 (H, W)
        enc2: 64 -> 128 (H/2, W/2)
        enc3: 128 -> 256 (H/4, W/4)
        enc4: 256 -> 512 (H/8, W/8)
        bottleneck: 512 -> 512 (H/16, W/16)
        
        up4: 512 -> 256, concat 512 -> 768, dec4: 768 -> 256 (H/8, W/8)
        up3: 256 -> 128, concat 256 -> 384, dec3: 384 -> 128 (H/4, W/4)
        up2: 128 -> 64, concat 128 -> 192, dec2: 192 -> 64 (H/2, W/2)
        up1: 64 -> 64, concat 64 -> 128, dec1: 128 -> 64 (H, W)
        """
        # 降维
        x = self.compress(x)  # [B, 64, H, W]
        
        # 编码器（下采样）
        x1 = self.enc1(x)     # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))  # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))  # [B, 256, H/4, W/4]
        x4 = self.enc4(self.pool(x3))  # [B, 512, H/8, W/8]
        
        # 底部（最深层）
        x5 = self.bottleneck(self.pool(x4))  # [B, 512, H/16, W/16]
        
        # 解码器（上采样 + 跳跃连接）
        # 注意：处理奇数维度的padding（与原版rdmri一致）
        x = self.up4(x5)                    # [B, 256, H/8, W/8]
        x = self._match_size_and_concat(x, x4)  # [B, 768, H/8, W/8]
        x = self.dec4(x)                    # [B, 256, H/8, W/8]
        
        x = self.up3(x)                     # [B, 128, H/4, W/4]
        x = self._match_size_and_concat(x, x3)  # [B, 384, H/4, W/4]
        x = self.dec3(x)                    # [B, 128, H/4, W/4]
        
        x = self.up2(x)                     # [B, 64, H/2, W/2]
        x = self._match_size_and_concat(x, x2)  # [B, 192, H/2, W/2]
        x = self.dec2(x)                    # [B, 64, H/2, W/2]
        
        x = self.up1(x)                     # [B, 64, H, W]
        x = self._match_size_and_concat(x, x1)  # [B, 128, H, W]
        x = self.dec1(x)                    # [B, 64, H, W]
        
        return x  # [B, 64, H, W]


class ParameterHeads(nn.Module):
    """
    参数预测头（与原版rdmri一致）
    
    从UNet特征预测物理参数
    - 全局参数: 4个 (ln_s0, r, sigma20, sigma30)
    - 方向参数: 30方向 × 6参数 = 180个
    
    总输出: 184个通道
    
    关键约束：
    - r (弛豫率) 必须 > 0，使用 softplus 约束
    - D (扩散系数) 必须 > 0，使用 softplus 约束
    """
    
    def __init__(self, feature_channels=64, n_dirs=30, hidden_channels=32):
        super().__init__()
        self.n_dirs = n_dirs
        self.softplus_beta = 1.0  # softplus参数
        
        # 全局参数头（使用3x3 + 1x1结构，与原版一致）
        self.global_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
        )
        
        # 方向参数头（使用3x3 + 1x1结构，与原版一致）
        self.dir_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_dirs * 6, kernel_size=1, bias=True),
        )
    
    def forward(self, features):
        """
        输入: [B, 64, H, W] - UNet特征
        输出:
            theta_global: [B, 4, H, W]
            theta_dir: [B, 30, 6, H, W]
        
        参数索引：
            Global (4):
                0: ln s0     - 对数初始信号强度
                1: <r>       - 平均弛豫率 (softplus约束 > 0)
                2: σ20       - 二阶TE系数
                3: σ30       - 三阶TE系数
            
            Direction (30×6 per direction):
                0: D         - 扩散系数 (softplus约束 > 0)
                1: σ11       - 混合一阶系数
                2: σ02       - 二阶b系数
                3: σ21       - 混合二阶系数
                4: σ12       - 混合二阶系数
                5: σ03       - 三阶b系数
        """
        B, C, H, W = features.shape
        
        # 1. Global 参数 [B, 4, H, W]
        global_raw = self.global_head(features)

        # 分离 raw 输出
        lns0 = global_raw[:, 0:1, :, :]          # [B, 1, H, W]
        r_raw = global_raw[:, 1:2, :, :]         # [B, 1, H, W]
        sigma20_raw = global_raw[:, 2:3, :, :]   # [B, 1, H, W]
        sigma30_raw = global_raw[:, 3:4, :, :]   # [B, 1, H, W]

        # 将 lns0 通过 sigmoid 重参数化到 [4.6, 11.2]
        lns0_min, lns0_max = 4.6, 11.2
        lns0 = lns0_min + (lns0_max - lns0_min) * torch.sigmoid(lns0)

        # 归一化到 [0,1] / [-1,1]
        r_norm = torch.sigmoid(r_raw)
        sigma20_norm = torch.sigmoid(sigma20_raw)
        sigma30_norm = torch.tanh(sigma30_raw)

        # 映射到模型空间的有界范围（与缩放后的 t_vec, b_vec 匹配）
        r_max = 10.0          # 对应 r_true ~ [0,0.1] 且 T=100
        sigma20_max = 100.0   # 对应 sigma20_true ~ [0,0.01] 且 T^2=1e4
        sigma30_max = 100.0   # 对应 sigma30_true ~ [-1e-4,1e-4] 且 T^3=1e6

        r = r_norm * r_max
        sigma20 = sigma20_norm * sigma20_max
        sigma30 = sigma30_norm * sigma30_max

        # 拼接 global 参数
        theta_global = torch.cat([lns0, r, sigma20, sigma30], dim=1)  # [B, 4, H, W]

        # 2. Direction 参数 [B, 180, H, W] -> [B, 30, 6, H, W]
        dir_raw = self.dir_head(features)  # [B, 180, H, W]
        dir_raw = dir_raw.view(B, self.n_dirs, 6, H, W)  # [B, 30, 6, H, W]

        # 分离各个参数 raw 输出
        D_raw = dir_raw[:, :, 0:1, :, :]        # [B, 30, 1, H, W]
        sigma11_raw = dir_raw[:, :, 1:2, :, :]  # [B, 30, 1, H, W]
        sigma02_raw = dir_raw[:, :, 2:3, :, :]  # [B, 30, 1, H, W]
        sigma21_raw = dir_raw[:, :, 3:4, :, :]  # [B, 30, 1, H, W]
        sigma12_raw = dir_raw[:, :, 4:5, :, :]  # [B, 30, 1, H, W]
        sigma03_raw = dir_raw[:, :, 5:6, :, :]  # [B, 30, 1, H, W]

        # 归一化到 [0,1] / [-1,1]
        D_norm = torch.sigmoid(D_raw)
        sigma02_norm = torch.sigmoid(sigma02_raw)

        sigma11_norm = torch.tanh(sigma11_raw)
        sigma21_norm = torch.tanh(sigma21_raw)
        sigma12_norm = torch.tanh(sigma12_raw)
        sigma03_norm = torch.tanh(sigma03_raw)

        # 映射到模型空间的有界范围
        D_max = 7.5             # 对应 D_true ~ [0,3] 且 B=2.5
        sigma11_max = 25.0      # 对应 sigma11_true ~ [-0.1,0.1] 且 T*B
        sigma02_max = 12.5      # 对应 sigma02_true ~ [0,2] 且 B^2
        sigma21_max = 25.0      # 对应 sigma21_true ~ [-1e-3,1e-3] 且 T^2*B
        sigma12_max = 62.5      # 对应 sigma12_true ~ [-0.1,0.1] 且 T*B^2
        sigma03_max = 7.8125    # 对应 sigma03_true ~ [-0.5,0.5] 且 B^3

        D = D_norm * D_max
        sigma11 = sigma11_norm * sigma11_max
        sigma02 = sigma02_norm * sigma02_max
        sigma21 = sigma21_norm * sigma21_max
        sigma12 = sigma12_norm * sigma12_max
        sigma03 = sigma03_norm * sigma03_max

        # 拼接 direction 参数
        theta_dir = torch.cat([D, sigma11, sigma02, sigma21, sigma12, sigma03], dim=2)  # [B, 30, 6, H, W]

        return theta_global, theta_dir


class UNetWithHeads(nn.Module):
    """完整模型：UNet + 参数头"""
    
    def __init__(self, in_channels=630, mid_channels=64, n_dirs=30):
        super().__init__()
        self.unet = SimpleUNet2D(in_channels, mid_channels)
        self.heads = ParameterHeads(mid_channels, n_dirs)
    
    def forward(self, x):
        """
        输入: [B, 630, H, W]
        输出:
            theta_global: [B, 4, H, W]
            theta_dir: [B, 30, 6, H, W]
        """
        features = self.unet(x)
        theta_global, theta_dir = self.heads(features)
        return theta_global, theta_dir

