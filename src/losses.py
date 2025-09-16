"""
損失関数の定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthLoss(nn.Module):
    """
    深度推定用の損失関数
    """
    
    def __init__(self, depth_weight=1.0, smoothness_weight=0.1):
        """
        Args:
            depth_weight: 深度損失の重み
            smoothness_weight: 平滑化損失の重み
        """
        super().__init__()
        self.depth_weight = depth_weight
        self.smoothness_weight = smoothness_weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_depth, target_depth, mask=None):
        """
        Args:
            pred_depth: 予測深度 [B, 1, H, W]
            target_depth: ターゲット深度 [B, 1, H, W]
            mask: 有効なピクセルのマスク [B, 1, H, W]
        
        Returns:
            total_loss: 総損失
            depth_loss: 深度損失
            smoothness_loss: 平滑化損失
        """
        if mask is not None:
            # マスクが指定されている場合、有効なピクセルのみで損失を計算
            pred_depth_masked = pred_depth * mask
            target_depth_masked = target_depth * mask
            
            # 深度損失
            depth_loss = self.l1_loss(pred_depth_masked, target_depth_masked)
        else:
            # マスクが指定されていない場合、全体で損失を計算
            depth_loss = self.l1_loss(pred_depth, target_depth)
        
        # 平滑化損失
        smoothness_loss = self.compute_smoothness_loss(pred_depth)
        
        # 総損失
        total_loss = self.depth_weight * depth_loss + self.smoothness_weight * smoothness_loss
        
        return total_loss, depth_loss, smoothness_loss
    
    def compute_smoothness_loss(self, depth):
        """
        深度の平滑化損失を計算
        
        Args:
            depth: 深度マップ [B, 1, H, W]
        
        Returns:
            smoothness_loss: 平滑化損失
        """
        # 水平方向の勾配
        grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        # 垂直方向の勾配
        grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        # 平滑化損失
        smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)
        
        return smoothness_loss

class VolumeLoss(nn.Module):
    """
    体積推定用の損失関数
    """
    
    def __init__(self, volume_weight=1.0):
        """
        Args:
            volume_weight: 体積損失の重み
        """
        super().__init__()
        self.volume_weight = volume_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_volume, target_volume):
        """
        Args:
            pred_volume: 予測体積 [B]
            target_volume: ターゲット体積 [B]
        
        Returns:
            volume_loss: 体積損失
        """
        volume_loss = self.mse_loss(pred_volume, target_volume)
        return self.volume_weight * volume_loss

class ReductionRateLoss(nn.Module):
    """
    前後縮小率推定用の損失関数
    """
    
    def __init__(self, reduction_weight=1.0):
        """
        Args:
            reduction_weight: 縮小率損失の重み
        """
        super().__init__()
        self.reduction_weight = reduction_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_reduction, target_reduction):
        """
        Args:
            pred_reduction: 予測縮小率 [B]
            target_reduction: ターゲット縮小率 [B]
        
        Returns:
            reduction_loss: 縮小率損失
        """
        reduction_loss = self.mse_loss(pred_reduction, target_reduction)
        return self.reduction_weight * reduction_loss

class CombinedLoss(nn.Module):
    """
    複合損失関数（深度 + 体積 + 縮小率）
    """
    
    def __init__(self, depth_weight=1.0, volume_weight=0.1, reduction_weight=0.1, 
                 smoothness_weight=0.1):
        """
        Args:
            depth_weight: 深度損失の重み
            volume_weight: 体積損失の重み
            reduction_weight: 縮小率損失の重み
            smoothness_weight: 平滑化損失の重み
        """
        super().__init__()
        self.depth_loss = DepthLoss(depth_weight, smoothness_weight)
        self.volume_loss = VolumeLoss(volume_weight)
        self.reduction_loss = ReductionRateLoss(reduction_weight)
    
    def forward(self, pred_depth, target_depth, pred_volume, target_volume, 
                pred_reduction, target_reduction, mask=None):
        """
        Args:
            pred_depth: 予測深度 [B, 1, H, W]
            target_depth: ターゲット深度 [B, 1, H, W]
            pred_volume: 予測体積 [B]
            target_volume: ターゲット体積 [B]
            pred_reduction: 予測縮小率 [B]
            target_reduction: ターゲット縮小率 [B]
            mask: 有効なピクセルのマスク [B, 1, H, W]
        
        Returns:
            total_loss: 総損失
            losses: 各損失の辞書
        """
        # 深度損失
        depth_total_loss, depth_loss, smoothness_loss = self.depth_loss(
            pred_depth, target_depth, mask
        )
        
        # 体積損失
        volume_loss = self.volume_loss(pred_volume, target_volume)
        
        # 縮小率損失
        reduction_loss = self.reduction_loss(pred_reduction, target_reduction)
        
        # 総損失
        total_loss = depth_total_loss + volume_loss + reduction_loss
        
        losses = {
            'total': total_loss,
            'depth': depth_loss,
            'smoothness': smoothness_loss,
            'volume': volume_loss,
            'reduction': reduction_loss
        }
        
        return total_loss, losses

