"""
MiDaS微調整用の学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import timm
import yaml
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

from datasets import create_data_loaders
from losses import DepthLoss, CombinedLoss
from utils import save_checkpoint, load_checkpoint, calculate_metrics

class MiDaSFinetuner(nn.Module):
    """
    MiDaS微調整用モデル
    """
    
    def __init__(self, model_name="midas_v21_small_256", pretrained=True, freeze_backbone=False):
        """
        Args:
            model_name: モデル名
            pretrained: 事前学習済みモデルを使用するか
            freeze_backbone: バックボーンを凍結するか
        """
        super().__init__()
        
        # MiDaSモデルを読み込み
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # バックボーンを凍結する場合
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 深度推定ヘッド
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.backbone.num_features, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        
        # 体積推定ヘッド
        self.volume_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        # 縮小率推定ヘッド
        self.reduction_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 入力画像 [B, 3, H, W]
        
        Returns:
            depth: 深度マップ [B, 1, H, W]
            volume: 体積 [B, 1]
            reduction: 縮小率 [B, 1]
        """
        # バックボーンの特徴を抽出
        features = self.backbone.forward_features(x)
        
        # 深度推定
        depth = self.depth_head(features)
        
        # 体積推定
        volume = self.volume_head(features)
        
        # 縮小率推定
        reduction = self.reduction_head(features)
        
        return depth, volume, reduction

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None):
    """
    1エポックの学習
    
    Args:
        model: モデル
        train_loader: 学習データローダー
        optimizer: オプティマイザー
        criterion: 損失関数
        device: デバイス
        epoch: エポック数
        writer: TensorBoardライター
    
    Returns:
        avg_loss: 平均損失
        metrics: メトリクス
    """
    model.train()
    total_loss = 0
    total_depth_loss = 0
    total_volume_loss = 0
    total_reduction_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # バッチデータを取得
        if len(batch) == 3:  # image, depth, metadata
            images, depths, metadata = batch
            targets = {
                'depth': depths.to(device),
                'volume': torch.tensor(metadata['volume_mm3'], dtype=torch.float32).to(device),
                'reduction': torch.tensor(metadata['reduction_rate_percent'], dtype=torch.float32).to(device)
            }
        else:  # image, metadata (depthなし)
            images, metadata = batch
            targets = {
                'volume': torch.tensor(metadata['volume_mm3'], dtype=torch.float32).to(device),
                'reduction': torch.tensor(metadata['reduction_rate_percent'], dtype=torch.float32).to(device)
            }
        
        images = images.to(device)
        
        # 前向き計算
        optimizer.zero_grad()
        
        pred_depth, pred_volume, pred_reduction = model(images)
        
        # 損失計算
        if 'depth' in targets:
            loss, losses = criterion(
                pred_depth, targets['depth'],
                pred_volume, targets['volume'],
                pred_reduction, targets['reduction']
            )
        else:
            # 深度なしの場合、体積と縮小率のみで損失計算
            volume_loss = nn.MSELoss()(pred_volume.squeeze(), targets['volume'])
            reduction_loss = nn.MSELoss()(pred_reduction.squeeze(), targets['reduction'])
            loss = volume_loss + reduction_loss
            losses = {
                'total': loss,
                'volume': volume_loss,
                'reduction': reduction_loss
            }
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        # 損失を記録
        total_loss += losses['total'].item()
        total_volume_loss += losses['volume'].item()
        total_reduction_loss += losses['reduction'].item()
        
        if 'depth' in losses:
            total_depth_loss += losses['depth'].item()
        
        # プログレスバーを更新
        progress_bar.set_postfix({
            'Loss': f"{losses['total'].item():.4f}",
            'Vol': f"{losses['volume'].item():.4f}",
            'Red': f"{losses['reduction'].item():.4f}"
        })
        
        # TensorBoardに記録
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', losses['total'].item(), global_step)
            writer.add_scalar('Train/Volume_Loss', losses['volume'].item(), global_step)
            writer.add_scalar('Train/Reduction_Loss', losses['reduction'].item(), global_step)
            if 'depth' in losses:
                writer.add_scalar('Train/Depth_Loss', losses['depth'].item(), global_step)
    
    # 平均損失を計算
    avg_loss = total_loss / len(train_loader)
    avg_volume_loss = total_volume_loss / len(train_loader)
    avg_reduction_loss = total_reduction_loss / len(train_loader)
    
    metrics = {
        'loss': avg_loss,
        'volume_loss': avg_volume_loss,
        'reduction_loss': avg_reduction_loss
    }
    
    if total_depth_loss > 0:
        metrics['depth_loss'] = total_depth_loss / len(train_loader)
    
    return avg_loss, metrics

def validate_epoch(model, val_loader, criterion, device, epoch, writer=None):
    """
    1エポックの検証
    
    Args:
        model: モデル
        val_loader: 検証データローダー
        criterion: 損失関数
        device: デバイス
        epoch: エポック数
        writer: TensorBoardライター
    
    Returns:
        avg_loss: 平均損失
        metrics: メトリクス
    """
    model.eval()
    total_loss = 0
    total_volume_loss = 0
    total_reduction_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            # バッチデータを取得
            if len(batch) == 3:  # image, depth, metadata
                images, depths, metadata = batch
                targets = {
                    'depth': depths.to(device),
                    'volume': torch.tensor(metadata['volume_mm3'], dtype=torch.float32).to(device),
                    'reduction': torch.tensor(metadata['reduction_rate_percent'], dtype=torch.float32).to(device)
                }
            else:  # image, metadata (depthなし)
                images, metadata = batch
                targets = {
                    'volume': torch.tensor(metadata['volume_mm3'], dtype=torch.float32).to(device),
                    'reduction': torch.tensor(metadata['reduction_rate_percent'], dtype=torch.float32).to(device)
                }
            
            images = images.to(device)
            
            # 前向き計算
            pred_depth, pred_volume, pred_reduction = model(images)
            
            # 損失計算
            if 'depth' in targets:
                loss, losses = criterion(
                    pred_depth, targets['depth'],
                    pred_volume, targets['volume'],
                    pred_reduction, targets['reduction']
                )
            else:
                # 深度なしの場合、体積と縮小率のみで損失計算
                volume_loss = nn.MSELoss()(pred_volume.squeeze(), targets['volume'])
                reduction_loss = nn.MSELoss()(pred_reduction.squeeze(), targets['reduction'])
                loss = volume_loss + reduction_loss
                losses = {
                    'total': loss,
                    'volume': volume_loss,
                    'reduction': reduction_loss
                }
            
            # 損失を記録
            total_loss += losses['total'].item()
            total_volume_loss += losses['volume'].item()
            total_reduction_loss += losses['reduction'].item()
    
    # 平均損失を計算
    avg_loss = total_loss / len(val_loader)
    avg_volume_loss = total_volume_loss / len(val_loader)
    avg_reduction_loss = total_reduction_loss / len(val_loader)
    
    metrics = {
        'loss': avg_loss,
        'volume_loss': avg_volume_loss,
        'reduction_loss': avg_reduction_loss
    }
    
    # TensorBoardに記録
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Volume_Loss', avg_volume_loss, epoch)
        writer.add_scalar('Val/Reduction_Loss', avg_reduction_loss, epoch)
    
    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Train MiDaS model for keloid depth estimation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # 環境変数を読み込み
    load_dotenv()
    
    # 設定ファイルを読み込み
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # デバイスを設定
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # データローダーを作成
    data_base_dir = os.getenv('DATA_BASE_DIR')
    if not data_base_dir:
        raise ValueError("DATA_BASE_DIR environment variable not set")
    
    train_loader, val_loader = create_data_loaders(
        "meta/index_train.csv",
        "meta/index_val.csv",
        data_base_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    # モデルを作成
    model = MiDaSFinetuner(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    ).to(device)
    
    # 損失関数を作成
    criterion = CombinedLoss(
        depth_weight=config['loss']['depth_weight'],
        volume_weight=0.1,
        reduction_weight=0.1,
        smoothness_weight=config['loss']['smoothness_weight']
    )
    
    # オプティマイザーを作成
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # スケジューラーを作成
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # TensorBoardライターを作成
    writer = SummaryWriter(config['output']['log_dir'])
    
    # 学習開始
    start_epoch = 0
    best_loss = float('inf')
    
    # チェックポイントから再開
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # 学習
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        
        # 検証
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # スケジューラーを更新
        if scheduler:
            scheduler.step()
        
        # ログ出力
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # ベストモデルを保存
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss,
                config['output']['checkpoint_dir'], f"best_model.pth"
            )
        
        # 定期的にチェックポイントを保存
        if epoch % config['output']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss,
                config['output']['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pth"
            )
    
    # 学習完了
    print("Training completed!")
    writer.close()

if __name__ == "__main__":
    main()

