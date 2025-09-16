"""
ユーティリティ関数
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import json
import os

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, checkpoint_dir, filename):
    """
    チェックポイントを保存
    
    Args:
        model: モデル
        optimizer: オプティマイザー
        scheduler: スケジューラー
        epoch: エポック数
        best_loss: ベスト損失
        checkpoint_dir: チェックポイントディレクトリ
        filename: ファイル名
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    チェックポイントを読み込み
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザー（オプション）
        scheduler: スケジューラー（オプション）
    
    Returns:
        checkpoint: チェックポイント辞書
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # モデルの状態を読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # オプティマイザーの状態を読み込み
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # スケジューラーの状態を読み込み
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def calculate_metrics(pred, target, mask=None):
    """
    メトリクスを計算
    
    Args:
        pred: 予測値
        target: ターゲット値
        mask: マスク（オプション）
    
    Returns:
        metrics: メトリクスの辞書
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask
    
    # 平均絶対誤差
    mae = torch.mean(torch.abs(pred - target)).item()
    
    # 平均二乗誤差
    mse = torch.mean((pred - target) ** 2).item()
    
    # 平方根平均二乗誤差
    rmse = np.sqrt(mse)
    
    # 相関係数
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    if mask is not None:
        mask_flat = mask.flatten()
        valid_indices = mask_flat > 0
        pred_flat = pred_flat[valid_indices]
        target_flat = target_flat[valid_indices]
    
    if len(pred_flat) > 1:
        correlation = np.corrcoef(pred_flat.cpu().numpy(), target_flat.cpu().numpy())[0, 1]
    else:
        correlation = 0.0
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation
    }
    
    return metrics

def depth_to_volume(depth_map, pixel_size_mm=0.1, lesion_mask=None):
    """
    深度マップから体積を計算
    
    Args:
        depth_map: 深度マップ
        pixel_size_mm: ピクセルサイズ（mm）
        lesion_mask: 病変マスク（オプション）
    
    Returns:
        volume: 体積（mm³）
    """
    if lesion_mask is not None:
        # マスクが指定されている場合、マスク内のピクセルのみを計算
        depth_masked = depth_map * lesion_mask
        pixel_count = np.sum(lesion_mask)
    else:
        # マスクが指定されていない場合、全体を計算
        depth_masked = depth_map
        pixel_count = depth_map.size
    
    # 体積を計算（深度 × ピクセル面積）
    volume = np.sum(depth_masked) * (pixel_size_mm ** 2)
    
    return volume

def calculate_reduction_rate(volume_before, volume_after):
    """
    前後縮小率を計算
    
    Args:
        volume_before: 治療前の体積
        volume_after: 治療後の体積
    
    Returns:
        reduction_rate: 縮小率（%）
    """
    if volume_before == 0:
        return 0.0
    
    reduction_rate = ((volume_before - volume_after) / volume_before) * 100
    return reduction_rate

def create_lesion_mask(image, threshold=128):
    """
    画像から病変マスクを作成（簡単な閾値処理）
    
    Args:
        image: 入力画像
        threshold: 閾値
    
    Returns:
        mask: 病変マスク
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 閾値処理
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # モルフォロジー処理でノイズを除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 正規化
    mask = mask / 255.0
    
    return mask

def visualize_depth_overlay(image, depth_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    深度マップを画像にオーバーレイ
    
    Args:
        image: 元の画像
        depth_map: 深度マップ
        alpha: 透明度
        colormap: カラーマップ
    
    Returns:
        overlay: オーバーレイ画像
    """
    # 深度マップをカラー化
    depth_normalized = ((depth_map - depth_map.min()) / 
                       (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    # 画像サイズを合わせる
    if image.shape[:2] != depth_colored.shape[:2]:
        depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))
    
    # オーバーレイ
    overlay = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)
    
    return overlay

def save_results(results, output_path):
    """
    結果をJSONファイルに保存
    
    Args:
        results: 結果の辞書
        output_path: 出力パス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # numpy配列をリストに変換
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {output_path}")

def load_results(input_path):
    """
    結果をJSONファイルから読み込み
    
    Args:
        input_path: 入力パス
    
    Returns:
        results: 結果の辞書
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results

def setup_logging(log_dir, log_level='INFO'):
    """
    ログ設定
    
    Args:
        log_dir: ログディレクトリ
        log_level: ログレベル
    """
    import logging
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイルのパス
    log_file = log_dir / 'training.log'
    
    # ログ設定
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def get_device():
    """
    利用可能なデバイスを取得
    
    Returns:
        device: デバイス
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

