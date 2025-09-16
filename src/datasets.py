"""
データセットクラスの定義
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

class KeloidDepthDataset(Dataset):
    """
    ケロイド深度データセット
    """
    
    def __init__(self, index_file, data_base_dir, image_size=(256, 256), 
                 transform=None, use_depth=True):
        """
        Args:
            index_file: インデックスCSVファイルのパス
            data_base_dir: データベースディレクトリのパス
            image_size: 画像サイズ (height, width)
            transform: データ拡張
            use_depth: 深度データを使用するかどうか
        """
        self.data_base_dir = Path(data_base_dir)
        self.image_size = image_size
        self.use_depth = use_depth
        
        # インデックスファイルを読み込み
        self.df = pd.read_csv(index_file)
        print(f"Loaded {len(self.df)} samples from {index_file}")
        
        # デフォルトの変換を設定
        if transform is None:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 画像を読み込み
        image_path = self.data_base_dir / row['image_path']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 深度データを読み込み（オプション）
        depth = None
        if self.use_depth and pd.notna(row['depth_path']) and row['depth_path']:
            depth_path = self.data_base_dir / row['depth_path']
            if depth_path.exists():
                depth = np.load(str(depth_path))
            else:
                print(f"Warning: Depth file not found: {depth_path}")
        
        # 変換を適用
        if depth is not None:
            transformed = self.transform(image=image, mask=depth)
            image = transformed['image']
            depth = transformed['mask']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # メタデータ
        metadata = {
            'patient_id': row['patient_id'],
            'date': row['date'],
            'lesion_id': row['lesion_id'],
            'volume_mm3': row['volume_mm3'],
            'reduction_rate_percent': row['reduction_rate_percent']
        }
        
        if depth is not None:
            return image, depth, metadata
        else:
            return image, metadata

def create_data_loaders(train_index_file, val_index_file, data_base_dir, 
                       batch_size=8, num_workers=4, image_size=(256, 256)):
    """
    データローダーを作成
    
    Args:
        train_index_file: 学習用インデックスファイル
        val_index_file: 検証用インデックスファイル
        data_base_dir: データベースディレクトリ
        batch_size: バッチサイズ
        num_workers: ワーカー数
        image_size: 画像サイズ
    
    Returns:
        train_loader, val_loader: データローダー
    """
    # 学習用の変換（データ拡張を含む）
    train_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 検証用の変換（データ拡張なし）
    val_transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # データセットを作成
    train_dataset = KeloidDepthDataset(
        train_index_file, data_base_dir, image_size, train_transform
    )
    val_dataset = KeloidDepthDataset(
        val_index_file, data_base_dir, image_size, val_transform
    )
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

