#!/usr/bin/env python3
"""
既存互換のデータセットインデックス作成スクリプト
（必要なら従来のimages/depthsから作る）
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from dotenv import load_dotenv

def create_dataset_index_from_images_depths(images_dir, depths_dir=None, output_file="dataset_index.csv"):
    """
    既存のimages/depthsディレクトリからデータセットインデックスを作成
    
    Args:
        images_dir: 画像ディレクトリのパス
        depths_dir: 深度ディレクトリのパス（オプション）
        output_file: 出力CSVファイル名
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    # 画像ファイルを検索
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"**/*{ext}"))
    
    print(f"Found {len(image_files)} image files")
    
    # データフレームを作成
    data = []
    
    for image_file in image_files:
        # 相対パスを取得
        rel_image_path = image_file.relative_to(images_path)
        
        # 深度ファイルのパス（オプション）
        rel_depth_path = None
        if depths_dir:
            depths_path = Path(depths_dir)
            if depths_path.exists():
                # 画像ファイル名から深度ファイル名を生成
                depth_file = depths_path / rel_image_path.with_suffix('.npy')
                if depth_file.exists():
                    rel_depth_path = depth_file.relative_to(depths_path)
        
        # ファイル名から情報を抽出（例：patient001_20240101_lesion01_20240101_photo.jpg）
        filename = image_file.stem
        parts = filename.split('_')
        
        patient_id = parts[0] if len(parts) > 0 else "unknown"
        date = parts[1] if len(parts) > 1 else "unknown"
        lesion_id = parts[2] if len(parts) > 2 else "unknown"
        
        data.append({
            'image_path': str(rel_image_path),
            'depth_path': str(rel_depth_path) if rel_depth_path else "",
            'patient_id': patient_id,
            'date': date,
            'lesion_id': lesion_id,
            'volume_mm3': 0,  # デフォルト値
            'reduction_rate_percent': 0.0  # デフォルト値
        })
    
    # データフレームを作成
    df = pd.DataFrame(data)
    
    # CSVファイルに保存
    df.to_csv(output_file, index=False)
    print(f"Dataset index saved to {output_file}")
    print(f"Total records: {len(df)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create dataset index from existing images/depths directories")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--depths_dir", type=str, help="Directory containing depth files (optional)")
    parser.add_argument("--output", type=str, default="dataset_index.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    create_dataset_index_from_images_depths(
        args.images_dir,
        args.depths_dir,
        args.output
    )

if __name__ == "__main__":
    main()

