#!/usr/bin/env python3
"""
ground_truth_master.csv から学習用インデックスファイルを生成するスクリプト
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from dotenv import load_dotenv

def create_index_from_ground_truth_master(data_base_dir, output_dir="meta"):
    """
    ground_truth_master.csv から学習用インデックスファイルを生成
    
    Args:
        data_base_dir: データベースディレクトリのパス
        output_dir: 出力ディレクトリ
    """
    # ground_truth_master.csv のパス
    gt_csv_path = Path(data_base_dir) / "ground_truth_master.csv"
    
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"ground_truth_master.csv not found at {gt_csv_path}")
    
    # CSVを読み込み
    df = pd.read_csv(gt_csv_path)
    print(f"Loaded {len(df)} records from ground_truth_master.csv")
    
    # 画像パスを生成
    image_paths = []
    depth_paths = []
    
    for _, row in df.iterrows():
        patient_id = str(row['patient_id'])
        date = str(row['date'])
        lesion_id = str(row['lesion_id'])
        
        # 画像パス
        image_path = f"02_data_processed/prospective/{patient_id}/{date}/{lesion_id}_{date}_photo.jpg"
        image_paths.append(image_path)
        
        # 深度パス（Vectra由来の擬似深度、任意）
        depth_path = f"02_data_processed/prospective/{patient_id}/{date}/{lesion_id}_{date}_depth.npy"
        depth_paths.append(depth_path)
    
    # インデックスDataFrameを作成
    index_df = pd.DataFrame({
        'image_path': image_paths,
        'depth_path': depth_paths,
        'patient_id': df['patient_id'],
        'date': df['date'],
        'lesion_id': df['lesion_id'],
        'volume_mm3': df['volume_mm3'],
        'reduction_rate_percent': df['reduction_rate_percent']
    })
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 学習・検証用に分割
    train_split = 0.8
    n_train = int(len(index_df) * train_split)
    
    train_df = index_df.iloc[:n_train]
    val_df = index_df.iloc[n_train:]
    
    # CSVファイルに保存
    train_path = Path(output_dir) / "index_train.csv"
    val_path = Path(output_dir) / "index_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Created {train_path} with {len(train_df)} records")
    print(f"Created {val_path} with {len(val_df)} records")

def main():
    parser = argparse.ArgumentParser(description="Create index files from ground_truth_master.csv")
    parser.add_argument("--data_base_dir", type=str, help="Base directory containing ground_truth_master.csv")
    parser.add_argument("--output_dir", type=str, default="meta", help="Output directory for index files")
    
    args = parser.parse_args()
    
    # 環境変数からデータベースディレクトリを取得
    load_dotenv()
    data_base_dir = args.data_base_dir or os.getenv('DATA_BASE_DIR')
    
    if not data_base_dir:
        raise ValueError("DATA_BASE_DIR environment variable not set. Use --data_base_dir or set DATA_BASE_DIR")
    
    create_index_from_ground_truth_master(data_base_dir, args.output_dir)

if __name__ == "__main__":
    main()

