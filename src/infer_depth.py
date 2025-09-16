"""
深度推定用の推論スクリプト
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
from dotenv import load_dotenv
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train_finetune import MiDaSFinetuner
from utils import load_checkpoint

class DepthInference:
    """
    深度推定用の推論クラス
    """
    
    def __init__(self, model_path, config_path, device='cuda'):
        """
        Args:
            model_path: モデルのチェックポイントパス
            config_path: 設定ファイルのパス
            device: デバイス
        """
        self.device = torch.device(device)
        
        # 設定ファイルを読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # モデルを作成
        self.model = MiDaSFinetuner(
            model_name=self.config['model']['name'],
            pretrained=False,  # 推論時は事前学習済みモデルは不要
            freeze_backbone=False
        ).to(self.device)
        
        # チェックポイントを読み込み
        checkpoint = load_checkpoint(model_path, self.model, None, None)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 前処理の変換を設定
        self.transform = A.Compose([
            A.Resize(self.config['data']['image_size'][0], self.config['data']['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path):
        """
        画像を前処理
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            processed_image: 前処理済み画像テンソル
            original_image: 元の画像
        """
        # 画像を読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 前処理を適用
        transformed = self.transform(image=image)
        processed_image = transformed['image'].unsqueeze(0).to(self.device)
        
        return processed_image, original_image
    
    def predict_depth(self, image_path):
        """
        単一画像の深度を推定
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            depth_map: 深度マップ
            volume: 体積
            reduction_rate: 縮小率
            original_image: 元の画像
        """
        # 画像を前処理
        processed_image, original_image = self.preprocess_image(image_path)
        
        # 推論
        with torch.no_grad():
            pred_depth, pred_volume, pred_reduction = self.model(processed_image)
        
        # 深度マップをCPUに移動してnumpy配列に変換
        depth_map = pred_depth.squeeze().cpu().numpy()
        volume = pred_volume.squeeze().cpu().numpy()
        reduction_rate = pred_reduction.squeeze().cpu().numpy()
        
        return depth_map, volume, reduction_rate, original_image
    
    def predict_batch(self, image_paths):
        """
        複数画像の深度を一括推定
        
        Args:
            image_paths: 画像ファイルのパスのリスト
        
        Returns:
            results: 結果のリスト
        """
        results = []
        
        for image_path in image_paths:
            try:
                depth_map, volume, reduction_rate, original_image = self.predict_depth(image_path)
                results.append({
                    'image_path': str(image_path),
                    'depth_map': depth_map,
                    'volume': volume,
                    'reduction_rate': reduction_rate,
                    'original_image': original_image,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def save_depth_map(self, depth_map, output_path, normalize=True):
        """
        深度マップを保存
        
        Args:
            depth_map: 深度マップ
            output_path: 出力パス
            normalize: 正規化するかどうか
        """
        if normalize:
            # 深度マップを0-255の範囲に正規化
            depth_normalized = ((depth_map - depth_map.min()) / 
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = depth_map.astype(np.uint8)
        
        cv2.imwrite(str(output_path), depth_normalized)
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_JET):
        """
        深度マップを可視化
        
        Args:
            depth_map: 深度マップ
            colormap: カラーマップ
        
        Returns:
            colored_depth: カラー化された深度マップ
        """
        # 深度マップを0-255の範囲に正規化
        depth_normalized = ((depth_map - depth_map.min()) / 
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        # カラーマップを適用
        colored_depth = cv2.applyColorMap(depth_normalized, colormap)
        
        return colored_depth

def main():
    parser = argparse.ArgumentParser(description="Infer depth from images using trained MiDaS model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, help="Path to output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--visualize", action="store_true", help="Create visualization images")
    
    args = parser.parse_args()
    
    # 環境変数を読み込み
    load_dotenv()
    
    # 推論クラスを作成
    inferencer = DepthInference(args.model, args.config, args.device)
    
    # 入力パスを処理
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 単一ファイル
        image_paths = [input_path]
    elif input_path.is_dir():
        # ディレクトリ内の画像ファイルを検索
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"**/*{ext}"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    print(f"Found {len(image_paths)} images")
    
    # 出力ディレクトリを作成
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # 推論を実行
    results = inferencer.predict_batch(image_paths)
    
    # 結果を保存
    for result in results:
        if result['success']:
            image_path = Path(result['image_path'])
            
            # 深度マップを保存
            if output_dir:
                depth_output_path = output_dir / f"{image_path.stem}_depth.png"
                inferencer.save_depth_map(result['depth_map'], depth_output_path)
            
            # 可視化画像を作成
            if args.visualize and output_dir:
                colored_depth = inferencer.visualize_depth(result['depth_map'])
                vis_output_path = output_dir / f"{image_path.stem}_depth_vis.png"
                cv2.imwrite(str(vis_output_path), colored_depth)
            
            # 結果を表示
            print(f"Image: {image_path.name}")
            print(f"  Volume: {result['volume']:.2f} mm³")
            print(f"  Reduction Rate: {result['reduction_rate']:.2f}%")
            print(f"  Depth Range: [{result['depth_map'].min():.3f}, {result['depth_map'].max():.3f}]")
            print()
        else:
            print(f"Error processing {result['image_path']}: {result['error']}")
    
    print("Inference completed!")

if __name__ == "__main__":
    main()

