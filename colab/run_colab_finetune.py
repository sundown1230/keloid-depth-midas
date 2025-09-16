#!/usr/bin/env python3
"""
Colab環境でのMiDaS微調整実行スクリプト
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse

def setup_colab_environment():
    """
    Colab環境をセットアップ
    """
    print("Setting up Colab environment...")
    
    # Google Driveをマウント
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except ImportError:
        print("Warning: Not running in Colab environment")
    
    # 必要なディレクトリを作成
    os.makedirs('/content/drive/MyDrive/keloid_depth_outputs/checkpoints', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/keloid_depth_outputs/logs', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/keloid_depth_outputs/samples', exist_ok=True)
    
    print("Output directories created")

def install_dependencies():
    """
    依存関係をインストール
    """
    print("Installing dependencies...")
    
    # requirements.txtから依存関係をインストール
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                   check=True)
    
    print("Dependencies installed successfully")

def run_training(config_file, resume_checkpoint=None):
    """
    学習を実行
    
    Args:
        config_file: 設定ファイルのパス
        resume_checkpoint: 再開するチェックポイントのパス
    """
    print(f"Starting training with config: {config_file}")
    
    # 学習コマンドを構築
    cmd = [sys.executable, 'src/train_finetune.py', '--config', config_file]
    
    if resume_checkpoint:
        cmd.extend(['--resume', resume_checkpoint])
    
    # 学習を実行
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise

def run_inference(model_path, config_file, input_path, output_path, visualize=True):
    """
    推論を実行
    
    Args:
        model_path: モデルのチェックポイントパス
        config_file: 設定ファイルのパス
        input_path: 入力画像のパス
        output_path: 出力ディレクトリのパス
        visualize: 可視化画像を作成するか
    """
    print(f"Starting inference with model: {model_path}")
    
    # 推論コマンドを構築
    cmd = [
        sys.executable, 'src/infer_depth.py',
        '--model', model_path,
        '--config', config_file,
        '--input', input_path,
        '--output', output_path,
        '--device', 'cuda'
    ]
    
    if visualize:
        cmd.append('--visualize')
    
    # 推論を実行
    try:
        subprocess.run(cmd, check=True)
        print("Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with error: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run MiDaS fine-tuning in Colab environment")
    parser.add_argument("--mode", type=str, choices=['train', 'infer', 'both'], 
                       default='both', help="Mode to run")
    parser.add_argument("--config", type=str, 
                       default='configs/finetune_colab_gpu.yaml',
                       help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--model", type=str, 
                       default='/content/drive/MyDrive/keloid_depth_outputs/checkpoints/best_model.pth',
                       help="Path to model checkpoint for inference")
    parser.add_argument("--input", type=str, 
                       help="Path to input images for inference")
    parser.add_argument("--output", type=str, 
                       default='/content/drive/MyDrive/keloid_depth_outputs/inference_results/',
                       help="Path to output directory for inference")
    parser.add_argument("--visualize", action="store_true", 
                       help="Create visualization images")
    parser.add_argument("--skip-setup", action="store_true", 
                       help="Skip environment setup")
    
    args = parser.parse_args()
    
    # 環境変数を設定
    if 'DATA_BASE_DIR' not in os.environ:
        print("Warning: DATA_BASE_DIR environment variable not set")
        print("Please set it to point to your data repository root")
    
    # 環境セットアップ
    if not args.skip_setup:
        setup_colab_environment()
        install_dependencies()
    
    # モードに応じて実行
    if args.mode in ['train', 'both']:
        run_training(args.config, args.resume)
    
    if args.mode in ['infer', 'both']:
        if not args.input:
            print("Error: --input is required for inference mode")
            return
        
        run_inference(args.model, args.config, args.input, args.output, args.visualize)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()

