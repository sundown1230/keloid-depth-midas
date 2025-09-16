#!/usr/bin/env python3
"""
カメラポーズ推定スクリプト（将来の3D復元用）
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def solve_pose_from_image(image_path, camera_matrix=None, dist_coeffs=None):
    """
    画像からカメラポーズを推定
    
    Args:
        image_path: 入力画像パス
        camera_matrix: カメラ内部パラメータ（Noneの場合はデフォルト値を使用）
        dist_coeffs: 歪み係数（Noneの場合はデフォルト値を使用）
    
    Returns:
        rotation_vector: 回転ベクトル
        translation_vector: 並進ベクトル
    """
    # 画像を読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # デフォルトのカメラ内部パラメータ（必要に応じて調整）
    if camera_matrix is None:
        h, w = gray.shape
        fx = fy = max(h, w)  # 焦点距離の推定
        cx, cy = w // 2, h // 2
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
    
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    
    # 特徴点検出（例：SIFT）
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if len(keypoints) < 4:
        print(f"Warning: Not enough keypoints found in {image_path}")
        return None, None
    
    # 仮想的な3D点（平面を仮定）
    # 実際の実装では、より適切な3D点の推定が必要
    object_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.float32)
    
    # 対応する2D点（特徴点から選択）
    image_points = np.array([kp.pt for kp in keypoints[:4]], dtype=np.float32)
    
    # PnP問題を解く
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if not success:
        print(f"Warning: Could not solve pose for {image_path}")
        return None, None
    
    return rotation_vector, translation_vector

def main():
    parser = argparse.ArgumentParser(description="Solve camera pose from image")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Output file for pose data")
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # ポーズを推定
    rvec, tvec = solve_pose_from_image(image_path)
    
    if rvec is not None and tvec is not None:
        print(f"Rotation vector: {rvec.flatten()}")
        print(f"Translation vector: {tvec.flatten()}")
        
        if args.output:
            # ポーズデータを保存
            pose_data = {
                'rotation_vector': rvec.flatten(),
                'translation_vector': tvec.flatten()
            }
            np.save(args.output, pose_data)
            print(f"Pose data saved to {args.output}")
    else:
        print("Failed to solve pose")

if __name__ == "__main__":
    main()

