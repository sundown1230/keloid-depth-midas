#!/usr/bin/env python3
"""
PyRenderを使用した深度レンダリングスクリプト（将来の3D復元用）
"""

import numpy as np
import pyrender
import trimesh
import cv2
from pathlib import Path
import argparse

def render_depth_from_mesh(mesh_path, camera_pose=None, image_size=(256, 256)):
    """
    メッシュから深度画像をレンダリング
    
    Args:
        mesh_path: メッシュファイルのパス
        camera_pose: カメラポーズ（4x4行列）
        image_size: 出力画像サイズ (width, height)
    
    Returns:
        depth_image: 深度画像
    """
    # メッシュを読み込み
    mesh = trimesh.load(str(mesh_path))
    
    # PyRenderシーンを作成
    scene = pyrender.Scene()
    
    # メッシュをシーンに追加
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    
    # カメラを設定
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=image_size[0] / image_size[1])
    
    # デフォルトのカメラポーズ
    if camera_pose is None:
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 1]
        ])
    
    camera_node = scene.add(camera, pose=camera_pose)
    
    # レンダラーを作成
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    
    # 深度画像をレンダリング
    color, depth = renderer.render(scene)
    
    return depth

def create_simple_mesh_from_depth(depth_image, scale=1.0):
    """
    深度画像から簡単なメッシュを作成
    
    Args:
        depth_image: 深度画像
        scale: スケールファクター
    
    Returns:
        mesh: Trimeshオブジェクト
    """
    h, w = depth_image.shape
    
    # グリッドを作成
    x = np.linspace(-1, 1, w) * scale
    y = np.linspace(-1, 1, h) * scale
    X, Y = np.meshgrid(x, y)
    
    # 深度を正規化
    Z = depth_image / depth_image.max() * scale
    
    # 頂点を作成
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 面を作成
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            # 四角形を2つの三角形に分割
            idx = i * w + j
            faces.append([idx, idx + 1, idx + w])
            faces.append([idx + 1, idx + w + 1, idx + w])
    
    faces = np.array(faces)
    
    # メッシュを作成
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh

def main():
    parser = argparse.ArgumentParser(description="Render depth from mesh using PyRender")
    parser.add_argument("input", type=str, help="Input mesh file or depth image")
    parser.add_argument("--output", type=str, help="Output depth image path")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Output image size")
    parser.add_argument("--create-mesh", action="store_true", help="Create mesh from depth image")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if args.create_mesh:
        # 深度画像からメッシュを作成
        depth_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError(f"Could not load depth image: {input_path}")
        
        mesh = create_simple_mesh_from_depth(depth_image)
        
        if args.output:
            mesh.export(args.output)
            print(f"Mesh saved to {args.output}")
        else:
            print("Mesh created successfully")
    else:
        # メッシュから深度をレンダリング
        depth_image = render_depth_from_mesh(input_path, image_size=tuple(args.size))
        
        if args.output:
            # 深度画像を保存
            depth_normalized = (depth_image / depth_image.max() * 65535).astype(np.uint16)
            cv2.imwrite(args.output, depth_normalized)
            print(f"Depth image saved to {args.output}")
        else:
            print(f"Depth image rendered: shape={depth_image.shape}, range=[{depth_image.min():.3f}, {depth_image.max():.3f}]")

if __name__ == "__main__":
    main()

