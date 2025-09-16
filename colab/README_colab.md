# Colab環境での実行方法

## 1. Google Driveのマウント

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. 環境変数の設定

```python
import os
# データベースディレクトリを設定（Google Drive内のパス）
os.environ['DATA_BASE_DIR'] = '/content/drive/MyDrive/your-data-repo-root'
```

## 3. プロジェクトのクローン

```python
# プロジェクトをクローン（またはアップロード）
!git clone https://github.com/your-username/keloid-depth-midas.git
# または、Google Driveからプロジェクトをコピー
```

## 4. 依存関係のインストール

```python
!pip install -r keloid-depth-midas/requirements.txt
```

## 5. データインデックスの生成

```python
# ground_truth_master.csvからインデックスファイルを生成
!python keloid-depth-midas/scripts/index_from_ground_truth_master.py
```

## 6. 学習の実行

```python
# Colab GPU設定で学習を実行
!python keloid-depth-midas/src/train_finetune.py --config keloid-depth-midas/configs/finetune_colab_gpu.yaml
```

## 7. 推論の実行

```python
# 学習済みモデルで推論を実行
!python keloid-depth-midas/src/infer_depth.py \
    --model /content/drive/MyDrive/keloid_depth_outputs/checkpoints/best_model.pth \
    --config keloid-depth-midas/configs/finetune_colab_gpu.yaml \
    --input /content/drive/MyDrive/your-data-repo-root/02_data_processed/prospective/ \
    --output /content/drive/MyDrive/keloid_depth_outputs/inference_results/ \
    --visualize
```

## 注意事項

- Google Driveのマウントには時間がかかる場合があります
- 大きなデータセットの場合、Colabの制限に注意してください
- 学習中は定期的にチェックポイントを保存することを推奨します
- 推論結果はGoogle Driveに保存されます

