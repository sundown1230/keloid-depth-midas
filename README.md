# keloid-depth-midas

既存データリポジトリの
`02_data_processed/prospective/<patient_id>/<yyyymmdd>/<lesion_id>_<yyyymmdd>_photo.jpg`
および `ground_truth_master.csv` を参照して、MiDaS微調整→単眼深度→3D体積（mm³）と前後縮小率(%)の評価基盤を構築する最小実装。

## 0. 前提
- 画像は**GitHubにコミットしない**。別プロジェクトに保管済み。
- 実行時は環境変数 `DATA_BASE_DIR` で別プロジェクトの**ルート**を指す（"C:\Users\pasca\Documents\project_keloid\ground_truth_master.csv"）。
  - その直下に `ground_truth_master.csv` と `02_data_processed/` が存在する想定。
- Colabでは Google Drive をマウントして同等に設定。

