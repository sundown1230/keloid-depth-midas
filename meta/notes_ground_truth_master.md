# Ground Truth Master CSV の構造

## 想定される列構造
- `patient_id`: 患者ID
- `date`: 撮影日 (yyyymmdd形式)
- `lesion_id`: 病変ID
- `volume_mm3`: 体積 (mm³)
- `reduction_rate_percent`: 前後縮小率 (%)

## データパス構造
```
02_data_processed/prospective/<patient_id>/<yyyymmdd>/<lesion_id>_<yyyymmdd>_photo.jpg
```

## 注意事項
- Vectra由来の擬似深度（.npy）は任意
- 利用可能なら学習に使用し、無ければインデックスは空欄にしてスキップ

