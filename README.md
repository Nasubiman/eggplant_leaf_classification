# eggplant

## データセット
- **URL/Kaggle:** `sujaykapadnis/eggplant-disease-recognition-dataset`
- **保存先:** 本番環境のパスは秘匿化のため `.gitignore` されている `config.json` 内 (`dataset_dir` 等) に保存されています。
- **クラス (7種類):**
  - Healthy Leaf
  - Insect Pest Disease
  - Leaf Spot Disease
  - Mosaic Virus Disease
  - Small Leaf Disease
  - White Mold Disease
  - Wilt Disease

## モデルと学習の方向性
- **使用モデル:** `google/paligemma2-3b-pt-224` (PaliGemma2 3B パラメータモデル)
- **タスク設計:** 画像分類・キャプション生成（1枚の画像に対する病害判定を行います）。
- **目的:** 12GB VRAM等の計算資源に制約がある環境下でも、量子化（4-bit等）やLoRA（Low-Rank Adaptation）を適用することでマルチモーダルな推論・ファインチューニングの実現を目指します。
