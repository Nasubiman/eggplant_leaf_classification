# eggplant

ナスの葉の画像から病害を判定するAIモデルのプロジェクトです。

## データセット
- **URL/Kaggle:** `sujaykapadnis/eggplant-disease-recognition-dataset`
- **保存先:** 本番環境のパスは秘匿化のため `.gitignore` されている `config.json` 内 (`dataset_dir` 等) に保存されています。
- **画像数:** 1,400枚 (Train: 980, Val: 210, Test: 210)
- **クラス (7種類):**

  | ID | クラス名 |
  |---|---|
  | 0 | Healthy Leaf（健康な葉） |
  | 1 | Insect Pest Disease（害虫） |
  | 2 | Leaf Spot Disease（斑点病） |
  | 3 | Mosaic Virus Disease（モザイク病） |
  | 4 | Small Leaf Disease（小葉病） |
  | 5 | White Mold Disease（白絹病） |
  | 6 | Wilt Disease（萎凋病） |

## モデルと学習の方向性
- **使用モデル:** `google/paligemma2-3b-pt-224` (PaliGemma2 3B パラメータモデル)
- **タスク設計:** 画像分類（1枚の画像に対して 0〜6 のクラスIDを出力する）
- **手法:** QLoRA (4bit量子化 + Low-Rank Adaptation) でファインチューニング
- **推論時の工夫:** Constrained Decoding（0〜6 のトークンのみ出力可能に制限）
- **GPU:** NVIDIA RTX 2080 Ti (11GB VRAM) × 2

## 学習結果

### 学習設定
| 設定 | 値 |
|---|---|
| データ | 元画像のみ (Train: 980枚) |
| 学習率 | 2e-4 |
| 実質バッチサイズ | 8 (batch=1 × accum=8) |
| Epoch数 | 3 |
| 学習時間 | 約32分 |

### Eval Loss 推移
| Epoch | Eval Loss |
|---|---|
| 1 | 0.1676 |
| 2 | 0.2537 |
| 3 | **0.1200** ← 最良 |

### テスト結果 (テストデータ 210枚)
- **Accuracy: 94.29% (198/210)**

### Classification Report
| クラス | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy Leaf | 0.95 | 0.97 | 0.96 | 36 |
| Insect Pest Disease | 1.00 | 0.93 | 0.96 | 27 |
| Leaf Spot Disease | 0.88 | 0.93 | 0.90 | 30 |
| Mosaic Virus Disease | 0.93 | 0.89 | 0.91 | 28 |
| Small Leaf Disease | 0.94 | 0.97 | 0.95 | 30 |
| White Mold Disease | 0.96 | 0.89 | 0.92 | 27 |
| Wilt Disease | 0.97 | 1.00 | 0.98 | 32 |
| **macro avg** | **0.94** | **0.94** | **0.94** | **210** |

### 混同行列
`confusion_matrix.png` を参照してください。

## ファイル構成
| ファイル | 説明 |
|---|---|
| `preprocess.py` | データセットをHugging Face datasets形式に変換し、Train/Val/Testに分割する |
| `train.py` | PaliGemma2-3BをQLoRAでファインチューニングする学習スクリプト |
| `evaluate.py` | テストデータで推論を実行し、Accuracy・混同行列・Classification Reportを出力するスクリプト |
| `class_mapping.json` | クラス名とIDの対応表 |
| `config.json` | データセットパス等の環境設定 (.gitignore対象) |
| `NOTICE` | Gemma Terms of Use に基づくライセンス情報 |
