# 前書き

「20XX年、世界は致死性の植物ウイルスにより崩壊した。かつて青々と茂っていた畑は荒野と化し、我々ナス族が安全に根を張れる場所は防衛都市・ナスカのみとなった。

あなたは、この都市のメインゲートで検問所の監視官を務めている。押し寄せる避難民の中から、健康なナスとウイルスに侵された不健康なナスを正確に仕分け、都市内部への侵入を防ぐことがあなたの任務だ。

一見すると立派な紫色のツヤを持っていても、決して騙されてはいけない。ウイルスの有無を正しく見分けるには、頭頂部についている葉っぱの部分を注意深く観察する必要があるのだ。ウイルスに侵された者は、例外なくこの葉の裏や付け根の部分に、微細な枯れや斑点が現れる。

少しでも兆候を見落とせば、限られた土壌は一瞬で汚染され、同胞たちは全滅してしまう。あなたは監視官として、深層学習を用いてナスを正確に仕分け、ナスカの平和を守らなければならない。」

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
- **使用モデル:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3 ViT-Large, 0.3Bパラメータ)
- **タスク設計:** 画像分類（1枚の画像に対して Softmax で 7 クラスに直接分類）
- **手法:** LoRA (Low-Rank Adaptation, r=64) + 分類ヘッド（LayerNorm → Dropout → Linear）
- **学習時の工夫:**
  - データ拡張（RandomHorizontalFlip, ColorJitter）
  - Label Smoothing (0.1)
  - Cosine Annealing LR Scheduler
  - Vision Encoder と分類ヘッドで学習率を分離
- **GPU:** NVIDIA RTX 2080 Ti (11GB VRAM) × 2

## 学習結果

### 学習設定
| 設定 | 値 |
|---|---|
| データ | 元画像 + データ拡張 (Train: 980枚) |
| 学習率 | Vision Encoder: 3e-5, 分類ヘッド: 1e-4 |
| バッチサイズ | 8 |
| Epoch数 | 20 |
| LoRA | r=64, alpha=256 |
| 学習時間 | 約5分 |

### Val Acc 推移（抜粋）
| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 0.3878 | 0.7000 | 1.1737 |
| 5 | 0.9653 | 0.9476 | 0.5986 |
| 9 | 0.9949 | 0.9619 | 0.5588 |
| 15 | 1.0000 | **0.9667** | **0.5447** ← 最良 |
| 20 | 1.0000 | 0.9667 | 0.5458 |

### テスト結果 (テストデータ 210枚)
- **Accuracy: 98.10% (206/210)**
- **推論速度: 11.5 ms/画像 (210枚を2.42秒)**

### Classification Report
| クラス | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy Leaf | 0.97 | 0.97 | 0.97 | 36 |
| Insect Pest Disease | 1.00 | 1.00 | 1.00 | 27 |
| Leaf Spot Disease | 0.97 | 0.93 | 0.95 | 30 |
| Mosaic Virus Disease | 0.93 | 1.00 | 0.97 | 28 |
| Small Leaf Disease | 1.00 | 1.00 | 1.00 | 30 |
| White Mold Disease | 1.00 | 0.96 | 0.98 | 27 |
| Wilt Disease | 1.00 | 1.00 | 1.00 | 32 |
| **macro avg** | **0.98** | **0.98** | **0.98** | **210** |

### 混同行列
`confusion_matrix_dinov3-vitl16-pretrain-lvd1689m.png` を参照してください。

## ファイル構成
| ファイル | 説明 |
|---|---|
| `preprocess.py` | データセットをHugging Face datasets形式に変換し、Train/Val/Testに分割する |
| `train.py` | DINOv3をLoRAでファインチューニングする学習スクリプト |
| `test.py` | テストデータで推論を実行し、Accuracy・混同行列・Classification Reportを出力するスクリプト |
| `class_mapping.json` | クラス名とIDの対応表 |
| `config.json` | データセットパス等の環境設定 (.gitignore対象) |
| `NOTICE` | DINOv3 ライセンス情報 |
