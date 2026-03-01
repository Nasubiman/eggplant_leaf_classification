# 前書き

「20XX年、世界は致死性の植物ウイルスにより崩壊した。かつて青々と茂っていた畑は荒野と化し、もはや我々にナスすべは無く、安全に根を張れる場所は防衛都市・那須のみとなった。

あなたは、この都市のメインゲートで検問所の監視官を務めている。押し寄せる避難民の中から、健康な個体とウイルスに侵された不健康な個体を正確に仕分け、都市内部への侵入を防ぐ。この極限の任務をこナスことができるのは、あなたしかいない。

一見すると立派な紫色のツヤを持っていても、すぐに健康だと見ナスようなことはあってはならない。ウイルスの有無を正しく判定するには、頭頂部についている葉っぱの部分を注意深く観察する必要があるのだ。ウイルスに侵された者は、例外なくこの葉の裏や付け根の部分に、微細な枯れや斑点が現れる。

少しでも兆候を見落とせば、限られた土壌は一瞬で汚染され、中にいる無実の市民をすべて死ナスことになってしまう。あなたは監視官として、深層学習の力で対象を正確に仕分け、那須の平和を守り抜かねばならない。」

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
- **使用モデル:** torchvision の ImageNet 事前学習済みモデル 3 種のアンサンブル
  - **ResNet-50** (25M パラメータ) — 残差接続ベースの標準的 CNN
  - **DenseNet-201** (20M パラメータ) — 全層の特徴を密に結合する CNN
  - **ConvNeXt-Base** (89M パラメータ) — Transformer の知見で再設計されたモダン CNN
- **タスク設計:** 画像分類（1枚の画像に対して Softmax で 7 クラスに直接分類）
- **アンサンブル方式:** Hard Voting（多数決）— 各モデルの予測を集計し最多票のクラスを採用
- **学習時の工夫:**
  - データ拡張（RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomErasing）
  - Label Smoothing (0.1)
  - Cosine Annealing LR Scheduler
  - Weight Decay (0.05)
- **GPU:** NVIDIA RTX 2080 Ti (11GB VRAM)

## 学習結果

### 学習設定
| 設定 | 値 |
|---|---|
| データ | 元画像 + データ拡張 (Train: 980枚) |
| 学習率 | 1e-4 (AdamW) |
| バッチサイズ | 32 |
| Epoch数 | 30 |
| Dropout | 0.4 |
| Weight Decay | 0.05 |
| 保存基準 | Val Loss が最小のモデル |

### 各モデルの Val 結果
| モデル | Best Epoch | Val Acc | Val Loss |
|---|---|---|---|
| ResNet-50 | 16 | 0.9476 | 0.5451 |
| DenseNet-201 | 29 | 0.9524 | 0.5411 |
| ConvNeXt-Base | 19 | 0.9524 | 0.5592 |

### テスト結果 (テストデータ 210枚, アンサンブル Hard Voting)
- **Accuracy: 97.62% (205/210)**
- **推論速度: 47.3 ms/画像 (210枚を9.93秒)**

### Classification Report
| クラス | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy Leaf | 0.97 | 0.97 | 0.97 | 36 |
| Insect Pest Disease | 1.00 | 1.00 | 1.00 | 27 |
| Leaf Spot Disease | 1.00 | 0.97 | 0.98 | 30 |
| Mosaic Virus Disease | 0.97 | 1.00 | 0.98 | 28 |
| Small Leaf Disease | 0.97 | 0.97 | 0.97 | 30 |
| White Mold Disease | 0.96 | 0.93 | 0.94 | 27 |
| Wilt Disease | 0.97 | 1.00 | 0.98 | 32 |
| **macro avg** | **0.98** | **0.98** | **0.98** | **210** |

### 混同行列
`confusion_matrix_ensemble_hard.png` を参照してください。

## ファイル構成
| ファイル | 説明 |
|---|---|
| `preprocess.py` | データセットをHugging Face datasets形式に変換し、Train/Val/Testに分割する |
| `train.py` | torchvision モデルの学習スクリプト（ResNet-50, DenseNet-201, ConvNeXt-Base） |
| `test.py` | アンサンブル推論スクリプト（Soft/Hard Voting） |
| `class_mapping.json` | クラス名とIDの対応表 |
| `config.json` | データセットパス等の環境設定 (.gitignore対象) |
| `NOTICE` | ライセンス情報 |

# 後書き

監視官であるあなたは、ResNet-50, DenseNet-201, ConvNeXt-Baseの3つのモデルをアンサンブルすることで、ナスを仕分けすることにした．

だが．作ったモデルの精度が100%ではなかったため，病気のナスを都市内部に侵入させてしまい．ナスすべなく滅びたとさ．

ちゃんちゃん