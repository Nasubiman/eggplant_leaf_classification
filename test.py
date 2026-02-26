import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModel
from peft import get_peft_model, LoraConfig
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

NUM_CLASSES = 7


class SigLIP2Classifier(nn.Module):
    """SigLIP2 の Vision Encoder + Linear 分類ヘッド。"""
    def __init__(self, vision_encoder, hidden_size, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(features)
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./dinov3-vitl16-pretrain-lvd1689m-eggplant",
        help="学習済みモデルのディレクトリ"
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    checkpoint_path = os.path.join(model_dir, "best_model.pt")

    # 1. 保存されたチェックポイントの読み込み
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_id = checkpoint["model_id"]
    hidden_size = checkpoint["hidden_size"]
    num_classes = checkpoint["num_classes"]
    print(f"Model: {model_id}, Best Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")

    # 2. データセットの読み込み (testを使う)
    dataset_dict = load_from_disk("./eggplant_dataset")
    test_ds = dataset_dict["test"]

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    id_to_class = {v: k for k, v in class_mapping.items()}

    # 3. Image Processor の読み込み
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    # 4. モデルの再構築 (学習時と同じアーキテクチャ)
    #    fp32 で読み込み（fp16直読みは NaN の原因）
    full_model = AutoModel.from_pretrained(model_id)
    if hasattr(full_model, "vision_model"):
        vision_encoder = full_model.vision_model
    else:
        vision_encoder = full_model

    # LoRA を同じ設定で適用 (重みはチェックポイントから上書きされる)
    module_names = {name.split(".")[-1] for name, _ in vision_encoder.named_modules()}
    if "q_proj" in module_names:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif "query" in module_names:
        target_modules = ["query", "key", "value", "dense"]
    else:
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=64,
        lora_alpha=256,
        target_modules=target_modules,
        bias="none",
    )
    vision_encoder = get_peft_model(vision_encoder, lora_config)

    model = SigLIP2Classifier(vision_encoder, hidden_size, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    print("Model loaded successfully.")

    # 5. 推論の実行
    all_preds = []
    all_trues = []
    total = len(test_ds)
    print(f"\n=== テストデータ {total}件で推論開始 ===")

    start_time = time.time()

    # バッチ推論 (PaliGemma と違い、1回のforward passで瞬時に分類できる)
    batch_size = 32
    for i in tqdm(range(0, total, batch_size)):
        batch = test_ds[i : i + batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        labels = [int(lbl) for lbl in batch["label"]]

        inputs = image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].cuda()

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = model(pixel_values)
            preds = logits.argmax(dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_trues.extend(labels)

        # 最初のバッチだけ詳細表示
        if i == 0:
            for j in range(min(10, len(preds))):
                pred_str = str(preds[j])
                true_str = str(labels[j])
                pred_class = id_to_class.get(pred_str, "Unknown")
                true_class = id_to_class.get(true_str, "Unknown")
                mark = "✅" if preds[j] == labels[j] else "❌"
                print(f"[{mark}] Pred: {pred_str}({pred_class}) | True: {true_str}({true_class})")

    elapsed = time.time() - start_time

    # 6. 結果の計算
    correct = sum(p == t for p, t in zip(all_preds, all_trues))
    acc = correct / total * 100
    print(f"\n=== 最終結果 ===")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Time: {elapsed:.2f} seconds ({elapsed/total*1000:.1f} ms/image)")

    # 7. 混同行列と Classification Report
    class_labels = list(range(num_classes))
    class_names_list = [id_to_class[str(i)] for i in range(num_classes)]

    cm = confusion_matrix(all_trues, all_preds, labels=class_labels)
    print("\n=== 混同行列 ===")
    print(cm)

    print("\n=== Classification Report ===")
    report = classification_report(
        all_trues, all_preds, labels=class_labels, target_names=class_names_list
    )
    print(report)

    # 8. 混同行列を画像として保存
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix (Accuracy: {acc:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    model_short = model_id.split('/')[-1]
    save_fig_path = f"confusion_matrix_{model_short}.png"
    plt.savefig(save_fig_path, dpi=150)
    print(f"\n混同行列を {save_fig_path} に保存しました。")


if __name__ == "__main__":
    main()
