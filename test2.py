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
from torchvision import transforms as T, models
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

NUM_CLASSES = 7


def build_model(model_name, num_classes, weights=None):
    """train2.py と同じ構造でモデルを構築する。"""
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.heads[0].in_features, num_classes),
        )
    elif model_name == "swin_t":
        model = models.swin_t(weights=weights)
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.head.in_features, num_classes),
        )
    elif model_name == "swin_b":
        model = models.swin_b(weights=weights)
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.head.in_features, num_classes),
        )
    elif model_name == "resnext50":
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes),
        )
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[2].in_features, num_classes),
        )
    elif model_name == "convnext_base":
        model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[2].in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./resnet50-eggplant",
        help="学習済みモデルのディレクトリ"
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    checkpoint_path = os.path.join(model_dir, "best_model.pt")

    # 1. チェックポイントの読み込み
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    print(f"Model: {model_name}, Best Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")

    # 2. データセットの読み込み
    dataset_dict = load_from_disk("./eggplant_dataset")
    test_ds = dataset_dict["test"]

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    id_to_class = {v: k for k, v in class_mapping.items()}

    # 3. 画像の前処理（torchvision 標準）
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    # 4. モデルの再構築と重みの読み込み
    model = build_model(model_name, num_classes, weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    print("Model loaded successfully.")

    # 5. 推論の実行
    all_preds = []
    all_trues = []
    total = len(test_ds)
    print(f"\n=== テストデータ {total}件で推論開始 ===")

    start_time = time.time()

    batch_size = 32
    for i in tqdm(range(0, total, batch_size)):
        batch = test_ds[i : i + batch_size]
        images = [img.convert("RGB") for img in batch["image"]]
        labels = [int(lbl) for lbl in batch["label"]]

        pixel_values = torch.stack([test_transform(img) for img in images]).cuda()

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
        title=f"Confusion Matrix - {model_name} (Accuracy: {acc:.2f}%)",
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
    save_fig_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(save_fig_path, dpi=150)
    print(f"\n混同行列を {save_fig_path} に保存しました。")


if __name__ == "__main__":
    main()
