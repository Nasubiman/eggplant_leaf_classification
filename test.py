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


def build_model(model_name, num_classes):
    """train2.py と同じ構造でモデルを構築する。"""
    if model_name == "resnet50":
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "densenet201":
        model = models.densenet201()
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes),
        )
    elif model_name == "convnext_base":
        model = models.convnext_base()
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[2].in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def load_model(model_dir):
    """チェックポイントからモデルを復元する。"""
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()

    val_acc = checkpoint.get("val_acc", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    epoch = checkpoint.get("epoch", 0)
    print(f"  ✅ {model_name} (Epoch {epoch}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f})")
    return model, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dirs", type=str, nargs="+",
        default=[
            "./resnet50-eggplant",
            "./densenet201-eggplant",
            "./convnext_base-eggplant",
        ],
        help="アンサンブルするモデルのディレクトリ（スペース区切りで複数指定）"
    )
    parser.add_argument(
        "--method", type=str, default="soft",
        choices=["soft", "hard"],
        help="soft: Softmax確率の平均, hard: 多数決"
    )
    args = parser.parse_args()

    # 1. データセットの読み込み
    dataset_dict = load_from_disk("./eggplant_dataset")
    test_ds = dataset_dict["test"]

    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    id_to_class = {v: k for k, v in class_mapping.items()}

    # 2. 画像の前処理
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    # 3. モデルの読み込み
    print(f"=== アンサンブルモデル読み込み ({len(args.model_dirs)}モデル) ===")
    ensemble_models = []
    model_names = []
    for model_dir in args.model_dirs:
        if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
            print(f"  ⚠️ {model_dir} にモデルが見つかりません。スキップします。")
            continue
        model, name = load_model(model_dir)
        ensemble_models.append(model)
        model_names.append(name)

    if len(ensemble_models) < 2:
        print("エラー: アンサンブルには2つ以上のモデルが必要です。")
        return

    print(f"\nアンサンブル方式: {'Soft Voting (確率平均)' if args.method == 'soft' else 'Hard Voting (多数決)'}")
    print(f"モデル数: {len(ensemble_models)}")

    # 4. 推論の実行
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

        if args.method == "soft":
            # Soft Voting: 各モデルの Softmax 確率を平均
            avg_probs = torch.zeros(len(images), NUM_CLASSES).cuda()
            with torch.no_grad():
                for model in ensemble_models:
                    with torch.amp.autocast("cuda"):
                        logits = model(pixel_values)
                    probs = torch.softmax(logits, dim=-1)
                    avg_probs += probs
            avg_probs /= len(ensemble_models)
            preds = avg_probs.argmax(dim=-1).cpu().tolist()
        else:
            # Hard Voting: 各モデルの予測の多数決
            all_model_preds = []
            with torch.no_grad():
                for model in ensemble_models:
                    with torch.amp.autocast("cuda"):
                        logits = model(pixel_values)
                    model_preds = logits.argmax(dim=-1).cpu()
                    all_model_preds.append(model_preds)
            # スタックして多数決
            stacked = torch.stack(all_model_preds, dim=0)  # (num_models, batch)
            preds = []
            for j in range(stacked.size(1)):
                votes = stacked[:, j].tolist()
                # 最も多い票のクラスを選択
                pred = max(set(votes), key=votes.count)
                preds.append(pred)

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

    # 5. 結果の計算
    correct = sum(p == t for p, t in zip(all_preds, all_trues))
    acc = correct / total * 100
    ensemble_label = " + ".join(model_names)
    print(f"\n=== 最終結果 ({ensemble_label}) ===")
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Time: {elapsed:.2f} seconds ({elapsed/total*1000:.1f} ms/image)")

    # 6. 混同行列と Classification Report
    class_labels = list(range(NUM_CLASSES))
    class_names_list = [id_to_class[str(i)] for i in range(NUM_CLASSES)]

    cm = confusion_matrix(all_trues, all_preds, labels=class_labels)
    print("\n=== 混同行列 ===")
    print(cm)

    print("\n=== Classification Report ===")
    report = classification_report(
        all_trues, all_preds, labels=class_labels, target_names=class_names_list
    )
    print(report)

    # 7. 混同行列を画像として保存
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix - Ensemble {args.method} (Accuracy: {acc:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    save_fig_path = f"confusion_matrix_ensemble_{args.method}.png"
    plt.savefig(save_fig_path, dpi=150)
    print(f"\n混同行列を {save_fig_path} に保存しました。")


if __name__ == "__main__":
    main()
