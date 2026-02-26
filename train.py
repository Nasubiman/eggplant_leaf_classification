import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms as T, models
from tqdm import tqdm

NUM_CLASSES = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet201",
                        choices=["resnet50", "densenet201", "convnext_base"],
                        help="torchvision のモデル名")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="ImageNet 事前学習済み重みを使う")
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained",
                        help="ランダム初期化から学習する")
    parser.add_argument("--freeze_ratio", type=float, default=0.0,
                        help="バックボーンの凍結割合 (0.0=全層学習, 0.5=前半凍結, 0.75=前3/4凍結)")
    args = parser.parse_args()

    output_dir = f"./{args.model}-eggplant"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model: {args.model} (pretrained={args.pretrained})")
    print(f"Output: {output_dir}")

    # 1. データセットの読み込み
    dataset_dict = load_from_disk("./eggplant_dataset")
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["val"]
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # 2. 画像の前処理（torchvision 標準の ImageNet 正規化）
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    def train_collate_fn(batch):
        images = torch.stack([train_transform(ex["image"].convert("RGB")) for ex in batch])
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        return images, labels

    def val_collate_fn(batch):
        images = torch.stack([val_transform(ex["image"].convert("RGB")) for ex in batch])
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        return images, labels

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=train_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=val_collate_fn, pin_memory=True,
    )

    # 3. モデルの構築
    weights = "DEFAULT" if args.pretrained else None

    if args.model == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
    elif args.model == "densenet201":
        model = models.densenet201(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier.in_features, NUM_CLASSES),
        )
    elif args.model == "convnext_base":
        model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[2].in_features, NUM_CLASSES),
        )

    # バックボーンの一部を凍結して過学習を抑制
    if args.freeze_ratio > 0:
        all_params = list(model.parameters())
        freeze_count = int(len(all_params) * args.freeze_ratio)
        for param in all_params[:freeze_count]:
            param.requires_grad = False
        print(f"Frozen: {freeze_count}/{len(all_params)} params ({args.freeze_ratio*100:.0f}%)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    model = model.cuda()

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. 学習ループ
    best_val_loss = float("inf")
    print(f"\n=== Start Training ({args.epochs} epochs, batch_size={args.batch_size}) ===")

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{train_correct/train_total:.4f}",
            )

        train_loss /= train_total
        train_acc = train_correct / train_total
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()

                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        # --- Best モデルの保存 (Val Loss が最小のモデル) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "pretrained": args.pretrained,
                "num_classes": NUM_CLASSES,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "epoch": epoch + 1,
            }, save_path)
            print(f"  → Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")

    print(f"\n=== Training Complete ===")
    print(f"Best Val Loss: {best_val_loss:.4f} (Val Acc: {best_val_acc:.4f})")
    print(f"Model saved to: {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
