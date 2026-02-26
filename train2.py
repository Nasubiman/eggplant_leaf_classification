import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms as T
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

NUM_CLASSES = 7


class SigLIP2Classifier(nn.Module):
    """SigLIP2 の Vision Encoder + Linear 分類ヘッド。
    画像を入力として受け取り、各クラスの logits を出力する。
    """
    def __init__(self, vision_encoder, hidden_size, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        # pooler_output があればそれを使う (DINOv3, SigLIP 等)
        # なければ patch embeddings の平均を使う
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(features)
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Vision モデルのHugging Face ID (SigLIP2, DINOv2, DINOv3 等)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_backbone", type=float, default=3e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    args = parser.parse_args()

    model_id = args.model_id
    # モデル名から保存ディレクトリを自動生成 (e.g., "siglip2-base-patch16-224")
    model_short_name = model_id.split("/")[-1]
    output_dir = f"./{model_short_name}-eggplant"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"Output: {output_dir}")

    # 1. データセットの読み込み
    dataset_dict = load_from_disk("./eggplant_dataset")
    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["val"]
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # 2. Image Processor の読み込み
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    # 3. Vision Encoder の読み込み (AutoModel で SigLIP2 / DINOv3 両方に対応)
    #    fp32 で読み込み、mixed precision は autocast に任せる（fp16直読みは NaN の原因）
    full_model = AutoModel.from_pretrained(model_id)
    # SigLIP の場合は .vision_model 、DINOv3 の場合はモデル直接
    if hasattr(full_model, "vision_model"):
        vision_encoder = full_model.vision_model
    else:
        vision_encoder = full_model
    hidden_size = vision_encoder.config.hidden_size
    print(f"Vision Encoder hidden_size: {hidden_size}")

    # 4. 分類モデルの構築
    model = SigLIP2Classifier(vision_encoder, hidden_size, NUM_CLASSES)

    # 5. LoRA を Vision Encoder に適用
    #    モデルによってアテンション層の名前が異なるため自動検出する
    module_names = {name.split(".")[-1] for name, _ in vision_encoder.named_modules()}
    if "q_proj" in module_names:
        # SigLIP2 系
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    elif "query" in module_names:
        # DINOv2 / DINOv3 系
        target_modules = ["query", "key", "value", "dense"]
    else:
        target_modules = "all-linear"
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        bias="none",
    )
    model.vision_encoder = get_peft_model(model.vision_encoder, lora_config)
    model.vision_encoder.print_trainable_parameters()

    # 分類ヘッド(classifier)は最初から学習対象
    model = model.cuda()

    # 6. DataLoader の準備
    #    学習時のみデータ拡張を適用し、過学習を抑制する
    train_augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    ])

    def train_collate_fn(batch):
        images = [train_augment(ex["image"].convert("RGB")) for ex in batch]
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        inputs = image_processor(images=images, return_tensors="pt")
        return inputs["pixel_values"], labels

    def val_collate_fn(batch):
        images = [ex["image"].convert("RGB") for ex in batch]
        labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
        inputs = image_processor(images=images, return_tensors="pt")
        return inputs["pixel_values"], labels

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=train_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=val_collate_fn, pin_memory=True,
    )

    # 7. Optimizer & Loss
    optimizer = torch.optim.AdamW([
        {"params": model.vision_encoder.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    # Label Smoothing: モデルが「100%確信」するのを防ぎ、過学習を抑制する
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()

    # 学習率を徐々に下げる (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 8. 学習ループ
    num_epochs = args.epochs
    best_val_acc = 0.0

    print(f"\n=== Start Training ({num_epochs} epochs, batch_size={args.batch_size}) ===")

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(pixel_values)
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
        scheduler.step()  # 学習率を更新

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.cuda()
                labels = labels.cuda()

                with torch.amp.autocast("cuda"):
                    logits = model(pixel_values)
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

        # --- Best モデルの保存 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_id": model_id,
                "hidden_size": hidden_size,
                "num_classes": NUM_CLASSES,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, save_path)
            print(f"  → Best model saved! (Val Acc: {val_acc:.4f})")

    print(f"\n=== Training Complete ===")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
