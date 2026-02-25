import os
import json
import random
from PIL import Image, ImageEnhance
from datasets import Dataset, DatasetDict
from datasets import Image as HFImage

random.seed(42)

# 拡張画像の保存先ディレクトリ
AUG_DIR = "./augmented_images"


def augment_and_save(img_path, class_name, idx):
    """1枚の画像に対して水平反転 + 明度/コントラスト調整の拡張画像を生成し、
    ディスクに保存してファイルパスを返す。
    """
    img = Image.open(img_path).convert("RGB")
    saved_paths = []

    class_dir = os.path.join(AUG_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # 拡張1: 水平反転
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_path = os.path.join(class_dir, f"{idx}_flip.jpg")
    flipped.save(flip_path, quality=95)
    saved_paths.append(flip_path)

    # 拡張2: 明度・コントラストをランダムに少しだけ変化させる
    brightness_factor = random.uniform(0.8, 1.2)  # ±20%
    contrast_factor = random.uniform(0.8, 1.2)     # ±20%
    adjusted = ImageEnhance.Brightness(img).enhance(brightness_factor)
    adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast_factor)
    adj_path = os.path.join(class_dir, f"{idx}_bc.jpg")
    adjusted.save(adj_path, quality=95)
    saved_paths.append(adj_path)

    return saved_paths


def main():
    # 1. config.jsonからデータセットのパスを取得
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_dir = config["dataset_dir"]

    # 2. クラスフォルダをスキャンしてIDを割り当てる
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_id = {name: str(i) for i, name in enumerate(class_names)}

    print("=== クラスとIDのマッピング ===")
    for name, cid in class_to_id.items():
        print(f"  ID {cid}: {name}")

    # 3. 全ての画像パスとラベルを収集
    all_images = []  # (image_path, class_name)
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                all_images.append((img_path, class_name))

    # シャッフルしてから分割（再現性のためseed固定済み）
    random.shuffle(all_images)

    # 4. Train(70%), Val(15%), Test(15%) に分割
    n = len(all_images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_items = all_images[:n_train]
    val_items = all_images[n_train:n_train + n_val]
    test_items = all_images[n_train + n_val:]

    print(f"\n--- 分割結果 (拡張前) ---")
    print(f"  Train: {len(train_items)} 枚")
    print(f"  Val:   {len(val_items)} 枚")
    print(f"  Test:  {len(test_items)} 枚")

    # 5. 拡張画像の生成 (Trainのみ) → ディスクに保存してパスを収集
    print("\n拡張画像を生成中...")
    train_data = {"image": [], "prompt": [], "label": []}
    prompt = "disease type"

    for idx, (img_path, class_name) in enumerate(train_items):
        label = class_to_id[class_name]

        # 元画像（パスだけ保存）
        train_data["image"].append(img_path)
        train_data["prompt"].append(prompt)
        train_data["label"].append(label)

        # 拡張画像（ディスクに保存し、パスだけ保存）
        aug_paths = augment_and_save(img_path, class_name, idx)
        for aug_path in aug_paths:
            train_data["image"].append(aug_path)
            train_data["prompt"].append(prompt)
            train_data["label"].append(label)

    print(f"  拡張画像の生成完了: {len(train_data['image'])} 枚 (元{len(train_items)}枚 × 3)")

    # 6. Val/Test はパスのみ（拡張なし）
    def build_data_no_aug(items):
        data = {"image": [], "prompt": [], "label": []}
        for img_path, class_name in items:
            data["image"].append(img_path)
            data["prompt"].append(prompt)
            data["label"].append(class_to_id[class_name])
        return data

    val_data = build_data_no_aug(val_items)
    test_data = build_data_no_aug(test_items)

    # 7. Hugging Face Dataset の作成（パスを Image 型にキャスト → 遅延読み込み）
    train_ds = Dataset.from_dict(train_data).cast_column("image", HFImage())
    val_ds = Dataset.from_dict(val_data).cast_column("image", HFImage())
    test_ds = Dataset.from_dict(test_data).cast_column("image", HFImage())

    dataset_dict = DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    })

    # 8. ディスクに保存
    save_path = "./eggplant_dataset_aug"
    dataset_dict.save_to_disk(save_path)

    print(f"\n=== データセット作成完了 (Data Augmentation 適用版) ===")
    print(f"保存先: {save_path}")
    print(f"Train件数: {len(dataset_dict['train'])} 件")
    print(f"Val件数:   {len(dataset_dict['val'])} 件")
    print(f"Test件数:  {len(dataset_dict['test'])} 件")


if __name__ == "__main__":
    main()
